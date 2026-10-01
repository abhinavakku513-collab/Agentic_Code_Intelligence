#!/usr/bin/env python
"""P0 DEV evaluation of the pipeline **as served** — the number the UI's "P0 evaluation" panel shows.

Every query is ranked by `AcisEngine._rank_one`, the function behind both `search()` (the API and the page) and
`search_batch()` (Mode A): routing, the candidate union, the learned ranker or its abstention, the generic route's
weighted fusion, the dense tail. Nothing here re-implements a stage, so the number cannot describe a pipeline the
page does not run. The configuration is the one `acis serve` loads (`configs/dev.yaml` by default).

**Out of fold, honestly.** The shipped ranker and the routing bank were both fitted on all 5,000 dev queries, so
scoring them on those queries would report memory. Each fold is therefore ranked with
* a ranker trained on the other four folds only (same features, hyper-parameters and seed as the shipped one), and
* a routing bank without that fold's queries — with its own query in the bank, a query is its own nearest
  neighbour and looks more statement-like than any unseen query can.

The dense channel (Mode B, the frozen encoder) is measured on the same queries, through the same function, as the
baseline; the difference is a paired bootstrap on the gate's rule.

Output: `runs/eval/p0-<sha12>/{summary.json,per_query.jsonl}` (content-addressed) and one `dev` ledger row per
system, each carrying the artifact's path and SHA-256, the encoder, its pinned commit and fingerprint, the config
and its hash, the prep hashes, the ranker and bank hashes, the split, the query count and the runtime. APPS TEST
labels are never read: `dev` is the TRAIN split (D19).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.routing import QueryBank
from acis.eval import ledger
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import K_VALUES, per_query, score_run
from acis.eval.splits import fold_members
from acis.obs.counters import Counters
from acis.rank import ltr
from acis.rank.compose import rank_derived_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltr_build import build_groups  # noqa: E402 — the same training groups the shipped ranker was fitted on

TOP_K = 100
ORDERED_BY = ("ltr.applied", "ltr.abstained", "fusion.applied", "fusion.untuned")


def plain(value: Any) -> Any:
    """Frozen config sections are read-only mappings; the ledger hashes plain JSON."""
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrated_tau(matrix: np.ndarray, *, k: int, coverage: float) -> float:
    """`scripts/bench/calibrate_route.py`'s rule: leave-one-out kNN scores of the bank's own queries, quantile."""
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    take = min(k, sims.shape[0] - 1)
    scores = np.partition(sims, -take, axis=1)[:, -take:].mean(axis=1)
    return float(np.quantile(scores, 1.0 - coverage))


def rank_of_gold(order: list[str], gold: set[str]) -> int | None:
    return next((i for i, d in enumerate(order, start=1) if d in gold), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--limit", type=int, default=0, help="first N dev queries (0 = all 5,000); a smoke run")
    parser.add_argument("--rounds", type=int, default=400, help="ranker boosting rounds (the shipped value)")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument(
        "--no-route-analysis",
        action="store_true",
        help="skip re-ranking each query under the other route (a diagnostic; with a route-sensitive second "
        "encoder it costs one extra full-length encode per query)",
    )
    args = parser.parse_args(argv)

    started_all = time.perf_counter()
    config = load_frozen_config(args.config)
    encoder = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=encoder)
    snapshot = engine.build_snapshot(apps.load_corpus(), source="serve:p0")
    data = engine.snapshot_data(snapshot)
    data.warm_features()

    ids = list(apps.dev_query_ids())
    if args.limit:
        ids = ids[: args.limit]
    wanted = set(ids)
    qrels = dev_qrels(ids)
    queries = apps.load_queries()
    folds = {qid: f for f, members in fold_members().items() for qid in members if qid in wanted}

    # Training groups for the fold rankers: the pools and features the shipped ranker was fitted on.
    print(f"building {len(ids)} training groups …", flush=True)
    groups, _, _, _ = build_groups(
        engine,
        snapshot,
        ids,
        dense_k=int(config.get("retrieve.dense_k", 100)),
        lexical_k=int(config.get("retrieve.lexical_k", 30)),
        cap=int(config.get("retrieve.union_cap", 100)),
    )
    by_query = {g.query_id: g for g in groups}

    bank_path = acis_root() / str(config.get("route.bank"))
    full_bank = engine.query_bank
    bank_ids = list(apps.dev_query_ids())  # `calibrate_route.py` builds the bank in this order
    ranker_path = acis_root() / str(config.get("rank.model"))

    base_config = config
    shipped = json.loads((acis_root() / "artifacts" / "route" / "route_calibration.json").read_text("utf-8"))
    coverage = float(shipped.get("coverage_target", 0.95))
    tau_by_fold: dict[int, float] = {}
    records: list[dict[str, Any]] = []
    run_full: dict[str, dict[str, float]] = {}
    run_dense: dict[str, dict[str, float]] = {}
    run_forced: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    started = time.perf_counter()
    for fold in sorted(set(folds.values())):
        members = [q for q in ids if folds[q] == fold]
        training = [by_query[q] for q in ids if folds[q] != fold]
        engine._ranker = ltr.train(training, rounds=args.rounds, seed=0)
        engine._ranker_loaded = True
        if full_bank is not None:
            keep = np.array([folds.get(q) != fold for q in bank_ids])
            engine._bank = QueryBank(matrix=full_bank.matrix[keep], k=full_bank.k)
            engine._bank_loaded = True
            # τ as `calibrate_route.py` sets it, re-applied to this fold's bank: the share of the bank's own
            # queries (leave-one-out) that must qualify. Keeping the full-bank τ against a bank 20 % smaller would
            # route more held-out statements generic than deployment does (every OOD score drops).
            tau_by_fold[fold] = calibrated_tau(engine._bank.matrix, k=full_bank.k, coverage=coverage)
            engine.config = base_config.with_overrides(**{"route.tau": tau_by_fold[fold]})
        for qid in members:
            text = queries[qid]
            gold = {d for d, rel in qrels[qid].items() if rel > 0}
            request = Counters()
            normalised, _ = engine.normalise_query(text)
            decision = engine.route_decision(normalised)
            full = engine._rank_one(data, text, top_k=TOP_K, strict=False, counters=request, route=decision.route)
            dense = engine._rank_one(data, text, top_k=TOP_K, strict=False, channel="dense")
            # Routing analysis: the same query under the other route — does the router cost accuracy?
            if not args.no_route_analysis:
                other = "generic" if decision.route == "statement_like" else "statement_like"
                forced = engine._rank_one(data, text, top_k=TOP_K, strict=False, counters=Counters(), route=other)
                run_forced[qid] = rank_derived_scores([d for d, _ in forced], TOP_K)
            full_ids, dense_ids = [d for d, _ in full], [d for d, _ in dense]
            run_full[qid] = rank_derived_scores(full_ids, TOP_K)
            run_dense[qid] = rank_derived_scores(dense_ids, TOP_K)
            z = engine.confidence_signal(data, normalised, full_ids, route=decision.route)
            seen = request.snapshot()
            ordered_by = next((k for k in ORDERED_BY if seen.get(k)), "dense")
            counts[f"route.{decision.route}"] = counts.get(f"route.{decision.route}", 0) + 1
            counts[f"ordered_by.{ordered_by}"] = counts.get(f"ordered_by.{ordered_by}", 0) + 1
            records.append(
                {
                    "query_id": qid,
                    "fold": fold,
                    "query_excerpt": text[:600],
                    "query_chars": len(text),
                    "gold": sorted(gold),
                    "route": decision.as_dict(),
                    "ordered_by": ordered_by,
                    "rank_full": rank_of_gold(full_ids, gold),
                    "rank_dense": rank_of_gold(dense_ids, gold),
                    "top10_full": full_ids[:10],
                    "confidence_z": None if z != z else round(z, 6),
                    "top10_dense": dense_ids[:10],
                }
            )
        print(f"fold {fold}: {len(members)} queries ({time.perf_counter() - started:.0f}s)", flush=True)
    rank_seconds = time.perf_counter() - started

    metrics_full = score_run(qrels, run_full, K_VALUES)
    metrics_dense = score_run(qrels, run_dense, K_VALUES)
    pq_full = per_query(qrels, run_full, "ndcg", 10)
    pq_dense = per_query(qrels, run_dense, "ndcg", 10)
    for record in records:
        record["ndcg_at_10_full"] = round(pq_full[record["query_id"]], 6)
        record["ndcg_at_10_dense"] = round(pq_dense[record["query_id"]], 6)
    boot = paired_bootstrap(pq_full, pq_dense)
    pq_forced = per_query(qrels, run_forced, "ndcg", 10) if run_forced else {}
    routing_analysis = {}
    for route, other in (("generic", "statement_like"), ("statement_like", "generic")) if run_forced else ():
        subset = [r["query_id"] for r in records if r["route"]["route"] == route]
        if not subset:
            continue
        served = {q: pq_full[q] for q in subset}
        forced_other = {q: pq_forced[q] for q in subset}
        result = paired_bootstrap(forced_other, served)
        routing_analysis[f"routed_{route}"] = {
            "n": len(subset),
            "ndcg_at_10_served": sum(served.values()) / len(subset),
            f"ndcg_at_10_if_routed_{other}": sum(forced_other.values()) / len(subset),
            "ndcg_at_10_dense": sum(pq_dense[q] for q in subset) / len(subset),
            "bootstrap_other_minus_served": result.as_dict(),
        }
    for record in records:
        if pq_forced:
            record["ndcg_at_10_other_route"] = round(pq_forced[record["query_id"]], 6)
    mrr_full = per_query(qrels, run_full, "mrr", 10)
    mrr_dense = per_query(qrels, run_dense, "mrr", 10)
    boot_mrr = paired_bootstrap(mrr_full, mrr_dense)

    # The artifact is content-addressed, so the ledger row can pin it and anyone can check it has not changed.
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out_dir = acis_root() / "runs" / "eval" / f"p0-{digest[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_query.jsonl").write_text(body, "utf-8")
    artifact = str((out_dir / "per_query.jsonl").relative_to(acis_root()))

    provenance = {
        "evaluation": "p0-dev-pipeline",
        "dataset": "APPS via CoIR AppsRetrieval (mteb), TRAIN split = dev; TEST labels never read (D19)",
        "split": "train",
        "decision_set": "oof_5fold_5000" if not args.limit else f"first_{len(ids)}",
        "n_queries": len(ids),
        "n_docs": snapshot.n_units,
        "config": args.config,
        "config_hash": config.config_hash,
        "encoder": encoder.name,
        "encoder_commit": getattr(getattr(encoder, "card", None), "base_commit", None),
        "model_fingerprint": engine.model_fingerprint,
        "numeric_profile": config.numeric_profile,
        "prep_hash_doc": getattr(encoder, "prep_hash", ""),
        "prep_hash_query": getattr(encoder, "query_prep_hash", ""),
        "prep": plain(config.section("prep")),
        "channel": str(config.get("run.channel")),
        "ranker": str(config.get("rank.model")),
        "ranker_sha256": sha256_file(ranker_path) if ranker_path.is_file() else None,
        "ranker_protocol": f"out of fold: 5 rankers, each trained on the other folds ({args.rounds} rounds, seed 0)",
        "route_bank": str(config.get("route.bank")),
        "route_bank_sha256": sha256_file(bank_path) if bank_path.is_file() else None,
        "route_protocol": (
            "each fold routed against the bank without that fold's queries, with tau re-derived on that bank by "
            "the shipped calibration rule (leave-one-out, coverage target)"
        ),
        "route_tau_by_fold": tau_by_fold,
        "route_tau": config.get("route.tau"),
        "route_rho": config.get("route.rho"),
        "generic_alpha": engine.generic_alpha,
        "threads": engine.threads,
        "blas_threads": engine.blas_threads,
        "artifact": artifact,
        "artifact_sha256": digest,
        "counts": counts,
        "routing_analysis": routing_analysis,
        "seconds_ranking": round(rank_seconds, 3),
        "seconds_total": round(time.perf_counter() - started_all, 3),
    }
    summary: dict[str, Any] = {
        **provenance,
        "systems": {
            "full": {"metrics": metrics_full},
            "dense": {"metrics": metrics_dense},
        },
        "bootstrap_full_vs_dense": {"ndcg_at_10": boot.as_dict(), "mrr_at_10": boot_mrr.as_dict()},
        "passes_gate_rule": boot.passes(),
    }
    if not args.no_ledger:
        for system, metrics in (("dense", metrics_dense), ("full", metrics_full)):
            row = (
                ledger.LedgerRowBuilder(kind="dev")
                .with_metrics(metrics)
                .with_fields(
                    rung=f"p0-pipeline:{system}",
                    system=system,
                    system_description=(
                        "dense channel only (frozen encoder, Mode B ranking)"
                        if system == "dense"
                        else "the served hybrid pipeline: routing, union, ranker (OOF) or fusion, dense tail"
                    ),
                    seconds=provenance["seconds_ranking"],
                    **{k: v for k, v in provenance.items() if k not in ("seconds_ranking",)},
                    **({"bootstrap_vs_dense": summary["bootstrap_full_vs_dense"]} if system == "full" else {}),
                )
                .build()
            )
            summary["systems"][system]["ledger_run_id"] = ledger.append(row).run_id
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str), "utf-8")

    print()
    print(f"{'system':<8} {'NDCG@10':>8} {'MRR@10':>8} {'R@100':>8}  ledger")
    for system in ("dense", "full"):
        m = summary["systems"][system]["metrics"]
        print(
            f"{system:<8} {100 * m['ndcg_at_10']:8.2f} {100 * m['mrr_at_10']:8.2f} {100 * m['recall_at_100']:8.2f}  "
            f"{summary['systems'][system].get('ledger_run_id', '(not recorded)')}"
        )
    print(
        f"full vs dense: Δ NDCG@10 {boot.delta:+.2f} CI [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}]  "
        f"{'PASS' if boot.passes() else 'no'}   Δ MRR@10 {boot_mrr.delta:+.2f}"
    )
    print(f"counts: {counts}")
    for name, block in routing_analysis.items():
        b = block["bootstrap_other_minus_served"]
        print(
            f"{name}: n={block['n']} served {100 * block['ndcg_at_10_served']:.2f}  other route "
            f"{100 * [v for k, v in block.items() if k.startswith('ndcg_at_10_if_routed')][0]:.2f}  "
            f"Δ(other−served) {b['delta']:+.2f} CI [{b['ci_low']:+.2f}, {b['ci_high']:+.2f}]"
        )
    print(f"artifact: {artifact}  sha256 {digest[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
