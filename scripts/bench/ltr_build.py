#!/usr/bin/env python
"""Build LTR training data, cross-fit a ranker, and measure it honestly (gates G2/G4/G5, docs/spec/02 §4).

Everything expensive has already happened by the time this runs: the encoder's vectors are in the content-
addressed cache, so this is matmuls, BM25 and parse-only features. That is deliberate — it means a ranking
experiment costs minutes instead of hours, and it is why the vector cache is keyed on content rather than on a
run id.

What it measures, and why each number is the one that decides something:

* **dense** — the frozen baseline. Every claim is relative to this.
* **fusion** — dense ∪ BM25, reciprocal-rank fused. Gate G2 asks whether the lexical channel earns its place.
* **LTR (out of fold)** — every query scored by a model that never saw it (`docs/spec/09` §2). A ranker scored on
  its own training queries reports its memory, and the difference is large enough to turn a regression into an
  apparent win.

Each comparison is a paired bootstrap against the baseline, reported with its CI, and the gate rule is the
frozen one: Δ ≥ +0.5 pt **and** the lower bound above zero.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ladder
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import K_VALUES, per_query, score_run
from acis.eval.splits import fold_members
from acis.features.query import bridge as bridge_features
from acis.features.query import extract as query_features
from acis.rank import candidates as cand
from acis.rank import ltr
from acis.rank.compose import rank_derived_scores

OUT_DIR = "runs/ltr"


def build_groups(engine: AcisEngine, snapshot: Any, query_ids: list[str], *, dense_k: int, lexical_k: int, cap: int):
    """One training group per query: the candidate union, its features, and the label from the dev qrels."""
    data = engine.snapshot_data(snapshot)
    queries = apps.load_queries()
    qrels = dev_qrels(query_ids)

    groups: list[ltr.TrainingGroup] = []
    dense_runs: dict[str, dict[str, float]] = {}
    fusion_runs: dict[str, dict[str, float]] = {}
    routes: dict[str, str] = {}

    for position, qid in enumerate(query_ids):
        text = queries[qid]
        normalised, _ = engine.normalise_query(text)
        route = engine.route(normalised)
        routes[qid] = route
        dense = engine._dense_ranking(data, normalised, route=route, k=max(dense_k, 100))
        lexical = engine._lexical_ranking(data, normalised, lexical_k)
        pool = cand.union(dense, lexical, dense_k=dense_k, lexical_k=lexical_k, cap=cap)

        qf = query_features(normalised)
        bridge = {c.doc_id: bridge_features(qf, data.features_of(c.doc_id)) for c in pool}
        meta = {
            c.doc_id: {
                "n_tokens": data.features_of(c.doc_id).n_tokens,
                "parse_ok": data.features_of(c.doc_id).parse_ok,
                "dup_cluster_size": data.duplicate_count(c.doc_id),
            }
            for c in pool
        }
        matrix = cand.feature_matrix(pool, bridge=bridge, query_tokens=qf.n_tokens, doc_meta=meta)
        gold = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        labels = np.array([1 if c.doc_id in gold else 0 for c in pool], dtype=np.int32)
        groups.append(ltr.TrainingGroup(qid, matrix, labels, tuple(c.doc_id for c in pool)))

        dense_runs[qid] = rank_derived_scores([d for d, _ in dense[:100]], 100)
        fusion_runs[qid] = rank_derived_scores(engine._rrf_order(pool)[:100], 100)
        if position and position % 250 == 0:
            print(f"  … {position}/{len(query_ids)} queries", flush=True)

    return groups, dense_runs, fusion_runs, routes


def shipped_run(
    oof: dict[str, list[tuple[str, float]]],
    groups: list[ltr.TrainingGroup],
    dense_runs: dict[str, dict[str, float]],
    routes: dict[str, str],
    *,
    rho: float = ltr.DEFAULT_RHO,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Out of fold, exactly as Mode A serves: the ranker's order only for a statement-like query whose evidence
    fired; the frozen dense order for every query routed off the ranker or on which it abstains."""
    oof_orders = oof_run(oof)
    run: dict[str, dict[str, float]] = {}
    counts = {"ranker": 0, "routed_off": 0, "abstained": 0}
    for group in groups:
        qid = group.query_id
        availability = ltr.group_availability(group.features)
        fired = sum(1 for value in availability.values() if value > 0.0) / max(1, len(ltr.GROUPS))
        if routes.get(qid) != "statement_like":
            counts["routed_off"] += 1
            run[qid] = dense_runs[qid]
        elif fired < rho:
            counts["abstained"] += 1
            run[qid] = dense_runs[qid]
        else:
            counts["ranker"] += 1
            run[qid] = oof_orders[qid]
    return run, counts


def oof_run(oof: dict[str, list[tuple[str, float]]], *, top_k: int = 100) -> dict[str, dict[str, float]]:
    """Turn out-of-fold scores into a run file, rank-derived so the harness sees an order rather than floats."""
    run: dict[str, dict[str, float]] = {}
    for qid, scored in oof.items():
        ordered = [doc for doc, _ in sorted(scored, key=lambda item: -item[1])][:top_k]
        run[qid] = rank_derived_scores(ordered, top_k)
    return run


def compare(
    name: str, run: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]], qrels
) -> dict[str, Any]:
    """Paired bootstrap against the baseline, on the metric the gate is written in.

    The per-query vector travels with the result. Comparing two *systems* later — the code tokeniser against the
    stock one, say — needs the same queries paired, and two independent deltas against a shared baseline are not
    that comparison however close their intervals look.
    """
    metrics = score_run(qrels, run, K_VALUES)
    a = per_query(qrels, run, "ndcg", 10)
    b = per_query(qrels, baseline, "ndcg", 10)
    result = paired_bootstrap(a, b)
    return {
        "name": name,
        "per_query_ndcg_at_10": {qid: round(v, 6) for qid, v in sorted(a.items())},
        "ndcg_at_10": round(metrics["ndcg_at_10"] * 100, 4),
        "mrr_at_10": round(metrics["mrr_at_10"] * 100, 4),
        "recall_at_100": round(metrics["recall_at_100"] * 100, 4),
        "delta_vs_baseline_pts": round(result.delta, 4),
        "ci": [round(result.ci_low, 4), round(result.ci_high, 4)],
        "passes_gate": bool(result.passes()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LTR training data, cross-fitting and the gate comparisons")
    parser.add_argument("--model", required=True, help="card key of the encoder whose vectors are cached")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--limit", type=int, default=0, help="first N dev queries (0 = all 5,000)")
    parser.add_argument("--tokenizer", default="stock", choices=["stock", "code"], help="lexical front end (G2)")
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--lexical-k", type=int, default=30)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--save", default="runs/ltr/ranker.txt")
    args = parser.parse_args(argv)

    config = load_frozen_config(args.config).with_overrides(
        **{"model.encoder": args.model, "lexical.tokenizer": args.tokenizer}
    )
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))

    started = time.perf_counter()
    print(f"building the snapshot ({args.model}, {args.tokenizer} tokeniser) …", flush=True)
    snapshot = engine.build_snapshot(apps.load_corpus(), source=f"ltr:{args.model}")
    print(f"  {snapshot.n_units} units in {time.perf_counter() - started:.0f}s", flush=True)

    ids = list(apps.dev_query_ids())
    if args.limit:
        ids = ids[: args.limit]
    qrels = dev_qrels(ids)

    started = time.perf_counter()
    groups, dense_runs, fusion_runs, routes = build_groups(
        engine, snapshot, ids, dense_k=args.dense_k, lexical_k=args.lexical_k, cap=args.cap
    )
    print(f"  {len(groups)} groups in {time.perf_counter() - started:.0f}s", flush=True)

    folds = {qid: fold for fold, members in fold_members().items() for qid in members if qid in set(ids)}
    started = time.perf_counter()
    oof = ltr.cross_fit(groups, folds=folds, rounds=args.rounds)
    print(f"  cross-fitted in {time.perf_counter() - started:.0f}s", flush=True)

    report = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "n_queries": len(ids),
        "config_hash": config.config_hash,
        "rows": [
            compare("dense (baseline)", dense_runs, dense_runs, qrels),
            compare("fusion (dense + BM25, RRF)", fusion_runs, dense_runs, qrels),
            compare("LTR (out of fold)", oof_run(oof), dense_runs, qrels),
        ],
    }
    shipped, shipped_counts = shipped_run(oof, groups, dense_runs, routes)
    report["rows"].append(compare("Mode A as shipped (routed, abstaining; out of fold)", shipped, dense_runs, qrels))
    report["shipped_counts"] = shipped_counts

    out = acis_root() / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model}.{args.tokenizer}.json").write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")

    print()
    print(f"{'system':<32} {'NDCG@10':>9} {'MRR@10':>9} {'R@100':>9} {'Δ vs dense':>12} {'CI':>20}  gate")
    for row in report["rows"]:
        ci = f"[{row['ci'][0]:+.2f}, {row['ci'][1]:+.2f}]"
        verdict = "PASS" if row["passes_gate"] else ("—" if row["name"].startswith("dense") else "no")
        print(
            f"{row['name']:<32} {row['ndcg_at_10']:>9.2f} {row['mrr_at_10']:>9.2f} "
            f"{row['recall_at_100']:>9.2f} {row['delta_vs_baseline_pts']:>+12.2f} {ci:>20}  {verdict}"
        )

    # The shipped ranker is a refit on everything with the hyper-parameters frozen (spec 02 §6).
    model = ltr.train(groups, rounds=args.rounds, seed=0)
    path = Path(args.save)
    model.save(path if path.is_absolute() else acis_root() / path)
    print(f"\nwritten: {out / f'{args.model}.{args.tokenizer}.json'}   ranker: {args.save}")
    _ = ladder
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
