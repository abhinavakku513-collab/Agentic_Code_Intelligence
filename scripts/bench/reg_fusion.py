#!/usr/bin/env python
"""Tune and measure the generic route's weighted fusion on REG data (docs/spec/02 §6b, spec 10 §5).

The generic route serves every query that is not an APPS-style problem statement, which is what a person typing
into the page usually writes. Its ranking is `α · minmax(cos) + (1 − α) · bm25 / max bm25` over the candidate
pool (`acis.rank.fusion`). This script chooses α and says what it is worth, on human-written queries:

1. **collect** — build a snapshot of each REG corpus with the shipping configuration and record, per query, the
   engine's own candidate pool (`AcisEngine.candidate_pool`, the code serving runs) and its route decision.
   Embedding the corpus is the only expensive step; vectors land in the content-addressed cache.
2. **tune** — α over a fixed grid on **CosQA valid** only; ties go to the larger α (closer to the frozen dense
   order: the cheaper, less assuming choice).
3. **report** — the chosen α against the dense order on **CosQA test** and **CodeSearchNet-Python**, which played
   no part in choosing it: paired bootstrap, gate rule Δ ≥ +0.5 pt with CI lower bound > 0. RRF is reported as a
   reference only. One `dev` ledger row per (set, system).
4. **verify** — CosQA test once more end to end through `engine.search_batch` with α in the config, with routing,
   and the scores must equal step 3's: the number reported is the number the engine produces.

Never touches APPS TEST labels; REG labels live under `reg_home()`, outside `ACIS_HOME`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

from acis.core.config import load_frozen_config
from acis.core.paths import acis_root, reg_home
from acis.core.types import Snippet
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ladder
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.metrics import K_VALUES, per_query, score_run
from acis.rank.candidates import Candidate, reciprocal_rank
from acis.rank.compose import rank_derived_scores
from acis.rank.fusion import weighted_fusion

OUT = "runs/reg"
GRID = tuple(round(0.05 * i, 2) for i in range(21))
TOP_K = 100


def load_task(name: str) -> tuple[list[Snippet], dict[str, list[dict[str, Any]]]]:
    root = reg_home() / "export" / name
    corpus = [json.loads(line) for line in (root / "corpus.jsonl").read_text("utf-8").splitlines()]
    splits = {}
    for path in sorted(root.glob("queries.*.jsonl")):
        splits[path.name.split(".")[1]] = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    return [Snippet(handle=row["id"], text=row["text"]) for row in corpus], splits


def build_engine(config_path: str, alpha: float | None = None) -> AcisEngine:
    overrides: dict[str, Any] = {} if alpha is None else {"rank.generic.alpha": alpha, "run.channel": "hybrid"}
    config = load_frozen_config(config_path).with_overrides(**overrides)
    return AcisEngine.from_config(config, encoder=build_encoder(config))


def collect(engine: AcisEngine, name: str) -> dict[str, Any]:
    """Pools and route decisions for every query of every split, through the engine's own code."""
    corpus, splits = load_task(name)
    started = time.perf_counter()
    snapshot = engine.build_snapshot(corpus, source=f"reg:{name}")
    data = engine.snapshot_data(snapshot)
    print(f"{name}: {snapshot.n_units} units in {time.perf_counter() - started:.0f}s", flush=True)
    out: dict[str, Any] = {"task": name, "n_docs": snapshot.n_units, "config_hash": engine.config_hash, "splits": {}}
    for split, rows in splits.items():
        records = []
        for row in rows:
            query, _ = engine.normalise_query(row["text"])
            decision = engine.route_decision(query)
            dense, pool = engine.candidate_pool(data, query, route="generic", want=TOP_K)
            # The fusion needs a cosine for every candidate; the engine fills lexical-only ones the same way.
            vector = engine._query_vector(snapshot.snapshot_id, query, route="generic")
            index = {d: i for i, d in enumerate(data.doc_ids)}
            pool = [
                c
                if c.dense_score == c.dense_score
                else Candidate(**{**asdict(c), "dense_score": float(data.vectors[index[c.doc_id]] @ vector)})
                for c in pool
            ]
            records.append(
                {
                    "id": row["id"],
                    "relevant": row["relevant"],
                    "route": decision.as_dict(),
                    "dense": [d for d, _ in dense[:TOP_K]],
                    "pool": [asdict(c) for c in pool],
                    "hash": {c.doc_id: data.hash_of[c.doc_id] for c in pool},
                    # Exact duplicates are ordered by corpus ordinal (D9); CosQA has 20,604 functions but only
                    # 6,267 distinct bodies, and its labels name one particular copy.
                    "ordinal": {c.doc_id: data.ordinal[c.doc_id] for c in pool},
                }
            )
        out["splits"][split] = records
        print(f"{name}/{split}: {len(records)} queries", flush=True)
    return out


def fused_run(record: dict[str, Any], alpha: float) -> list[str]:
    pool = [Candidate(**c) for c in record["pool"]]
    fused = weighted_fusion(pool, alpha=alpha)
    hashes = record["hash"]
    ordinal = record["ordinal"]
    # `AcisEngine._stable_order`: score, then content hash, then corpus ordinal.
    head = [d for d, _ in sorted(fused.items(), key=lambda item: (-item[1], hashes[item[0]], ordinal[item[0]]))]
    seen = set(head)
    return head + [d for d in record["dense"] if d not in seen]


def rrf_run(record: dict[str, Any]) -> list[str]:
    pool = [Candidate(**c) for c in record["pool"]]
    head = [
        c.doc_id
        for c in sorted(
            pool,
            key=lambda c: (-(reciprocal_rank(c.dense_rank) + reciprocal_rank(c.lexical_rank)), c.dense_rank or 1 << 30),
        )
    ]
    seen = set(head)
    return head + [d for d in record["dense"] if d not in seen]


def as_run(orders: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return {qid: rank_derived_scores(order[:TOP_K], TOP_K) for qid, order in orders.items()}


def evaluate(records: list[dict[str, Any]], system: str, alpha: float | None = None) -> tuple[dict, dict]:
    qrels = {r["id"]: {d: int(s) for d, s in r["relevant"].items()} for r in records}
    if system == "dense":
        orders = {r["id"]: r["dense"] for r in records}
    elif system == "rrf":
        orders = {r["id"]: rrf_run(r) for r in records}
    else:
        assert alpha is not None
        orders = {r["id"]: fused_run(r, alpha) for r in records}
    run = as_run(orders)
    return score_run(qrels, run, K_VALUES), per_query(qrels, run, "ndcg", 10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--tasks", default="cosqa,csn-python")
    parser.add_argument("--stage", default="all", choices=["collect", "tune", "all"])
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)
    out_dir = acis_root() / OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks.split(",")
    if args.stage in ("collect", "all"):
        engine = build_engine(args.config)
        for name in tasks:
            (out_dir / f"{name}.pools.json").write_text(json.dumps(collect(engine, name)), "utf-8")
    if args.stage == "collect":
        return 0

    pools = {name: json.loads((out_dir / f"{name}.pools.json").read_text("utf-8")) for name in tasks}
    tuning = pools["cosqa"]["splits"]["valid"]
    grid = {alpha: evaluate(tuning, "fused", alpha)[0]["ndcg_at_10"] for alpha in GRID}
    best = max(grid.values())
    alpha = max(a for a, v in grid.items() if v == best)  # ties -> closer to the dense order
    print("CosQA valid, NDCG@10 by alpha:", {a: round(100 * v, 2) for a, v in grid.items()})
    print(f"chosen alpha = {alpha}")

    report: dict[str, Any] = {"alpha": alpha, "grid_cosqa_valid": grid, "config": args.config, "sets": {}}
    sets = [("cosqa", "valid"), ("cosqa", "test")] + [(t, "test") for t in tasks if t != "cosqa"]
    for name, split in sets:
        records = pools[name]["splits"][split]
        dense_m, dense_pq = evaluate(records, "dense")
        rows = {"dense": dense_m}
        for system in ("fused", "rrf"):
            m, pq = evaluate(records, system, alpha if system == "fused" else None)
            boot = paired_bootstrap(pq, dense_pq)
            rows[system] = {**m, "bootstrap_vs_dense": boot.as_dict(), "passes_gate": boot.passes()}
        routes: dict[str, int] = {}
        for r in records:
            routes[r["route"]["route"]] = routes.get(r["route"]["route"], 0) + 1
        report["sets"][f"{name}/{split}"] = {"n_queries": len(records), "routes": routes, **rows}
        f = rows["fused"]
        print(
            f"{name}/{split:6s} n={len(records):4d} dense {100 * dense_m['ndcg_at_10']:6.2f}  "
            f"fused(α={alpha}) {100 * f['ndcg_at_10']:6.2f}  Δ {f['bootstrap_vs_dense']['delta']:+.2f} "
            f"CI [{f['bootstrap_vs_dense']['ci_low']:+.2f}, {f['bootstrap_vs_dense']['ci_high']:+.2f}] "
            f"{'PASS' if f['passes_gate'] else 'no'}  | rrf {100 * rows['rrf']['ndcg_at_10']:6.2f}  routes {routes}",
            flush=True,
        )
        if not args.no_ledger:
            for system in ("dense", "fused", "rrf"):
                metrics = (
                    rows[system]
                    if system == "dense"
                    else {k: v for k, v in rows[system].items() if isinstance(v, float)}
                )
                result = ladder.LadderResult(
                    rung=f"reg-fusion:{name}/{split}:{system}",
                    run={},
                    metrics=metrics,
                    seconds=0.0,
                    n_queries=len(records),
                    n_docs=int(pools[name]["n_docs"]),
                    notes=f"generic-route fusion on REG data; alpha tuned on cosqa/valid only; {system}",
                )
                extra = {"alpha": alpha if system == "fused" else None, "reg_task": name, "reg_split": split}
                if system != "dense":
                    extra["bootstrap_vs_dense"] = rows[system]["bootstrap_vs_dense"]
                rows[system]["ledger_run_id"] = ladder.record(result, kind="dev", extra=extra)

    # 4. End to end: the engine with alpha in its config, routing on, must reproduce the offline fused numbers for
    # every query it routes generic.
    engine = build_engine(args.config, alpha=alpha)
    corpus, splits = load_task("cosqa")
    snapshot = engine.build_snapshot(corpus, source="reg:cosqa")
    records = pools["cosqa"]["splits"]["test"]
    texts = {r["id"]: t["text"] for r in records for t in splits["test"] if t["id"] == r["id"]}
    ranked = engine.search_batch(snapshot, list(texts), list(texts.values()), top_k=TOP_K)
    generic = [r for r in records if r["route"]["route"] == "generic"]
    mismatched = [r["id"] for r in generic if [d for d, _ in ranked[r["id"]]][:10] != fused_run(r, alpha)[:10]]
    qrels = {r["id"]: {d: int(s) for d, s in r["relevant"].items()} for r in records}
    engine_metrics = score_run(qrels, {q: rank_derived_scores([d for d, _ in h], TOP_K) for q, h in ranked.items()})
    report["end_to_end_cosqa_test"] = {
        "engine_ndcg_at_10": engine_metrics["ndcg_at_10"],
        "engine_mrr_at_10": engine_metrics["mrr_at_10"],
        "generic_queries": len(generic),
        "generic_top10_mismatches": mismatched,
        "counters": engine.counters.snapshot(),
    }
    print(
        f"end to end (engine.search_batch, routing on): NDCG@10 {100 * engine_metrics['ndcg_at_10']:.2f}; "
        f"{len(generic)} generic queries, top-10 mismatches vs offline: {len(mismatched)}",
        flush=True,
    )
    (out_dir / "fusion_report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), "utf-8")
    return 0 if not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
