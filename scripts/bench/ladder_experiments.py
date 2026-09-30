#!/usr/bin/env python
"""Accuracy ladder experiments on the full dev set, out of fold, through the engine's own stages (spec 08 §3).

Each configuration is a set of config overrides (candidate pool sizes, BM25 parameters, PRF, ranker rounds). For
each, and for all 5,000 dev queries:

* the candidate pool comes from `AcisEngine.candidate_pool`, PRF from `_with_prf`, the ranker's features from
  `pool_features`, the generic route's order from `fused_order` — the functions serving calls;
* rankers are cross-fitted (each fold scored by a model trained on the other four) and applied with the
  ranker's own `rerank`, abstention included;
* each query keeps the route it was given in the recorded pipeline evaluation (per-fold τ, routing depends on the
  query only), so only the lever under test changes.

Reported per configuration: NDCG@10, MRR@10, Recall@100 of the final list, and **pool recall** (is the relevant
document among the candidates the ranker sees at all), each paired-bootstrapped against the first configuration.
One `dev` ledger row per configuration. Winners are then confirmed end to end by `eval_pipeline.py`.
"""

from __future__ import annotations

import argparse
import json
import time
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
from acis.rank import ltr
from acis.rank.compose import rank_derived_scores

TOP_K = 100
CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "pool_lex100_cap130": {"retrieve.lexical_k": 100, "retrieve.union_cap": 130},
    "pool_dense200_lex50_cap200": {"retrieve.dense_k": 200, "retrieve.lexical_k": 50, "retrieve.union_cap": 200},
    "pool_dense200_lex100_cap250": {"retrieve.dense_k": 200, "retrieve.lexical_k": 100, "retrieve.union_cap": 250},
    "prf": {"retrieve.prf.enabled": True},
    "bm25_k1_1.2_b_0.75": {"lexical.k1": 1.2},
    "bm25_k1_0.9_b_0.4": {"lexical.k1": 0.9, "lexical.b": 0.4},
    "bm25_k1_1.5_b_0.9": {"lexical.b": 0.9},
}


def run_config(name: str, overrides: dict[str, Any], *, base: str, routes: dict[str, str], rounds: int) -> dict:
    overrides = dict(overrides)
    rounds = int(overrides.pop("_rounds", rounds))  # ranker boosting rounds (spec 02 §4 allows 300-600)
    config = load_frozen_config(base).with_overrides(**overrides)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    snapshot = engine.build_snapshot(apps.load_corpus(), source=f"ladder:{name}")
    data = engine.snapshot_data(snapshot)
    data.warm_features()
    ids = list(apps.dev_query_ids())
    qrels = dev_qrels(ids)
    queries = apps.load_queries()
    fold_of = {q: f for f, members in fold_members().items() for q in members}

    started = time.perf_counter()
    items: dict[str, dict[str, Any]] = {}
    groups: list[ltr.TrainingGroup] = []
    for qid in ids:
        query, _ = engine.normalise_query(queries[qid])
        route = routes[qid]
        dense, pool = engine.candidate_pool(data, query, route=route, want=TOP_K)
        gold = {d for d, rel in qrels[qid].items() if rel > 0}
        entry: dict[str, Any] = {"dense": [d for d, _ in dense], "pool_hit": any(c.doc_id in gold for c in pool)}
        if route == "statement_like":
            pool = engine._with_prf(data, query, route=route, pool=pool)
            matrix = engine.pool_features(data, query, pool)
            labels = np.array([int(c.doc_id in gold) for c in pool], dtype=np.int32)
            groups.append(ltr.TrainingGroup(qid, matrix, labels, tuple(c.doc_id for c in pool)))
            entry.update(doc_ids=[c.doc_id for c in pool], matrix=matrix)
        else:
            entry["head"] = engine.fused_order(data, query, pool, route=route)
        items[qid] = entry
    build_seconds = time.perf_counter() - started

    # The ranker learns from statement-like queries' pools, as the shipped one does from all dev pools.
    by_fold: dict[int, list[ltr.TrainingGroup]] = {}
    for g in groups:
        by_fold.setdefault(fold_of[g.query_id], []).append(g)
    orders: dict[str, list[str]] = {}
    for fold in sorted(by_fold):
        model = ltr.train([g for f, gs in by_fold.items() if f != fold for g in gs], rounds=rounds, seed=0)
        for g in by_fold[fold]:
            head, abstained = model.rerank(list(g.doc_ids), g.features)
            items[g.query_id]["head"] = [] if abstained else head
    for qid, entry in items.items():
        seen = set(entry["head"])
        orders[qid] = entry["head"] + [d for d in entry["dense"] if d not in seen]
    run = {q: rank_derived_scores(o[:TOP_K], TOP_K) for q, o in orders.items()}
    metrics = score_run(qrels, run, K_VALUES)
    return {
        "name": name,
        "overrides": {**overrides, "_rounds": rounds},
        "config_hash": config.config_hash,
        "metrics": metrics,
        "pool_recall": float(np.mean([e["pool_hit"] for e in items.values()])),
        "per_query_ndcg": per_query(qrels, run, "ndcg", 10),
        "per_query_mrr": per_query(qrels, run, "mrr", 10),
        "per_query_pool": {q: float(e["pool_hit"]) for q, e in items.items()},
        "seconds_build": round(build_seconds, 1),
        "n_queries": len(ids),
        "n_docs": snapshot.n_units,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--extra", default="", help='JSON {"name": {overrides}} added to the grid')
    parser.add_argument("--artifact", default="runs/eval/p0-7a62372fe351/per_query.jsonl")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)

    grid = dict(CONFIGS)
    if args.extra:
        grid.update(json.loads(args.extra))
    names = [n for n in args.configs.split(",") if n] + [
        n for n in json.loads(args.extra or "{}") if n not in args.configs
    ]
    records = [json.loads(line) for line in (acis_root() / args.artifact).read_text("utf-8").splitlines() if line]
    routes = {r["query_id"]: r["route"]["route"] for r in records}

    out_dir = acis_root() / "runs" / "ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    baseline = None
    print(f"{'config':<30} {'NDCG@10':>8} {'MRR@10':>8} {'R@100':>7} {'pool R':>7}  Δ NDCG vs first [CI]", flush=True)
    for name in names:
        r = run_config(name, grid[name], base=args.config, routes=routes, rounds=args.rounds)
        if baseline is None:
            baseline = r
        boot = paired_bootstrap(r["per_query_ndcg"], baseline["per_query_ndcg"])
        boot_mrr = paired_bootstrap(r["per_query_mrr"], baseline["per_query_mrr"])
        boot_pool = paired_bootstrap(r["per_query_pool"], baseline["per_query_pool"])
        r["vs_first"] = {
            "ndcg_at_10": boot.as_dict(),
            "mrr_at_10": boot_mrr.as_dict(),
            "pool_recall": boot_pool.as_dict(),
        }
        m = r["metrics"]
        print(
            f"{name:<30} {100 * m['ndcg_at_10']:8.2f} {100 * m['mrr_at_10']:8.2f} {100 * m['recall_at_100']:7.2f} "
            f"{100 * r['pool_recall']:7.2f}  {boot.delta:+.2f} [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}]"
            f"{'  PASS' if boot.passes() else ''}   ({r['seconds_build']}s)",
            flush=True,
        )
        if not args.no_ledger:
            result = ladder.LadderResult(
                rung=f"ladder:{name}",
                run={},
                metrics={**m, "pool_recall": r["pool_recall"]},
                seconds=r["seconds_build"],
                n_queries=r["n_queries"],
                n_docs=r["n_docs"],
                notes="spec 08 §3 ladder experiment; engine stages, OOF rankers, recorded routes; dev split",
            )
            r["ledger_run_id"] = ladder.record(
                result,
                kind="dev",
                extra={
                    "overrides": r["overrides"],
                    "config_hash": r["config_hash"],
                    "vs_first": r["vs_first"],
                    "compared_to": names[0],
                },
            )
        results.append({k: v for k, v in r.items() if not k.startswith("per_query")})
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
