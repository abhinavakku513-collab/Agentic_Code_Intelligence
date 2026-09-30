#!/usr/bin/env python
"""Tune the generic fusion's corpus-defined-symbol weight β on REG data, through the engine (spec 10 §5).

β scales how much a unit *defining* a rare name the query uses (`Dijkstra` -> `def dijkstra`) counts in the generic
route's fusion. It is chosen on **CosQA valid** only (ties -> the smaller β, i.e. closer to the tuned fusion), then
reported against β = 0 on data that played no part in choosing it: CosQA test, CodeSearchNet-Python test, and the
APPS dev statements the router sends generic (paired bootstrap). One `dev` ledger row per evaluated set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acis.appsdata import apps
from acis.core.paths import acis_root
from acis.eval import ladder
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import per_query, score_run
from acis.rank.compose import rank_derived_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reg_fusion as rf  # noqa: E402

GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
ALPHAS = (0.8, 0.85, 0.9, 0.95, 1.0)


def run_set(engine, base_config, data, rows, beta: float, alpha: float) -> dict[str, dict[str, float]]:
    engine.config = base_config.with_overrides(**{"rank.generic.symbol_beta": beta, "rank.generic.alpha": alpha})
    return {
        r["id"]: rank_derived_scores(
            [d for d, _ in engine._rank_one(data, r["text"], top_k=100, strict=False, route="generic")], 100
        )
        for r in rows
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)
    engine = rf.build_engine(args.config)
    base = engine.config

    sets = {}
    for task, split in (("cosqa", "valid"), ("cosqa", "test"), ("csn-python", "test")):
        corpus, splits = rf.load_task(task)
        data = engine.snapshot_data(engine.build_snapshot(corpus, source=f"reg:{task}"))
        rows = splits[split]
        sets[f"{task}/{split}"] = (data, rows, {r["id"]: {d: int(s) for d, s in r["relevant"].items()} for r in rows})
    artifact = acis_root() / "runs" / "eval" / "p0-a72f7aa27c09" / "per_query.jsonl"
    generic = [json.loads(x) for x in artifact.read_text("utf-8").splitlines() if x]
    generic_ids = [r["query_id"] for r in generic if r["route"]["route"] == "generic"]
    apps_data = engine.snapshot_data(engine.build_snapshot(apps.load_corpus(), source="serve:p0"))
    queries = apps.load_queries()
    sets["apps-dev/generic-routed"] = (
        apps_data,
        [{"id": q, "text": queries[q]} for q in generic_ids],
        dev_qrels(generic_ids),
    )

    data, rows, qrels = sets["cosqa/valid"]
    grid = {
        (a, b): score_run(qrels, run_set(engine, base, data, rows, b, a))["ndcg_at_10"] for a in ALPHAS for b in GRID
    }
    best = max(grid.values())
    # Ties -> the smaller beta, then the alpha closest to the previously tuned 0.9 (the less changed fusion).
    alpha, beta = min((k for k, v in grid.items() if v >= best - 1e-12), key=lambda k: (k[1], abs(k[0] - 0.9)))
    print("CosQA valid NDCG@10:", {f"a={a},b={b}": round(100 * v, 2) for (a, b), v in grid.items()}, flush=True)
    print(f"-> alpha {alpha}, beta {beta}", flush=True)
    old_alpha = float(base.get("rank.generic.alpha"))

    report = {
        "alpha": alpha,
        "beta": beta,
        "baseline_alpha": old_alpha,
        "grid_cosqa_valid": {f"{a},{b}": v for (a, b), v in grid.items()},
        "sets": {},
    }
    for name, (data, rows, qrels) in sets.items():
        base_run = run_set(engine, base, data, rows, 0.0, old_alpha)
        new_run = run_set(engine, base, data, rows, beta, alpha)
        a, b = per_query(qrels, new_run, "ndcg", 10), per_query(qrels, base_run, "ndcg", 10)
        boot = paired_bootstrap(a, b)
        m0, m1 = score_run(qrels, base_run), score_run(qrels, new_run)
        report["sets"][name] = {"n": len(rows), "beta0": m0, "beta": m1, "bootstrap": boot.as_dict()}
        print(
            f"{name:26s} n={len(rows):4d} NDCG@10 {100 * m0['ndcg_at_10']:.2f} -> {100 * m1['ndcg_at_10']:.2f} "
            f"Δ {boot.delta:+.2f} [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}]  MRR@10 {100 * m0['mrr_at_10']:.2f} -> "
            f"{100 * m1['mrr_at_10']:.2f}",
            flush=True,
        )
        if not args.no_ledger:
            result = ladder.LadderResult(
                rung=f"symbol-beta:{name}",
                run={},
                metrics=dict(m1),
                seconds=0.0,
                n_queries=len(rows),
                n_docs=data.size,
                notes=f"generic fusion alpha={alpha}, symbol beta={beta} (both tuned on cosqa/valid) vs alpha={old_alpha}, beta=0",
            )
            report["sets"][name]["ledger_run_id"] = ladder.record(
                result,
                kind="dev",
                extra={"alpha": alpha, "beta": beta, "bootstrap_vs_served": boot.as_dict(), "served_metrics": m0},
            )
    (acis_root() / "runs" / "reg" / "symbol_beta.json").write_text(json.dumps(report, indent=1, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
