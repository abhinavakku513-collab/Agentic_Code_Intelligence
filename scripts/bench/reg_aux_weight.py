#!/usr/bin/env python
"""Generic-route fusion with a second dense encoder on human-written queries (REG; never APPS TEST).

For each REG set: one snapshot with the configured second encoder, then for every query the engine's own
`candidate_pool` and `fused_order` under `rank.generic.aux_weight` ∈ grid — the exact code serving runs — scored
with our metric code and paired-bootstrapped against w = 0 (the previous generic route). Used to check that a
weight chosen on APPS statements does not hurt the queries the generic route exists for.
"""

from __future__ import annotations

import argparse
import json
import time

from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.metrics import K_VALUES, per_query, score_run
from acis.rank.compose import rank_derived_scores

TOP_K = 100


def main(argv: list[str] | None = None) -> int:
    import sys

    sys.path.insert(0, str(acis_root() / "scripts" / "bench"))
    from reg_fusion import load_task  # noqa: PLC0415 — the same REG loader the alpha tuning used

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--task", default="csn-python")
    parser.add_argument("--split", default="test")
    parser.add_argument("--aux", default="qwen3-embedding-0.6b")
    parser.add_argument("--grid", default="0,0.5,0.75,1.0")
    args = parser.parse_args(argv)
    grid = [float(w) for w in args.grid.split(",")]
    config = load_frozen_config(args.config).with_overrides(
        **{"model.aux_encoder": args.aux, "run.channel": "hybrid"}
    )
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    corpus, splits = load_task(args.task)
    started = time.perf_counter()
    snapshot = engine.build_snapshot(corpus, source=f"reg:{args.task}")
    data = engine.snapshot_data(snapshot)
    print(f"{args.task}: {snapshot.n_units} units embedded in {time.perf_counter() - started:.0f}s", flush=True)
    rows = splits[args.split]
    qrels = {r["id"]: {d: int(s) for d, s in r["relevant"].items()} for r in rows}
    orders: dict[float, dict[str, list[str]]] = {w: {} for w in grid}
    for n, row in enumerate(rows, 1):
        query, _ = engine.normalise_query(row["text"])
        dense, pool = engine.candidate_pool(data, query, route="generic", want=TOP_K)
        for w in grid:
            engine.config = config.with_overrides(**{"rank.generic.aux_weight": w})
            head = engine.fused_order(data, query, pool, route="generic")
            seen = set(head)
            orders[w][row["id"]] = head + [d for d, _ in dense if d not in seen]
        engine.config = config
        if n % 200 == 0:
            print(f"  {n}/{len(rows)} queries ({time.perf_counter() - started:.0f}s)", flush=True)
    results = {}
    base = None
    for w in grid:
        run = {q: rank_derived_scores(o[:TOP_K], TOP_K) for q, o in orders[w].items()}
        m = score_run(qrels, run, K_VALUES)
        pq = per_query(qrels, run, "ndcg", 10)
        base = base or pq
        boot = paired_bootstrap(pq, base)
        results[str(w)] = {"ndcg_at_10": m["ndcg_at_10"], "mrr_at_10": m["mrr_at_10"], "vs_w0": boot.as_dict()}
        print(
            f"w={w:<5} NDCG@10 {100 * m['ndcg_at_10']:.2f} MRR@10 {100 * m['mrr_at_10']:.2f} "
            f"Δ {boot.delta:+.2f} [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}]",
            flush=True,
        )
    out = acis_root() / "runs" / "reg" / f"aux_weight.{args.task}.{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": args.task, "split": args.split, "aux": args.aux, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
