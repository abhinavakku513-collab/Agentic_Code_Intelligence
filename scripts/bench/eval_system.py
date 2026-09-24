#!/usr/bin/env python
"""Evaluate the system as configured, end to end (docs/spec/03 §5).

The gate reports measure one component against one baseline. This measures what a judge would actually run: the
configured engine, through `search_batch`, with whatever routing, channels and ranker the config names — and
reports the degradation counters alongside, so the number comes with the story of how it was produced.

It is the difference between "the ranker is worth +1.93 points" and "the system scores X", and only the second
one is a claim about the submission.
"""

from __future__ import annotations

import argparse
import json
import time

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ladder
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import K_VALUES, score_run
from acis.rank.compose import rank_derived_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="end-to-end evaluation of the configured system on the dev split")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--label", default="system")
    parser.add_argument("--record", action="store_true", help="append a ledger row (needs a clean tree)")
    parser.add_argument("--out", default="runs/system")
    args = parser.parse_args(argv)

    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))

    started = time.perf_counter()
    snapshot = engine.build_snapshot(apps.load_corpus(), source=f"system:{args.label}")
    build_seconds = time.perf_counter() - started

    ids = list(apps.dev_query_ids())
    if args.limit:
        ids = ids[: args.limit]
    queries = apps.load_queries()

    started = time.perf_counter()
    ranked = engine.search_batch(snapshot, ids, [queries[q] for q in ids], top_k=args.top_k)
    seconds = time.perf_counter() - started

    run = {qid: rank_derived_scores(hits, args.top_k) for qid, hits in ranked.items()}
    qrels = dev_qrels(ids)
    metrics = score_run(qrels, run, K_VALUES)
    counters = dict(engine.counters.snapshot())

    report = {
        "label": args.label,
        "config": args.config,
        "config_hash": config.config_hash,
        "encoder": getattr(engine.encoder, "name", "none"),
        "channel": str(config.get("run.channel", "auto")),
        "tokenizer": str(config.get("lexical.tokenizer", "stock")),
        "ranker": str(config.get("rank.model", "") or "none"),
        "route_bank": str(config.get("route.bank", "") or "none"),
        "n_queries": len(ids),
        "n_docs": snapshot.n_units,
        "snapshot_build_seconds": round(build_seconds, 2),
        "search_seconds": round(seconds, 2),
        "ms_per_query": round(seconds * 1000 / max(1, len(ids)), 2),
        "metrics": {k: round(v * 100, 4) for k, v in metrics.items() if k.endswith(("_at_10", "_at_100"))},
        "counters": counters,
    }

    out = acis_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    run_id = ""
    if args.record:
        result = ladder.LadderResult(
            rung=f"system:{args.label}",
            run=run,
            metrics=metrics,
            seconds=seconds,
            n_queries=len(ids),
            n_docs=snapshot.n_units,
            notes="end-to-end, the configured system; dev split only",
        )
        kind = "gate" if len(ids) >= ladder.FULL_DEV_POOL else "dev"
        run_id = ladder.record(result, kind=kind, extra={k: v for k, v in report.items() if k != "counters"})

    print(
        f"{args.label}: NDCG@10 {metrics['ndcg_at_10'] * 100:.2f}  MRR@10 {metrics['mrr_at_10'] * 100:.2f}  "
        f"R@100 {metrics['recall_at_100'] * 100:.2f}  ({len(ids)} queries, {report['ms_per_query']} ms/query)"
    )
    for name, value in sorted(counters.items()):
        if name.startswith("degradation."):
            print(f"  {name}: {value}")
    print(f"written: {out / f'{args.label}.json'}" + (f"   ledger={run_id}" if run_id else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
