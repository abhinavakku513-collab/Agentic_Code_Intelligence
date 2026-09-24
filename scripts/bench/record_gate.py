#!/usr/bin/env python
"""Append a gate row for a measured `ltr_build` report (docs/spec/03 §6, INV-14).

Measuring and recording are separate for the same reason they are in the bake-off: the measurement is half an
hour of CPU and the row must be written from a clean tree, so that its `git_sha` rebuilds the number it quotes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acis.eval import ladder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="record a gate row from an ltr_build report")
    parser.add_argument("report")
    parser.add_argument("--system", default="LTR", help="which row of the report to record")
    parser.add_argument("--gate", required=True, help="the gate this row is evidence for, e.g. G5")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    row = next(r for r in report["rows"] if r["name"].startswith(args.system))
    result = ladder.LadderResult(
        rung=f"{args.gate}:{report['model']}:{report['tokenizer']}:{args.system.lower()}",
        run={},
        metrics={
            "ndcg_at_10": row["ndcg_at_10"] / 100.0,
            "mrr_at_10": row["mrr_at_10"] / 100.0,
            "recall_at_100": row["recall_at_100"] / 100.0,
        },
        seconds=0.0,
        n_queries=int(report["n_queries"]),
        n_docs=8765,
        notes=f"{args.gate}; dev split only; out-of-fold where the system is trained",
    )
    run_id = ladder.record(
        result,
        kind="gate",
        extra={
            "gate": args.gate,
            "model": report["model"],
            "tokenizer": report["tokenizer"],
            "system": row["name"],
            "delta_vs_dense_pts": row["delta_vs_baseline_pts"],
            "ci": row["ci"],
            "passes_gate": row["passes_gate"],
            "config_hash": report.get("config_hash", ""),
        },
    )
    print(f"{args.gate} {report['model']} {report['tokenizer']} {args.system}: recorded {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
