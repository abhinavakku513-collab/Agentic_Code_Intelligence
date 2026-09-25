#!/usr/bin/env python
"""Run the G1 query-representation sweep over the real encoder, one cell at a time (docs/spec/02 §2, §5).

The same shape as `bakeoff_run.py`, for the same reasons. A cell is an hour or more of CPU, so each one is written
to `runs/g1/<route>/<cell>.json` (per-query NDCG included, which the paired bootstrap needs) the moment it finishes,
and a crash costs one cell rather than the sweep. Gate rows may only be appended from a clean tree, so
`--record-only` adds them afterwards from the same JSON; `--decide` applies the G1 rule and writes the route's
decision into `configs/gates/G1.yaml` only when every cell is recorded on all 5,000 dev queries.

The grid comes from `plan_grid`: only what the encoder's card can actually express — no task strings for a model
that takes no instruction, no lengths the encoder would truncate anyway. Cheapest cells run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.registry import load_card
from acis.eval import ladder
from acis.eval.sweep import Cell, Measurement, decide_g1, measure_cell, plan_baseline, plan_grid, render_table
from acis.eval.sweep import record_decision as write_decision

OUT_DIR = "runs/g1"
CONFIG = "configs/dev.yaml"


def out_path(route: str, cell: Cell) -> Path:
    directory = acis_root() / OUT_DIR / route
    directory.mkdir(parents=True, exist_ok=True)
    safe = cell.key.replace("/", "_").replace("-", "notask")
    return directory / f"{safe}.json"


def to_json(m: Measurement, *, route: str) -> dict[str, Any]:
    return {
        "route": route,
        "cell": {"task": m.cell.task, "view": m.cell.view, "max_tokens": m.cell.max_tokens},
        "metrics": dict(m.metrics),
        "per_query": dict(m.per_query),
        "seconds": m.seconds,
        "ledger_run_id": m.ledger_run_id,
    }


def from_json(payload: dict[str, Any]) -> Measurement:
    return Measurement(
        cell=Cell(**payload["cell"]),
        per_query={str(k): float(v) for k, v in payload["per_query"].items()},
        metrics={str(k): float(v) for k, v in payload["metrics"].items()},
        seconds=float(payload["seconds"]),
        ledger_run_id=str(payload.get("ledger_run_id", "")),
    )


def load_measured(route: str, cells: list[Cell]) -> list[Measurement]:
    return [
        from_json(json.loads(out_path(route, c).read_text(encoding="utf-8")))
        for c in cells
        if out_path(route, c).is_file()
    ]


def record(m: Measurement, *, route: str, encoder: str) -> str:
    """Append this cell's row. The decision set is what it was measured on, never inflated."""
    n = len(m.per_query)
    result = ladder.LadderResult(
        rung=f"G1:{route}:{m.cell.key}",
        run={},
        metrics=dict(m.metrics),
        seconds=m.seconds,
        n_queries=n,
        n_docs=0,
        notes="G1 query-representation sweep; dev split only",
    )
    kind = "gate" if n >= ladder.FULL_DEV_POOL else "dev"
    return ladder.record(
        result,
        kind=kind,
        extra={"gate": "G1", "route": route, "cell": m.cell.key, "model": encoder},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G1 sweep over the real encoder, one route")
    parser.add_argument("--route", default="statement_like")
    parser.add_argument("--baseline", default="T1/V0/1024", help="the incumbent cell (mapped onto the card)")
    parser.add_argument("--limit", type=int, default=0, help="first N dev queries (0 = all 5,000; a gate needs all)")
    parser.add_argument("--record-only", action="store_true", help="append ledger rows for measured cells")
    parser.add_argument("--decide", action="store_true", help="write the route's decision into configs/gates/G1.yaml")
    parser.add_argument("--force", action="store_true", help="re-measure a cell that already has a JSON")
    args = parser.parse_args(argv)

    encoder = str(load_frozen_config(CONFIG).get("model.encoder"))
    card = load_card(encoder)
    cells, skipped = plan_grid(card)
    cells.sort(key=lambda c: (c.cost_rank, c.key))
    baseline = plan_baseline(args.baseline, card)
    for value, reason in skipped.items():
        print(f"not sweeping {value}: {reason}", flush=True)

    if args.record_only:
        for m in load_measured(args.route, cells):
            path = out_path(args.route, m.cell)
            if m.ledger_run_id:
                print(f"{m.cell.key}: already recorded as {m.ledger_run_id}")
                continue
            run_id = record(m, route=args.route, encoder=encoder)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["ledger_run_id"] = run_id
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            print(f"{m.cell.key}: recorded {run_id}")
        return 0

    if not args.decide:
        ids = list(apps.dev_query_ids())
        if args.limit > 0:
            ids = ids[: args.limit]
        for cell in cells:
            path = out_path(args.route, cell)
            if path.is_file() and not args.force:
                print(f"{cell.key}: already measured ({path}); --force to re-measure", flush=True)
                continue
            started = time.perf_counter()
            print(f"{cell.key}: measuring on {len(ids)} queries …", flush=True)
            m = measure_cell(cell, route=args.route, query_ids=ids, config_path=CONFIG)
            path.write_text(json.dumps(to_json(m, route=args.route), sort_keys=True), encoding="utf-8")
            print(f"{cell.key}: NDCG@10 {m.ndcg_pts:.2f} pt  {time.perf_counter() - started:.0f}s", flush=True)

    measured = load_measured(args.route, cells)
    if len(measured) < 2 or baseline not in {m.cell.key for m in measured}:
        print(f"not enough measured to decide (have {len(measured)}, baseline {baseline})", file=sys.stderr)
        return 1
    decision = decide_g1(measured, route=args.route, baseline=baseline)
    print()
    print(render_table(measured, decision))
    print(f"\nwinner={decision.winner}  adopted={decision.adopted}  n={decision.n_queries}")
    if args.decide:
        unrecorded = [m.cell.key for m in measured if not m.ledger_run_id]
        if unrecorded or decision.n_queries < ladder.FULL_DEV_POOL:
            print(f"refusing to decide: unrecorded={unrecorded} n={decision.n_queries}", file=sys.stderr)
            return 1
        print(f"wrote {write_decision(decision)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
