#!/usr/bin/env python
"""Run the G-M bake-off over real encoders, one candidate at a time (docs/spec/02 §3, D4).

Measuring is hours of CPU and recording is milliseconds, so the two are separated here. Each candidate is
measured, written to `runs/bakeoff/<key>.json` the moment it finishes, and only then — optionally — appended to
the ledger. That matters for a practical reason: a gate row may only be recorded from a clean working tree
(`docs/spec/03` §6), and a multi-hour pass cannot hold development hostage to that. `--record-only` adds the
rows afterwards from a clean tree, reading the same JSON files rather than re-measuring.

Candidates are run smallest-first, so the cheap answers arrive first and a long pass on the largest model never
blocks the rest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from acis.core.paths import acis_root
from acis.embed.registry import available_cards, load_card
from acis.embed.scorecard import Scorecard
from acis.eval import ladder
from acis.eval.bakeoff import Candidate, GateRule, decide, measure_card, render_table

OUT_DIR = "runs/bakeoff"


def out_path(key: str) -> Path:
    directory = acis_root() / OUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "name": candidate.name,
        "ndcg_at_10_pts": candidate.ndcg_at_10,
        "params": candidate.params,
        "permissive": candidate.permissive,
        "pinned": candidate.pinned,
        "reference_only": candidate.reference_only,
        "n_queries": candidate.n_queries,
        "ledger_run_id": candidate.ledger_run_id,
        "metrics": dict(candidate.metrics),
        "scorecard": asdict(candidate.scorecard) if candidate.scorecard else None,
    }


def from_json(payload: dict[str, Any]) -> Candidate:
    card = payload.get("scorecard")
    return Candidate(
        key=payload["key"],
        name=payload["name"],
        ndcg_at_10=float(payload["ndcg_at_10_pts"]),
        params=payload.get("params"),
        scorecard=Scorecard(**card) if card else None,
        permissive=bool(payload.get("permissive", True)),
        pinned=bool(payload.get("pinned", True)),
        reference_only=bool(payload.get("reference_only", False)),
        n_queries=int(payload["n_queries"]),
        ledger_run_id=str(payload.get("ledger_run_id", "")),
        metrics=dict(payload.get("metrics") or {}),
    )


def load_measured(keys: list[str]) -> list[Candidate]:
    found = []
    for key in keys:
        path = out_path(key)
        if path.is_file():
            found.append(from_json(json.loads(path.read_text(encoding="utf-8"))))
    return found


def record(candidate: Candidate, *, limit: int) -> str:
    """Append this candidate's row after the fact. The decision set is what it was measured on, never inflated."""
    result = ladder.LadderResult(
        rung=f"bakeoff:{candidate.key}",
        run={},
        metrics=dict(candidate.metrics),
        seconds=float(candidate.scorecard.cold_seconds if candidate.scorecard else 0.0),
        n_queries=candidate.n_queries,
        n_docs=0,
        notes="G-M bake-off; dev split only",
    )
    kind = "gate" if candidate.n_queries >= ladder.FULL_DEV_POOL and limit == 0 else "dev"
    return ladder.record(
        result,
        kind=kind,
        extra={
            "model": candidate.key,
            "model_name": candidate.name,
            "params": candidate.params,
            "scorecard": asdict(candidate.scorecard) if candidate.scorecard else None,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G-M bake-off over real encoders")
    parser.add_argument("--models", default="", help="comma-separated card keys (default: every card, smallest first)")
    parser.add_argument("--limit", type=int, default=0, help="first N dev queries (0 = all 5,000; a gate needs all)")
    parser.add_argument("--reference", default="", help="keys measured for reference, never selected")
    parser.add_argument("--record", action="store_true", help="append ledger rows as each candidate finishes")
    parser.add_argument("--record-only", action="store_true", help="record rows for already-measured candidates")
    parser.add_argument("--force", action="store_true", help="re-measure a candidate that already has a JSON")
    args = parser.parse_args(argv)

    keys = [k.strip() for k in args.models.split(",") if k.strip()] or available_cards()
    reference = {k.strip() for k in args.reference.split(",") if k.strip()}

    if args.record_only:
        for candidate in load_measured(keys):
            if candidate.ledger_run_id:
                print(f"{candidate.key}: already recorded as {candidate.ledger_run_id}")
                continue
            run_id = record(candidate, limit=args.limit)
            payload = json.loads(out_path(candidate.key).read_text(encoding="utf-8"))
            payload["ledger_run_id"] = run_id
            out_path(candidate.key).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"{candidate.key}: recorded {run_id}")
        return 0

    for key in keys:
        path = out_path(key)
        if path.is_file() and not args.force:
            print(f"{key}: already measured ({path}); --force to re-measure", flush=True)
            continue
        started = time.perf_counter()
        print(f"{key}: measuring {load_card(key).name} …", flush=True)
        candidate = measure_card(key, limit=args.limit, reference_only=key in reference, record_row=args.record)
        path.write_text(json.dumps(to_json(candidate), indent=2, sort_keys=True), encoding="utf-8")
        card = candidate.scorecard
        print(
            f"{key}: NDCG@10 {candidate.ndcg_at_10:.2f} pt  MRR@10 {candidate.metrics.get('mrr_at_10', 0) * 100:.2f} pt"
            f"  R@100 {candidate.metrics.get('recall_at_100', 0) * 100:.2f}"
            f"  params {candidate.params}  {time.perf_counter() - started:.0f}s"
            + (f"  cold-pass proj {card.projected_cold_pass_hours} h" if card else ""),
            flush=True,
        )

    measured = load_measured(keys)
    if not measured:
        print("nothing measured", file=sys.stderr)
        return 1
    decision = decide(measured, rule=GateRule.load())
    print()
    print(render_table(measured, decision))
    print(f"\nwinner={decision.winner}  best={decision.best}  tolerance={decision.tolerance_pts} pt")
    (acis_root() / OUT_DIR / "decision.json").write_text(
        json.dumps(decision.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
