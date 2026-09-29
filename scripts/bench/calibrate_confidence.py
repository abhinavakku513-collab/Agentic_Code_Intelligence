#!/usr/bin/env python
"""Fit the confidence calibration: z of the served #1 → P(served #1 relevant), per route (`acis.rank.confidence`).

* `statement_like` — the latest P0 pipeline artifact (APPS dev, out of fold): each fold's map is fitted on the
  other four and scored on it (held-out reliability), then the shipped map is fitted on all of them.
* `generic` — CosQA `valid` (REG, human-written queries) under the served fusion weight; held-out reliability on
  CosQA `test`, and on the APPS dev statements the router sends generic.

The bands are fixed (`high` ≥ 0.6, `medium` ≥ 0.3), so what this records is how often each band's top result was
actually relevant on data the map never saw — the number that says whether a "high" deserves the word.
Writes `artifacts/confidence/calibration.json` and one `dev` ledger row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from acis.core.hashing import hash_obj
from acis.core.paths import acis_root
from acis.eval import ledger
from acis.rank.confidence import band, isotonic, lookup, z_top1

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_fusion import fused_run  # noqa: E402 — the same served order the fusion report measured

OUT = "artifacts/confidence/calibration.json"


def reliability(zs: list[float], hits: list[int], knots: list[float], values: list[float]) -> dict[str, Any]:
    probs = [lookup(knots, values, z) for z in zs]
    ok = [(p, h) for p, h in zip(probs, hits, strict=True) if p == p]
    out: dict[str, Any] = {
        "n": len(ok),
        "brier": float(np.mean([(p - h) ** 2 for p, h in ok])) if ok else None,
        "base_rate": float(np.mean([h for _, h in ok])) if ok else None,
    }
    for level in ("high", "medium", "low"):
        members = [h for p, h in ok if band(p) == level]
        out[level] = {"n": len(members), "top1_relevant_rate": float(np.mean(members)) if members else None}
    return out


def generic_points(records: list[dict[str, Any]], alpha: float) -> tuple[list[float], list[int]]:
    zs, hits = [], []
    for r in records:
        order = fused_run(r, alpha)
        pool = {c["doc_id"]: c for c in r["pool"]}
        crowd = sorted((c["dense_score"] for c in r["pool"] if c["dense_rank"]), reverse=True)[:100]
        zs.append(z_top1(crowd, float(pool[order[0]]["dense_score"])))
        hits.append(int(order[0] in r["relevant"]))
    return zs, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default="", help="per_query.jsonl of a P0 pipeline run (default: latest)")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)
    root = acis_root()

    artifact = (
        (root / args.artifact).resolve()
        if args.artifact
        else max((root / "runs" / "eval").glob("p0-*/per_query.jsonl"), key=lambda p: p.stat().st_mtime)
    )
    apps = [json.loads(line) for line in artifact.read_text("utf-8").splitlines() if line]
    missing = [r for r in apps if r.get("confidence_z") is None]
    if len(missing) == len(apps):
        raise SystemExit(f"{artifact} has no confidence signal; re-run scripts/bench/eval_pipeline.py")

    report: dict[str, Any] = {"artifact": str(artifact.relative_to(root)), "routes": {}}
    # statement_like: out of fold on APPS dev.
    st = [r for r in apps if r["route"]["route"] == "statement_like" and r.get("confidence_z") is not None]
    held_z, held_h, held_p = [], [], []
    for fold in sorted({r["fold"] for r in st}):
        train = [r for r in st if r["fold"] != fold]
        test = [r for r in st if r["fold"] == fold]
        knots, values = isotonic([r["confidence_z"] for r in train], [int(r["rank_full"] == 1) for r in train])
        for r in test:
            held_z.append(r["confidence_z"])
            held_h.append(int(r["rank_full"] == 1))
            held_p.append(lookup(knots, values, r["confidence_z"]))
    knots, values = isotonic([r["confidence_z"] for r in st], [int(r["rank_full"] == 1) for r in st])
    report["routes"]["statement_like"] = {
        "knots": knots,
        "values": values,
        "n": len(st),
        "fitted_on": f"APPS dev statements routed statement_like, out of fold ({len(st)} queries)",
        "held_out": _oof(held_p, held_h),
    }

    # generic: CosQA valid; held out on CosQA test and on APPS dev statements routed generic.
    fusion = json.loads((root / "runs" / "reg" / "fusion_report.json").read_text("utf-8"))
    alpha = float(fusion["alpha"])
    pools = json.loads((root / "runs" / "reg" / "cosqa.pools.json").read_text("utf-8"))
    vz, vh = generic_points(pools["splits"]["valid"], alpha)
    gk, gv = isotonic(vz, vh)
    tz, th = generic_points(pools["splits"]["test"], alpha)
    ag = [r for r in apps if r["route"]["route"] == "generic" and r.get("confidence_z") is not None]
    report["routes"]["generic"] = {
        "knots": gk,
        "values": gv,
        "n": len(vz),
        "alpha": alpha,
        "fitted_on": f"CosQA valid under the served fusion weight α={alpha} ({len(vz)} queries)",
        "held_out_cosqa_test": reliability(tz, th, gk, gv),
        "held_out_apps_generic": reliability(
            [r["confidence_z"] for r in ag], [int(r["rank_full"] == 1) for r in ag], gk, gv
        ),
    }
    content = hash_obj({k: {"knots": v["knots"], "values": v["values"]} for k, v in report["routes"].items()})
    run_id = ""
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="dev")
            .with_metrics(
                {
                    "statement_like_heldout_brier": report["routes"]["statement_like"]["held_out"]["brier"],
                    "generic_heldout_brier_cosqa_test": report["routes"]["generic"]["held_out_cosqa_test"]["brier"],
                }
            )
            .with_fields(
                rung="confidence-calibration",
                calibration_sha=content,
                source_artifact=report["artifact"],
                reliability={
                    k: {kk: vv for kk, vv in v.items() if kk not in ("knots", "values")}
                    for k, v in report["routes"].items()
                },
            )
            .build()
        )
        run_id = ledger.append(row).run_id
    out = root / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ledger_run_id": run_id, "calibration_sha": content, **report}, indent=1), "utf-8")

    for route, fit in report["routes"].items():
        for name, rel in fit.items():
            if name.startswith("held_out"):
                bands = "  ".join(
                    f"{lvl}: n={rel[lvl]['n']} top1-relevant {rel[lvl]['top1_relevant_rate']:.2f}"
                    if rel[lvl]["top1_relevant_rate"] is not None
                    else f"{lvl}: n=0"
                    for lvl in ("high", "medium", "low")
                )
                print(
                    f"{route:<15} {name:<24} n={rel['n']:<5} brier {rel['brier']:.3f}  "
                    f"base {rel['base_rate']:.2f}  {bands}"
                )
    print(f"written {out.relative_to(root)}  ledger={run_id or '(not recorded)'}")
    return 0


def _oof(probs: list[float], hits: list[int]) -> dict[str, Any]:
    ok = [(p, h) for p, h in zip(probs, hits, strict=True) if p == p]
    out: dict[str, Any] = {
        "n": len(ok),
        "brier": float(np.mean([(p - h) ** 2 for p, h in ok])),
        "base_rate": float(np.mean([h for _, h in ok])),
    }
    for level in ("high", "medium", "low"):
        members = [h for p, h in ok if band(p) == level]
        out[level] = {"n": len(members), "top1_relevant_rate": float(np.mean(members)) if members else None}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
