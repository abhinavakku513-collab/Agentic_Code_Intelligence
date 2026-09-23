#!/usr/bin/env python
"""P1 update times: how long until a change is searchable (docs/spec/04 §5, D11).

The official requirement talks about commits arriving "every minute", and the spec turns that into a target: **≤
10 changed units searchable within ≤ 30 s p95**. This measures it the only way that means anything — end to end,
from "here is a new version" to "a query against it returns the new content" — because a build that finishes
quickly and then takes twenty seconds to open has not made anything searchable.

Three change sizes (1, 10, 100 units) against one base corpus, plus a full rebuild for comparison, repeated for a
p95. The comparison is the point: if a 1-unit change costs what a full rebuild costs, the incremental story is
decoration.

Each run is written to `runs/p1_update.json` and recorded in the ledger, so the numbers can be quoted with a run
id (INV-14). The encoder in use is reported: with the stand-in this measures the *store*, and the embedding cost
that dominates a real run is absent — which the report says in as many words rather than leaving to the reader.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from acis.core.config import load_frozen_config
from acis.core.paths import acis_home, acis_root
from acis.core.types import SearchRequest, SourceSpec
from acis.embed.factory import build_encoder
from acis.embed.scorecard import peak_rss_mb, percentiles
from acis.engine import AcisEngine
from acis.eval import ledger
from acis.eval.appsevolve import constant, rename

TARGET_S = 30.0
CHANGE_SIZES = (1, 10, 100)
REPO = "p1-bench"


def base_corpus(n_units: int) -> dict[str, str]:
    """Real Python where we have it, generated bodies otherwise — the shape matters more than the source."""
    try:
        from acis.appsdata import apps

        if apps.is_available():
            docs = apps.load_corpus()[:n_units]
            return {f"unit{i:05d}.py": d.text for i, d in enumerate(docs)}
    except Exception:  # noqa: BLE001 — the benchmark must run without dataset assets
        pass
    body = "def solve_{i}(xs):\n    total = {i}\n    for x in xs:\n        total += x\n    return total\n"
    return {f"unit{i:05d}.py": body.format(i=i) for i in range(n_units)}


def change(units: dict[str, str], k: int, rng: random.Random) -> dict[str, str]:
    """Edit `k` units the way a commit would: rename an identifier, or change a constant."""
    out = dict(units)
    for key in rng.sample(sorted(out), min(k, len(out))):
        out[key] = rename(out[key], rng) if rng.random() < 0.5 else constant(out[key], rng)
    return out


def searchable_seconds(engine: AcisEngine, label: str, units: dict[str, str]) -> float:
    """Build, activate, open and query. The clock stops when a search against the new version answers."""
    spec = SourceSpec(kind="memory", location=REPO, options={"versions": {label: units}})
    started = time.perf_counter()
    engine.update_version(REPO, spec)
    engine.search(SearchRequest(query="total for loop return", repo_id=REPO, version=label, top_k=10))
    return time.perf_counter() - started


def measure(config_path: str, *, n_units: int, repeats: int, seed: int) -> dict[str, Any]:
    config = load_frozen_config(config_path)
    encoder = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=encoder)
    rng = random.Random(seed)

    units = base_corpus(n_units)
    started = time.perf_counter()
    engine.ingest(SourceSpec(kind="memory", location=REPO, options={"versions": {"v1": units}}), repo_id=REPO)
    first_build = time.perf_counter() - started

    samples: dict[str, list[float]] = {f"change_{k}": [] for k in CHANGE_SIZES}
    samples["full_rebuild"] = []
    version = 1
    current = units
    for repeat in range(repeats):
        for k in CHANGE_SIZES:
            version += 1
            current = change(current, k, rng)
            samples[f"change_{k}"].append(searchable_seconds(engine, f"v{version}", current))
        version += 1
        # A "full rebuild" is every unit rewritten: the honest worst case, and the number the incremental sizes
        # are compared against. It is slower, and reporting it is the point (spec 04 §3).
        current = change(current, len(current), rng)
        samples["full_rebuild"].append(searchable_seconds(engine, f"v{version}", current))
        _ = repeat

    return {
        "config": config_path,
        "config_hash": config.config_hash,
        "encoder": encoder.name,
        "submission_capable": encoder.submission_capable,
        "numeric_profile": config.numeric_profile,
        "n_units": len(units),
        "repeats": repeats,
        "seed": seed,
        "first_build_seconds": round(first_build, 3),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "target_seconds": TARGET_S,
        "seconds": {name: percentiles(values) for name, values in samples.items()},
        "raw_seconds": {name: [round(v, 4) for v in values] for name, values in samples.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 update-time benchmark (docs/spec/04 §5)")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--units", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/p1_update.json")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--keep", action="store_true", help="leave the benchmark repository in the store")
    args = parser.parse_args(argv)

    report = measure(args.config, n_units=args.units, repeats=args.repeats, seed=args.seed)

    run_id = ""
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="bench")
            .with_metrics(
                {f"{name}_p95_s": float(values.get("p95", 0.0)) / 1000.0 for name, values in report["seconds"].items()}
            )
            .with_fields(rung="p1_update", dataset="synthetic", decision_set="n/a", **report)
            .build()
        )
        run_id = ledger.append(row).run_id
    report["ledger_run_id"] = run_id

    out = Path(args.out)
    if not out.is_absolute():
        out = acis_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    stand_in = "" if report["submission_capable"] else "  (stand-in: the embedding cost of a real encoder is absent)"
    print(f"encoder : {report['encoder']}{stand_in}")
    print(f"corpus  : {report['n_units']} units, first build {report['first_build_seconds']}s")
    print()
    for name in ("change_1", "change_10", "change_100", "full_rebuild"):
        p95_ms = report["seconds"][name].get("p95")
        if p95_ms is None:
            continue
        seconds = p95_ms / 1000.0
        verdict = "ok" if seconds <= TARGET_S else "OVER"
        print(f"{name:<14} p95 {seconds:7.2f}s   target {TARGET_S:>5.1f}s   {verdict}")
    print(f"\nwritten : {out}" + (f"   ledger={run_id}" if run_id else "   (not recorded)"))

    if not args.keep:
        shutil.rmtree(acis_home() / "repos" / REPO, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
