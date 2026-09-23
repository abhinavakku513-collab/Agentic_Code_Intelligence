#!/usr/bin/env python
"""`make bench` — the latency and memory figures the scorecard quotes (docs/spec/02 §7, D2).

Resource use is scored by the organisers (FAQ), so these numbers are part of the submission rather than a
developer curiosity. Three of them are measured separately on purpose, because they answer different questions
and only the third is the one spec 02 §7 puts a target on:

* **encode** — one query through the model. With a real encoder this dominates everything else, and it is the
  number a smaller model is chosen to reduce (D4, gate G-M).
* **search (cold query)** — encode plus the retrieval core: what a judge typing a new question waits for.
* **search (repeated query)** — the retrieval core alone, because the query vector is already cached. This is the
  `p95 ≤ 15 ms` line of spec 02 §7, and it is the only way to see the core without the model in front of it.

Everything is recorded in the ledger, so the figures can be quoted with a run id (INV-14). Nothing here decides
anything: it measures, prints the targets it can check, and exits 0 even when a target is missed — a benchmark
that fails the build gets deleted rather than fixed.
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
from acis.core.types import SearchRequest
from acis.embed.factory import build_encoder
from acis.embed.scorecard import peak_rss_mb, percentiles
from acis.engine import AcisEngine
from acis.eval import ledger

#: spec 02 §7, all [E] until measured on the declared host. `None` means "measured, not yet targeted".
TARGETS_MS = {"search_warm": 15.0, "encode_query": 150.0}


def _timed(fn: Any, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return samples


def measure(config_path: str, *, n_queries: int, top_k: int) -> dict[str, Any]:
    config = load_frozen_config(config_path)
    encoder = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=encoder)

    corpus = apps.load_corpus()
    started = time.perf_counter()
    snapshot = engine.build_snapshot(corpus, source="bench")
    build_seconds = time.perf_counter() - started

    queries = list(apps.load_queries().values())[:n_queries]
    for text in queries[:3]:  # warm the interpreter, the BLAS threads and the allocator, not the caches
        engine.search(SearchRequest(query=text, top_k=top_k))

    encode_samples: list[float] = []
    cold_samples: list[float] = []
    warm_samples: list[float] = []
    for text in queries:
        encode_samples += _timed(lambda t=text: encoder.encode([t], is_query=True), 1)
        engine._query_vector_cache.clear()
        cold_samples += _timed(lambda t=text: engine.search(SearchRequest(query=t, top_k=top_k)), 1)
        warm_samples += _timed(lambda t=text: engine.search(SearchRequest(query=t, top_k=top_k)), 1)

    return {
        "config": config_path,
        "config_hash": config.config_hash,
        "encoder": encoder.name,
        "submission_capable": encoder.submission_capable,
        "numeric_profile": config.numeric_profile,
        "threads": engine.threads,
        "n_units": snapshot.n_units,
        "n_queries": len(queries),
        "top_k": top_k,
        "snapshot_build_seconds": round(build_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "latency_ms": {
            "encode_query": percentiles(encode_samples),
            "search_cold_query": percentiles(cold_samples),
            "search_warm": percentiles(warm_samples),
        },
    }


def check_targets(report: dict[str, Any]) -> list[str]:
    """Compare p95s against spec 02 §7. Returns the lines to print; a miss is reported, never raised."""
    lines = []
    for name, target in TARGETS_MS.items():
        measured = report["latency_ms"].get(name, {}).get("p95")
        if measured is None:
            continue
        verdict = "ok" if measured <= target else "OVER"
        lines.append(f"{name:<18} p95 {measured:8.2f} ms   target {target:>6.1f} ms   {verdict}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="latency and memory benchmarks (docs/spec/02 §7)")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="runs/bench.json")
    parser.add_argument("--no-ledger", action="store_true", help="measure without recording (a scratch run)")
    args = parser.parse_args(argv)

    if not apps.is_available():
        print("bench: dataset assets are missing — run `make fetch` first", file=sys.stderr)
        return 2

    report = measure(args.config, n_queries=args.queries, top_k=args.top_k)

    run_id = ""
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="bench")
            .with_metrics(
                {
                    f"{name}_p95_ms": float(values.get("p95", 0.0))
                    for name, values in report["latency_ms"].items()
                    if values
                }
            )
            .with_fields(rung="bench", dataset="dev", decision_set="n/a", **report)
            .build()
        )
        run_id = ledger.append(row).run_id
    report["ledger_run_id"] = run_id

    out = Path(args.out)
    if not out.is_absolute():
        out = acis_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    stand_in = "" if report["submission_capable"] else "  (stand-in: not submission-capable)"
    print(f"encoder : {report['encoder']}{stand_in}")
    print(f"corpus  : {report['n_units']} units, snapshot built in {report['snapshot_build_seconds']}s")
    print(f"threads : {report['threads']}   profile {report['numeric_profile']}   peak RSS {report['peak_rss_mb']} MB")
    print()
    for line in check_targets(report):
        print(line)
    print(f"\nwritten : {out}" + (f"   ledger={run_id}" if run_id else "   (not recorded)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
