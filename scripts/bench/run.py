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
import random
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
TARGETS_MS = {"short_warm": 15.0, "long_warm": 15.0}
#: Encode targets, checked on the encode stage of cold queries (spec 02 §7).
ENCODE_TARGETS_MS = {"short_cold": 150.0, "long_cold": 1500.0}


def _timed(fn: Any, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return samples


#: Short hands-on questions, written for latency only: nothing is tuned to them and their rankings are not scored.
SHORT_QUERIES = (
    "find the shortest path in a weighted graph",
    "reverse a linked list",
    "count inversions in an array",
    "binary search on a sorted list",
    "longest common subsequence of two strings",
    "check whether a number is prime",
    "minimum spanning tree of a graph",
    "sum of digits of a large number",
)


def _fresh(text: str, rng: random.Random) -> str:
    """The same query with a letters-only nonce, so neither the process nor the on-disk vector cache has seen it.

    Letters only: a digit would add a numeric literal and could change the route, i.e. the work being timed.
    """
    return f"{text} {''.join(rng.choice(string.ascii_lowercase) for _ in range(8))}"


def _search_ms(engine: AcisEngine, text: str, top_k: int) -> tuple[float, dict[str, float]]:
    started = time.perf_counter()
    response = engine.search(SearchRequest(query=text, top_k=top_k))
    return (time.perf_counter() - started) * 1000, dict(response.timings_ms)


def measure(config_path: str, *, n_queries: int, top_k: int, pause: float = 0.5) -> dict[str, Any]:
    """Cold and warm, short and long, one at a time and four at once.

    * **cold** — text neither the process nor the on-disk vector cache has seen: what a judge typing a new question
      waits for (model included).
    * **warm** — the same text again: the query vector is cached, so this is the retrieval core.
    * **paced** — a pause between queries, as a person types. Back-to-back loops overstate the encoder: numpy's
      BLAS threads keep spinning after the dense matmul and slow the next forward pass (~+70 ms measured).
    * **concurrent** — four requests at once, which the API allows; the encoder serialises them and the queueing
      is reported as its own stage (`encode_wait`).
    """
    config = load_frozen_config(config_path)
    encoder = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=encoder)

    started = time.perf_counter()
    snapshot = engine.build_snapshot(apps.load_corpus(), source="bench")
    engine.snapshot_data(snapshot).warm_features()
    build_seconds = time.perf_counter() - started

    rng = random.Random(0)  # which dev queries are sampled: fixed
    nonce = random.Random()  # the nonces: fresh every run, or a second run would hit the first one's cache
    dev = apps.load_queries()
    ids = sorted(dev)
    long_texts = [dev[q] for q in rng.sample(ids, n_queries)]
    short_texts = [SHORT_QUERIES[i % len(SHORT_QUERIES)] for i in range(n_queries)]
    for text in ("warm-up query one", "warm-up query two"):  # interpreter, allocator, thread pools — not caches
        _search_ms(engine, _fresh(text, nonce), top_k)

    samples: dict[str, list[float]] = {}
    stage_samples: dict[str, dict[str, list[float]]] = {}
    routes: dict[str, dict[str, int]] = {}

    def record(name: str, ms: float, timings: dict[str, float]) -> None:
        samples.setdefault(name, []).append(ms / 1000)
        for key, value in timings.items():
            if key.startswith("stage."):
                stage_samples.setdefault(name, {}).setdefault(key[6:], []).append(value / 1000)

    for kind, texts in (("short", short_texts), ("long", long_texts)):
        for text in texts:
            fresh = _fresh(text, nonce)
            time.sleep(pause)
            ms, timings = _search_ms(engine, fresh, top_k)
            record(f"{kind}_cold", ms, timings)
            route = engine.route(engine.normalise_query(fresh)[0])
            routes.setdefault(kind, {}).setdefault(route, 0)
            routes[kind][route] += 1
            time.sleep(pause)
            ms, timings = _search_ms(engine, fresh, top_k)
            record(f"{kind}_warm", ms, timings)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, n_queries, 4):
            time.sleep(pause)
            batch = [_fresh(short_texts[(i + j) % len(short_texts)], nonce) for j in range(4)]
            for ms, timings in pool.map(lambda t: _search_ms(engine, t, top_k), batch):
                record("short_cold_4_concurrent", ms, timings)

    return {
        "config": config_path,
        "config_hash": config.config_hash,
        "encoder": encoder.name,
        "submission_capable": encoder.submission_capable,
        "numeric_profile": config.numeric_profile,
        "threads": engine.threads,
        "blas_threads": engine.blas_threads,  # None = the process default (16 on the dev host)
        "n_units": snapshot.n_units,
        "n_queries": n_queries,
        "top_k": top_k,
        "pause_s": pause,
        "routes": routes,
        "snapshot_build_seconds": round(build_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "latency_ms": {name: percentiles(values) for name, values in samples.items()},
        "stage_ms": {
            name: {stage: percentiles(values) for stage, values in stages.items()}
            for name, stages in stage_samples.items()
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
        lines.append(f"{name + ' search':<26} p95 {measured:8.2f} ms   target {target:>7.1f} ms   {verdict}")
    for name, target in ENCODE_TARGETS_MS.items():
        stages = report["stage_ms"].get(name, {})
        encode = [stages[s]["p95"] for s in ("route_encode", "encode") if s in stages and stages[s]]
        if not encode:
            continue
        measured = max(encode)
        verdict = "ok" if measured <= target else "OVER"
        lines.append(f"{name + ' encode':<26} p95 {measured:8.2f} ms   target {target:>7.1f} ms   {verdict}")
    lines.append("")
    for name, values in report["latency_ms"].items():
        lines.append(f"{name:<26} p50 {values['p50']:8.1f} ms   p95 {values['p95']:8.1f} ms")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="latency and memory benchmarks (docs/spec/02 §7)")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between queries, as a person types")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="runs/bench.json")
    parser.add_argument("--no-ledger", action="store_true", help="measure without recording (a scratch run)")
    parser.add_argument(
        "--without-forward-lock",
        action="store_true",
        help="the pre-fix behaviour (concurrent forward passes), for a before/after comparison only",
    )
    args = parser.parse_args(argv)
    if args.without_forward_lock:
        from acis.embed import runtime

        class _NoLock:
            def acquire(self) -> bool:
                return True

            def release(self) -> None:
                return None

        runtime._FORWARD_LOCK = _NoLock()  # type: ignore[assignment]

    if not apps.is_available():
        print("bench: dataset assets are missing — run `make fetch` first", file=sys.stderr)
        return 2

    report = measure(args.config, n_queries=args.queries, top_k=args.top_k, pause=args.pause)

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
            .with_fields(
                rung="bench", dataset="dev", decision_set="n/a", forward_lock=not args.without_forward_lock, **report
            )
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
