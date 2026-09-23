#!/usr/bin/env python
"""The commit-stream demo: a corpus changing while it is being searched (docs/spec/04 §6, docs/spec/09 §6).

This is the P1 claim made visible. A version lands every few seconds, the freshness gauge shows how long ago the
index caught up, and the same query is re-run against the new version each time — so the thing on screen is the
engine keeping up, not a slide claiming it does.

It also does the two things a demo usually skips because they are awkward: it **rolls back** (instant, because
snapshots are immutable and activation is a ref rename), and it **kills a build with SIGKILL** and recovers, so
the audience sees that an interrupted build leaves either the old version or the new one and never a mixture.

Everything is local and offline. `--speed` controls the cadence; `--kill` injects the crash.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.core.types import SearchRequest, SourceSpec
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval.appsevolve import EDIT_OPERATORS, generate
from acis.store import snapshots

REPO = "demo-stream"
SEEDS = [
    (
        "binary_search.py",
        "def binary_search(xs, target):\n"
        "    lo, hi = 0, len(xs)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid\n"
        "    return lo\n",
    ),
    (
        "dijkstra.py",
        "import heapq\n\n\n"
        "def dijkstra(graph, start):\n"
        "    dist = {start: 0}\n"
        "    queue = [(0, start)]\n"
        "    while queue:\n"
        "        d, node = heapq.heappop(queue)\n"
        "        for nxt, w in graph.get(node, []):\n"
        "            if d + w < dist.get(nxt, 1 << 60):\n"
        "                dist[nxt] = d + w\n"
        "                heapq.heappush(queue, (d + w, nxt))\n"
        "    return dist\n",
    ),
    (
        "modpow.py",
        "MOD = 10**9 + 7\n\n\n"
        "def power(base, exponent):\n"
        "    result = 1\n"
        "    while exponent:\n"
        "        if exponent & 1:\n"
        "            result = result * base % MOD\n"
        "        base = base * base % MOD\n"
        "        exponent >>= 1\n"
        "    return result\n",
    ),
    (
        "palindrome.py",
        "def is_palindrome(text):\n"
        "    cleaned = [c.lower() for c in text if c.isalnum()]\n"
        "    return cleaned == cleaned[::-1]\n",
    ),
]

BOLD, DIM, GREEN, YELLOW, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


def banner(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}\n{DIM}{'─' * len(text)}{RESET}")


def show(engine: AcisEngine, query: str, *, label: str, top_k: int = 3) -> None:
    started = time.perf_counter()
    response = engine.search(SearchRequest(query=query, repo_id=REPO, version="latest", top_k=top_k))
    elapsed = (time.perf_counter() - started) * 1000
    print(f"  {DIM}query{RESET} {query!r}   {DIM}version{RESET} {CYAN}{label}{RESET}   {DIM}{elapsed:.0f} ms{RESET}")
    for hit in response.results:
        first = next((line for line in hit.source.splitlines() if line.strip()), "")
        print(f"    {hit.rank}. {hit.unit.key:<22} {hit.score:8.4f}  {DIM}{first[:56]}{RESET}")


def freshness(engine: AcisEngine, since: float) -> str:
    active = snapshots.active_snapshot_id(REPO)
    age = time.time() - since
    colour = GREEN if age < 30 else YELLOW
    return f"{DIM}active{RESET} {active[:14] if active else '-'}  {colour}fresh {age:4.1f}s ago{RESET}"


def kill_a_build(home: Path, units: dict[str, str], label: str) -> None:
    """Interrupt a real build with SIGKILL, then recover. No handler runs — this is what a power cut looks like."""
    child = (
        "import os, json, sys\n"
        "from acis.core.config import load_frozen_config\n"
        "from acis.embed.factory import build_encoder\n"
        "from acis.engine import AcisEngine\n"
        "from acis.core.types import SourceSpec\n"
        "cfg = load_frozen_config('configs/dev.yaml')\n"
        "eng = AcisEngine.from_config(cfg, encoder=build_encoder(cfg))\n"
        "os.environ['ACIS_CRASH_AT'] = 'after_manifest'\n"
        f"eng.update_version({REPO!r}, SourceSpec(kind='memory', location='stream',\n"
        f"    options={{'versions': {{{label!r}: {units!r}}}}}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=acis_root(),
        env={**os.environ, "ACIS_HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    signalled = result.returncode in (-9, 137)
    print(f"  {YELLOW}build killed mid-flight{RESET} (SIGKILL {'delivered' if signalled else 'not delivered'})")
    report = snapshots.recover(REPO)
    print(
        f"  recovery: removed {report['removed_partial_builds']} unfinished build(s); "
        f"active is {report['active'][:14] if report['active'] else 'none'}"
    )
    if report["incidents"]:
        print(f"  {YELLOW}incidents{RESET}: {'; '.join(report['incidents'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="commit-stream demo (P1 + Bonus)")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--versions", type=int, default=6)
    parser.add_argument("--speed", type=float, default=1.5, help="seconds between commits")
    parser.add_argument("--query", default="shortest path in a weighted graph using a priority queue")
    parser.add_argument("--kill", action="store_true", help="kill one build mid-flight and recover")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true", help="keep the structured logs on stderr")
    args = parser.parse_args(argv)

    if not args.verbose:
        # A demo is read, not grepped: the structured log stays available (`--verbose`) but is not printed over
        # the thing the audience is looking at.
        from acis.obs.log import configure

        configure("CRITICAL")

    home = Path(os.environ.get("ACIS_HOME", str(Path.home() / ".acis" / "home")))
    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    encoder = engine.encoder

    banner("ACIS · commit stream")
    print(
        f"  encoder {getattr(encoder, 'name', 'none')}"
        f"{'' if getattr(encoder, 'submission_capable', False) else DIM + '  (stand-in encoder)' + RESET}"
    )
    print(f"  corpus  {len(SEEDS)} units, {args.versions} versions at {args.speed}s apart")

    evolution = generate(
        SEEDS, n_versions=args.versions, seed=args.seed, edits_per_version=1, operators=list(EDIT_OPERATORS)
    )
    rng = random.Random(args.seed)

    banner("v1 · the first index")
    first = dict(evolution.units(evolution.labels[0]))
    report = engine.ingest(
        SourceSpec(kind="memory", location="stream", options={"versions": {evolution.labels[0]: first}}),
        repo_id=REPO,
    )
    built_at = time.time()
    build = engine.build_reports(REPO)[-1]
    print(f"  {report.state}: {build.units_total} units, {build.units_new} embedded, {build.seconds:.2f}s")
    show(engine, args.query, label=evolution.labels[0])

    banner("commits arriving")
    for label in evolution.labels[1:]:
        time.sleep(max(0.0, args.speed))
        units = dict(evolution.units(label))
        if args.kill and label == evolution.labels[len(evolution.labels) // 2]:
            kill_a_build(home, units, label)
            continue
        started = time.perf_counter()
        build = engine.update_version(
            REPO, SourceSpec(kind="memory", location="stream", options={"versions": {label: units}})
        )
        built_at = time.time()
        searchable = time.perf_counter() - started
        changed = [k for k in units if units[k] != first.get(k)]
        print(
            f"\n  {BOLD}{label}{RESET}  {build.units_new} new / {build.units_total} units  "
            f"{GREEN}searchable in {searchable:.2f}s{RESET}   {freshness(engine, built_at)}"
        )
        if changed:
            print(f"    {DIM}changed: {', '.join(sorted(changed)[:4])}{RESET}")
        show(engine, args.query, label=label, top_k=2)
        first = units

    banner("rollback")
    previous = engine.rollback(REPO)
    print(f"  active is now {previous[:14]} {DIM}(a ref rename: nothing was rebuilt){RESET}")
    show(engine, args.query, label="latest (rolled back)", top_k=2)

    banner("Bonus · one answer per lineage")
    from acis.core.types import EvolveRequest

    response = engine.retrieve_evolution(EvolveRequest(query=args.query, repo_id=REPO, top_k=4))
    for note in response.degradations:
        print(f"  {DIM}{note}{RESET}")
    for position, group in enumerate(response.groups, start=1):
        best = group["best"]
        marks = "".join("●" if step["changed"] else "○" for step in group["timeline"])
        print(
            f"  {position}. {best['key']:<22} {group['score']:7.4f}  best={best['version']:<4} "
            f"{group['n_revisions']} revisions  {CYAN}{marks}{RESET}  {group['span'][0]}→{group['span'][1]}"
        )

    _ = rng
    print(f"\n{DIM}Nothing above left this machine. Every result was re-read from the content store by hash.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
