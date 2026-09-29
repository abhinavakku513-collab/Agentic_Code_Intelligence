#!/usr/bin/env python
"""P1 and Bonus on a real git repository with hand-made commits — not the synthetic mutator (docs/spec/04).

The page's "synthetic commit" button changes stored text with Apps-Evolve's edit operators; it demonstrates
incremental rebuilds, but it is not ingestion. This script makes a real repository with `git` itself — one
commit per kind of change a developer makes — ingests it through `acis`'s git source (read-only, no checkout), and
reports what ACIS shows after each commit:

1. `initial`   — three files;
2. `edit`      — one function body changed;
3. `move`      — `git mv` of a file into a package, content unchanged;
4. `unrelated` — a new, unrelated file;
5. `revert`    — `git revert` of the edit, so a file returns to content the store has already seen.

For every version: units, units newly embedded vs reused (from the build report, i.e. the CAS), the diff against
the previous version, the top hit of a pinned search; then the lineage the Bonus builds across all versions.
Runs in a temporary ACIS_HOME, so the live catalog is untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GRAPH_V1 = '''import heapq


def dijkstra(graph, source):
    """Shortest distances from source in a graph with non-negative edge weights."""
    dist = {source: 0}
    heap = [(0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
            continue
        for neighbour, weight in graph[node]:
            nd = d + weight
            if nd < dist.get(neighbour, float("inf")):
                dist[neighbour] = nd
                heapq.heappush(heap, (nd, neighbour))
    return dist
'''
GRAPH_V2 = GRAPH_V1.replace(
    "def dijkstra(graph, source):",
    "def dijkstra(graph, source, target=None):",
).replace(
    "        d, node = heapq.heappop(heap)\n",
    "        d, node = heapq.heappop(heap)\n        if node == target:\n            break\n",
)
TEXT = """def reverse_words(sentence):
    return " ".join(reversed(sentence.split()))


def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
"""
NUMBERS = """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def primes_below(n):
    sieve = [True] * n
    sieve[0:2] = [False, False]
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = [False] * len(range(i * i, n, i))
    return [i for i, p in enumerate(sieve) if p]
"""
CONFIG = """import configparser


def load_settings(path):
    parser = configparser.ConfigParser()
    parser.read(path)
    return {section: dict(parser[section]) for section in parser.sections()}
"""


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "acis-check",
        "GIT_AUTHOR_EMAIL": "check@localhost",
        "GIT_COMMITTER_NAME": "acis-check",
        "GIT_COMMITTER_EMAIL": "check@localhost",
    }
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def make_repo(repo: Path) -> list[tuple[str, str]]:
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    steps = []
    (repo / "graph.py").write_text(GRAPH_V1)
    (repo / "text_utils.py").write_text(TEXT)
    (repo / "numbers_util.py").write_text(NUMBERS)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial: graph, text, numbers")
    steps.append(("initial", git(repo, "rev-parse", "HEAD").strip()))
    (repo / "graph.py").write_text(GRAPH_V2)
    git(repo, "commit", "-q", "-am", "edit: dijkstra stops at an optional target")
    edit = git(repo, "rev-parse", "HEAD").strip()
    steps.append(("edit", edit))
    (repo / "strings").mkdir()
    git(repo, "mv", "text_utils.py", "strings/text.py")
    git(repo, "commit", "-q", "-m", "move: text utils into a package")
    steps.append(("move", git(repo, "rev-parse", "HEAD").strip()))
    (repo / "config_loader.py").write_text(CONFIG)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "unrelated: settings loader")
    steps.append(("unrelated", git(repo, "rev-parse", "HEAD").strip()))
    git(repo, "revert", "--no-edit", edit)
    steps.append(("revert", git(repo, "rev-parse", "HEAD").strip()))
    return steps


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="acis-real-git-"))
    home = work / "home"
    home.mkdir(parents=True)
    # The pinned weights are read from the real home (read-only); the store, catalog and caches start empty.
    real_home = Path(os.environ.get("ACIS_HOME", Path.home() / ".acis" / "home"))
    (home / "models").symlink_to(real_home / "models", target_is_directory=True)
    os.environ["ACIS_HOME"] = str(home)
    steps = make_repo(work / "repo")
    kinds = {sha[:12]: kind for kind, sha in steps}

    from acis.core.config import load_frozen_config
    from acis.core.types import EvolveRequest, SearchRequest, SourceSpec
    from acis.embed.factory import build_encoder
    from acis.engine import AcisEngine

    config = load_frozen_config("configs/dev.yaml")
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    engine.ingest(SourceSpec(kind="git", location=str(work / "repo")), repo_id="real-git")
    reports = engine.build_reports("real-git")
    versions = engine.versions("real-git")

    out: dict[str, object] = {"versions": []}
    previous = None
    for row, report in zip(versions, reports, strict=True):
        label = str(row["label"])
        entry: dict[str, object] = {
            "commit": label,
            "kind": kinds.get(label, "?"),
            "snapshot": row["snapshot_id"],
            "units": report.units_total,
            "units_embedded_new": report.units_new,
            "units_reused": report.units_reused,
            "build_seconds": report.seconds,
        }
        if previous is not None:
            diff = engine.compare_versions("real-git", previous, label)
            entry["diff"] = {"added": list(diff.added), "removed": list(diff.removed), "changed": list(diff.changed)}
        hit = engine.search(
            SearchRequest(query="shortest path in a weighted graph", repo_id="real-git", version=label, top_k=1)
        ).results[0]
        entry["pinned_search_top1"] = {
            "key": hit.unit.key,
            "version": hit.unit.version_id,
            "body_hash": hit.unit.body_hash[:12],
        }
        out["versions"].append(entry)
        previous = label

    evolve = engine.retrieve_evolution(
        EvolveRequest(query="reverse the words of a sentence", repo_id="real-git", top_k=3)
    )
    out["lineage_text"] = [
        {
            "lineage": g["lineage_id"],
            "best": {"key": g["best"]["key"], "version": g["best"]["version"]},
            "timeline": [
                {"version": s["version"], "key": s.get("key"), "relation": s["relation"], "changed": s["changed"]}
                for s in g["timeline"]
            ],
        }
        for g in evolve.groups[:1]
    ]
    evolve = engine.retrieve_evolution(EvolveRequest(query="dijkstra shortest path heap", repo_id="real-git", top_k=3))
    out["lineage_graph"] = [
        {
            "lineage": g["lineage_id"],
            "n_revisions": g["n_revisions"],
            "timeline": [
                {
                    "version": s["version"],
                    "key": s.get("key"),
                    "relation": s["relation"],
                    "changed": s["changed"],
                    "body": s.get("body_hash", "")[:12],
                }
                for s in g["timeline"]
            ],
        }
        for g in evolve.groups[:1]
    ]
    out["lineages_total"] = engine.lineage_index("real-git").size
    json.dump(out, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
