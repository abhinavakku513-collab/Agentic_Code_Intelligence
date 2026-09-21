"""Shared pytest configuration. Paths resolve from the repo root (D16), never the CWD; network/GPU tests are opt-in."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CLAUDE_PROJECT_DIR", str(ROOT))
os.environ.setdefault("ACIS_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def pytest_collection_modifyitems(config, items):
    skip = {
        "network": pytest.mark.skip(reason="set ACIS_RUN_NETWORK=1 to run network tests"),
        "gpu": pytest.mark.skip(reason="set ACIS_RUN_GPU=1 to run GPU tests"),
    }
    enabled = {"network": os.environ.get("ACIS_RUN_NETWORK") == "1", "gpu": os.environ.get("ACIS_RUN_GPU") == "1"}
    no_dataset = pytest.mark.skip(reason="dataset assets are missing: run `make fetch`")
    for item in items:
        for name, marker in skip.items():
            if name in item.keywords and not enabled[name]:
                item.add_marker(marker)
        if "dataset" in item.keywords and not _DATASET_AVAILABLE:
            item.add_marker(no_dataset)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


def dataset_available() -> bool:
    """True when the allow-listed dataset assets have been fetched (`make fetch`). Never touches held-out labels."""
    try:
        from acis.appsdata import apps

        return apps.is_available()
    except Exception:  # noqa: BLE001 — a missing dependency must skip, not error
        return False


_DATASET_AVAILABLE = dataset_available()


@pytest.fixture
def tiny_corpus() -> list[tuple[str, str]]:
    """A deterministic, label-free fixture corpus used by the engine and adapter unit tests."""
    return [
        ("d1", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"),
        ("d2", "import sys\nn = int(sys.stdin.readline())\nprint(sum(range(n + 1)))\n"),
        ("d3", "from collections import deque\n\ndef bfs(graph, start):\n    seen, q = {start}, deque([start])\n"),
        (
            "d4",
            "MOD = 1000000007\n\ndef fact(n):\n    r = 1\n    for i in range(2, n + 1):\n        r = r * i % MOD\n    return r\n",
        ),
        ("d5", "s = input().strip()\nprint('YES' if s == s[::-1] else 'NO')\n"),
        ("d6", "import heapq\n\ndef dijkstra(graph, src):\n    dist = {src: 0}\n    pq = [(0, src)]\n"),
        (
            "d7",
            "import bisect\n\ndef lis(xs):\n    tails = []\n    for x in xs:\n        i = bisect.bisect_left(tails, x)\n",
        ),
        ("d8", "for _ in range(int(input())):\n    a, b = map(int, input().split())\n    print(a + b)\n"),
    ]
