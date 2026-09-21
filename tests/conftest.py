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
    for item in items:
        for name, marker in skip.items():
            if name in item.keywords and not enabled[name]:
                item.add_marker(marker)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
