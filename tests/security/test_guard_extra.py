"""Extra guard-hook contract rows (v1.1.1): the physical-seal directory outside the repo, and false-positive guards.
Complements tests/security/test_guard_hook.py (which is the primary contract)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
GUARD = ROOT / ".claude" / "hooks" / "guard.py"


def run_guard(tool: str, **tool_input) -> str:
    p = subprocess.run(
        [sys.executable, str(GUARD)],
        capture_output=True,
        text=True,
        input=json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(ROOT)}),
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT)},
    )
    if p.returncode == 2:
        return "deny"
    assert p.returncode == 0, p.stderr
    return "ask" if '"ask"' in p.stdout else "allow"


DENY = [
    ("Read", dict(file_path="/home/u/.acis-sealed/hf/datasets/qrels/test-00000.parquet")),
    ("Read", dict(file_path="~/.acis-sealed/x.tsv")),
    ("Bash", dict(command="cat ~/.acis-sealed/hf/qrels/test.tsv")),
    ("Bash", dict(command="cp ~/.acis-sealed/hf/x.parquet /tmp/x.parquet")),
    ("Glob", dict(pattern="**/*qrels*test*")),
    ("Grep", dict(pattern="x", path="data/sealed")),
    ("Bash", dict(command="cat data/raw/qrels/test.tsv | head")),
]
ALLOW = [
    ("Read", dict(file_path="tests/security/test_qrels_guard.py")),  # a test that merely mentions the words
    ("Read", dict(file_path="data/raw/apps/qrels_train.tsv")),
    ("Glob", dict(pattern="**/*.py")),
    ("Grep", dict(pattern="qrels_test")),  # searching FOR a word is fine
    ("Bash", dict(command="uv run pytest tests/security/test_qrels_guard.py -q")),
    ("Bash", dict(command="tail -n 3 runs/ledger.jsonl")),
    ("Bash", dict(command="scripts/job_status.sh bakeoff")),
    (
        "Write",
        dict(file_path="src/acis/rank/ltr.py", content="def predict(self, X):\n    return X\n"),
    ),  # predict is banned only in the adapter
    (
        "Write",
        dict(file_path="tests/security/test_x.py", content="eval('1')\n"),
    ),  # banned constructs apply to src/ only
]
ASK = [
    ("Bash", dict(command="git push origin main")),
    ("Bash", dict(command="uv add lightgbm")),
    ("Edit", dict(file_path="CLAUDE.md", old_string="a", new_string="b")),
    ("Edit", dict(file_path="docs/spec/02-p0-engine.md", old_string="a", new_string="b")),
]


@pytest.mark.parametrize("tool,ti", DENY, ids=lambda x: json.dumps(x)[:70] if isinstance(x, dict) else x)
def test_denied(tool, ti):
    assert run_guard(tool, **ti) == "deny"


@pytest.mark.parametrize("tool,ti", ALLOW, ids=lambda x: json.dumps(x)[:70] if isinstance(x, dict) else x)
def test_allowed(tool, ti):
    assert run_guard(tool, **ti) == "allow"


@pytest.mark.parametrize("tool,ti", ASK, ids=lambda x: json.dumps(x)[:70] if isinstance(x, dict) else x)
def test_asks(tool, ti):
    assert run_guard(tool, **ti) == "ask"


def test_garbage_stdin_fails_open():
    p = subprocess.run([sys.executable, str(GUARD)], input="not json", capture_output=True, text=True)
    assert p.returncode == 0
