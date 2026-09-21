"""Behavioural tests for .claude/hooks/guard.py (the PreToolUse hook).

Run:  uv run pytest tests/security/test_guard_hook.py -q
Every row is an executed contract: DENY (exit 2), ASK (JSON on stdout), ALLOW (silent exit 0).
The GAP_ rows document what a regex hook can NOT see - they are the reason the TEST seal must be physical
(labels outside the repo tree), see docs/spec/09. They are marked xfail(strict=True) so that if the hook ever starts
catching them, the test tells you to promote the row.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
GUARD = ROOT / ".claude" / "hooks" / "guard.py"
HF = "/home/u/.cache/huggingface/hub/datasets--CoIR-Retrieval--apps/snapshots/f22508f/default/test-00000-of-00001.parquet"


def run_guard(tool: str, **tool_input) -> str:
    p = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(ROOT)}),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT)},
    )
    if p.returncode == 2:
        return "deny"
    if p.returncode == 0 and '"ask"' in p.stdout:
        return "ask"
    assert p.returncode == 0, p.stderr
    return "allow"


DENY = [
    ("Read", dict(file_path="data/sealed/apps_test_qrels.tsv")),
    ("Read", dict(file_path="data/hf/qrels/test.tsv")),
    ("Read", dict(file_path=HF)),  # where mteb/datasets actually keep TEST qrels (best effort)
    ("Bash", dict(command="cat data/sealed/x.tsv")),
    ("Bash", dict(command=f"head -3 {HF}")),
    ("Write", dict(file_path="src/acis/x.py", content="y = eval(user)\n")),
    ("Write", dict(file_path="src/acis/x.py", content="p = torch.load(f, map_location='cpu')\n")),
    ("Write", dict(file_path="src/acis/x.py", content="subprocess.run(cmd, shell = True)\n")),
    ("Write", dict(file_path="src/acis/mteb_adapter.py", content="class A:\n    def predict(self, x): ...\n")),
    ("Edit", dict(file_path=".claude/hooks/guard.py", old_string="a", new_string="b")),
    ("Bash", dict(command="pip install lightgbm")),
    ("Bash", dict(command="git push --force origin main")),
    ("Bash", dict(command="git commit --no-verify -m x")),
    # ledger / split-lock writes, however they are spelled
    ("Bash", dict(command="sed -i 's/a/b/' runs/ledger.jsonl")),
    ("Bash", dict(command="echo x >> runs/ledger.jsonl")),
    ("Bash", dict(command="cat foo | tee -a runs/ledger.jsonl")),
    ("Bash", dict(command="cp /dev/null runs/ledger.jsonl")),
    ("Bash", dict(command="python3 -c \"open('runs/ledger.jsonl','a').write('x')\"")),
    ("Bash", dict(command="dd if=/dev/zero of=configs/splits.lock.json")),
]
ASK = [
    ("Edit", dict(file_path="CLAUDE.md", old_string="a", new_string="b")),
    ("Write", dict(file_path="configs/gates/G3.yaml", content="x")),
    ("Bash", dict(command="uv add lightgbm")),
    ("Bash", dict(command="curl -s https://huggingface.co/x")),
    ("Bash", dict(command="grep -r corpus- data/")),
    ("Bash", dict(command="find data -name '*.tsv' | xargs head -3")),
    ("Grep", dict(pattern="corpus-", path="data")),
]
ALLOW = [
    ("Read", dict(file_path="tests/security/test_qrels_guard.py")),
    ("Read", dict(file_path="docs/spec/02-p0-engine.md")),
    ("Glob", dict(pattern="tests/**/test_label*.py")),  # v1.0 false positive
    ("Grep", dict(pattern="def foo", path="src")),
    ("Bash", dict(command="uv run pytest -q 2>&1 | tail -5; git add runs/ledger.jsonl")),  # v1.0 false positive
    ("Bash", dict(command="git add runs/ledger.jsonl && git commit -m 'chore: ledger'")),
    ("Bash", dict(command="cat runs/ledger.jsonl | tail -3")),
    ("Bash", dict(command="cp runs/ledger.jsonl /tmp/ledger.bak")),  # copying FROM the ledger is a read
    ("Bash", dict(command="uv run python scripts/report.py > /tmp/out.txt")),
    ("Write", dict(file_path="src/acis/x.py", content="model.eval()\nx = ast.literal_eval(s)\n")),
    (
        "Write",
        dict(file_path="src/acis/x.py", content="# never call eval( on repo text\nx = 1\n"),
    ),  # v1.0 false positive
    ("Write", dict(file_path="docs/STATUS.md", content="x")),
    ("Write", dict(file_path="configs/targets.yaml", content="x")),
]
GAPS = [  # a regex hook cannot see these; the seal must be physical (docs/spec/09)
    ("Bash", dict(command="cat data/*/*.tsv")),
    ("Bash", dict(command="uv run python scripts/peek.py")),
    ("Bash", dict(command="grep -r corpus- .")),
]


@pytest.mark.parametrize("tool,ti", DENY, ids=lambda x: json.dumps(x)[:60] if isinstance(x, dict) else x)
def test_denied(tool, ti):
    assert run_guard(tool, **ti) == "deny"


@pytest.mark.parametrize("tool,ti", ASK, ids=lambda x: json.dumps(x)[:60] if isinstance(x, dict) else x)
def test_asks(tool, ti):
    assert run_guard(tool, **ti) == "ask"


@pytest.mark.parametrize("tool,ti", ALLOW, ids=lambda x: json.dumps(x)[:60] if isinstance(x, dict) else x)
def test_allowed(tool, ti):
    assert run_guard(tool, **ti) == "allow"


@pytest.mark.xfail(
    strict=True, reason="regex hooks cannot see expansion/indirection: keep TEST labels outside the repo tree"
)
@pytest.mark.parametrize("tool,ti", GAPS, ids=lambda x: json.dumps(x)[:60] if isinstance(x, dict) else x)
def test_known_gaps(tool, ti):
    assert run_guard(tool, **ti) != "allow"


def test_malformed_input_fails_open():
    p = subprocess.run([sys.executable, str(GUARD)], input="not json", capture_output=True, text=True)
    assert p.returncode == 0
