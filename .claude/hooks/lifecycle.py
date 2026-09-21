#!/usr/bin/env python3
"""ACIS lifecycle hooks (stdlib only). Usage: lifecycle.py {session|post|stop}

session  SessionStart (startup/resume/clear/compact): plain stdout becomes context for Claude.
post     PostToolUse after Edit/Write/MultiEdit: ruff format + check --fix on the edited .py file;
         exit 2 (stderr shown to Claude) only for issues ruff cannot fix.
stop     Stop: if Python files changed, the fast unit tests must pass or Claude keeps working.
         Never traps a session: honours stop_hook_active and caps blocks at 2 per session.
Every path fails OPEN when tooling is missing (no ruff / no .venv / no tests yet).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # missing binary, timeout, ...
        return 127, str(e)


def venv_tool(name: str) -> str | None:
    for base in (ROOT / ".venv" / "bin", ROOT / ".venv" / "Scripts"):
        for ext in ("", ".exe"):
            c = base / (name + ext)
            if c.exists():
                return str(c)
    return shutil.which(name) if name == "ruff" else None


def session(_: dict) -> int:
    out = ["ACIS session context (.claude/hooks/lifecycle.py):"]
    status = ROOT / "docs" / "STATUS.md"
    if status.exists():
        out += status.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
    else:
        out.append("docs/STATUS.md is missing: restore it before starting work.")
    if status.exists():
        s = status.read_text(encoding="utf-8", errors="replace")
        for key, msg in (("hook_canary:", "hook canary (G0.0) not passed: hooks are unverified — run /canary (docs/CANARY.md)"),
                         ("deadline:", "deadline undeclared: ask the owner for the real date (docs/TRIAGE.md)"),
                         ("compute:", "compute undeclared: ask the owner for cores/RAM, GPU access and who runs the cold pass")):
            line = next((ln for ln in s.splitlines() if ln.startswith(key)), "")
            if (key == "hook_canary:" and "passed" not in line) or (key != "hook_canary:" and "undeclared" in line):
                out.append("WARNING: " + msg)
    faq = ROOT / "docs" / "official" / "FAQ.md"
    if not faq.exists() or "PASTE THE FAQ" in faq.read_text(encoding="utf-8", errors="replace"):
        out.append("WARNING: docs/official/FAQ.md is missing/unfilled; resource-scoring rules may be unknown.")
    rc, g = run(["git", "status", "--porcelain", "-b"], 5)
    if rc == 0 and g.strip():
        lines = g.strip().splitlines()
        out.append(f"git: {lines[0]} | {len(lines) - 1} changed path(s)")
    ledger = ROOT / "runs" / "ledger.jsonl"
    if ledger.exists():
        rows = [ln for ln in ledger.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        touches = 0
        for ln in rows:
            try:
                touches = max(touches, int(json.loads(ln).get("test_touch_count", 0)))
            except Exception:
                pass
        out.append(f"ledger: {len(rows)} run(s); TEST touches used: {touches} (budget in configs/official.yaml)")
    out.append("Reminders: TEST labels outside the tree (INV-8) | phase gates in docs/STATUS.md | numbers only from the ledger | /status, /phase-start N, /phase-gate N.")
    try:
        print("\n".join(out)[:6000])
    except BrokenPipeError:  # reader closed early; nothing to do
        pass
    return 0


def post(d: dict) -> int:
    fp = (d.get("tool_input") or {}).get("file_path")
    if not fp or not str(fp).endswith(".py"):
        return 0
    p = Path(fp) if os.path.isabs(fp) else ROOT / fp
    ruff = venv_tool("ruff")
    if not p.exists() or not ruff:
        return 0
    run([ruff, "format", "--quiet", str(p)])
    rc, out = run([ruff, "check", "--fix", "--quiet", str(p)])
    if rc != 0:
        sys.stderr.write(f"ruff reports issues it cannot auto-fix in {p.name}:\n{out[-1500:]}\nFix them before continuing.\n")
        return 2
    return 0


def stop(d: dict) -> int:
    if d.get("stop_hook_active"):
        return 0
    counter = Path(tempfile.gettempdir()) / f"acis_stop_{d.get('session_id', 'x')}.count"
    n = int(counter.read_text()) if counter.exists() and counter.read_text().strip().isdigit() else 0
    if n >= 2:
        return 0
    rc, out = run(["git", "status", "--porcelain"], 5)
    if rc != 0 or not any(ln[3:].strip().endswith(".py") for ln in out.splitlines()):
        return 0
    py = venv_tool("python")
    if not py or not (ROOT / "tests" / "unit").exists():
        return 0
    rc, out = run([py, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider",
                   "-m", "not slow and not gpu and not network", "tests/unit"], 240)
    if rc in (0, 5) or "No module named pytest" in out:
        return 0
    counter.write_text(str(n + 1))
    sys.stderr.write("Fast unit tests are failing. Fix them before finishing (CLAUDE.md section 4):\n" + out[-2500:])
    return 2


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        d = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except Exception:
        d = {}
    sys.exit({"session": session, "post": post, "stop": stop}.get(mode, lambda _: 0)(d))


if __name__ == "__main__":
    main()
