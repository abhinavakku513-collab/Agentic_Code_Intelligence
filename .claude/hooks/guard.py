#!/usr/bin/env python3
"""ACIS PreToolUse guard: deterministic enforcement of CLAUDE.md section 4.

Context is advisory; hooks are not. Contract (Claude Code hooks reference):
  * stdin  = hook JSON (tool_name, tool_input, cwd, ...)
  * exit 2 + stderr            -> BLOCK; Claude sees the reason
  * stdout JSON "ask" decision -> owner is asked to confirm
  * exit 0 with no output      -> no opinion (normal permission flow)
Malformed input fails OPEN (never bricks a session); a matched rule fails CLOSED.
Only the standard library is used so the hook works before any environment exists.

v1.1 changes (all covered by tests/security/test_guard_hook.py):
  * ledger/split-lock protection is segment- and target-aware (no false positive on `2>&1 ...; git add runs/ledger.jsonl`;
    python open(...,"a"), `cp x runs/ledger.jsonl`, `tee`, `dd of=` are now caught)
  * `#` comments are ignored when scanning src/ for banned constructs; Glob patterns no longer trip on `test_label*.py`
  * best-effort match for the HF-cache paths where mteb stores the TEST qrels; recursive searches rooted at data/ ask first
IMPORTANT: this hook is a speed-bump against ACCIDENTAL reads/writes, not a security boundary. The real seal is physical
(keep TEST labels outside the repo tree / a separate HF cache) - see docs/spec/09.
"""
from __future__ import annotations

import json
import os
import re
import sys


def ask(reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "ask",
        "permissionDecisionReason": reason}}))
    sys.exit(0)


def deny(reason: str) -> None:
    sys.stderr.write(
        f"ACIS guard blocked this call: {reason}\n"
        "Rules: CLAUDE.md section 4. If you believe the block is wrong, tell the owner; do not work around it.\n")
    sys.exit(2)


# TEST labels live under data/sealed/ or in data-format files named like *qrels*test* / *test*labels*.
# Source/docs files (.py/.md/...) that merely mention these words (e.g. tests/security/test_qrels_guard.py) are NOT sealed.
DATA_EXT = re.compile(r"\.(tsv|csv|jsonl?|parquet|arrow|txt|gz|zst|npy)$", re.I)
LABEL_NAME = re.compile(r"qrels.{0,4}test|test.{0,4}qrels|(^|[/_.-])(sealed|test)[_-]?labels?", re.I)
SEALED_DIR = re.compile(r"(^|/)(data/sealed|\.acis-sealed)(/|$)", re.I)   # in-repo seal dir + the physical seal outside the repo (docs/spec/09 s3)
# Best-effort: where the HF hub/datasets cache keeps the CoIR-APPS TEST qrels (layout not verified offline).
HF_TEST_LABELS = re.compile(
    r"(datasets--CoIR-Retrieval--apps|CoIR-Retrieval___apps)[^ ]*/(default|qrels)/(?:[^ ]*/)?[^/ ]*test[^/ ]*", re.I)
LABEL_NAME_LOOSE = re.compile(r"qrels.{0,4}test|test.{0,4}qrels|(^|[/_.-])sealed([/_.-]|$)", re.I)
# The only sanctioned readers of sealed labels.
FINAL_EVAL_OK = re.compile(
    r"^\s*(make\s+rc-official|(uv\s+run\s+)?acis\s+eval\s+(official|verify-submission|repro)\b"
    r"|(uv\s+run\s+)?python3?\s+-m\s+acis\.eval\.final\b)")

NO_WRITE = [re.compile(p) for p in (
    r"^configs/splits\.lock\.json$", r"^runs/ledger\.jsonl$", r"^runs/[^/]+/(manifest\.json|SHA256SUMS)$",
    r"^data/", r"^docs/official/", r"^\.claude/hooks/", r"^\.claude/settings(\.local)?\.json$", r"^uv\.lock$",
)]
ASK_WRITE = [re.compile(p) for p in (
    r"^CLAUDE\.md$", r"^AGENTS\.md$", r"^docs/spec/", r"^\.claude/(rules|agents|skills)/",
    r"^configs/(official\.yaml|gates/)",
)]
# Constructs banned in src/ (INV-5, supply-chain rules). Checked on the text about to be written.
SRC_BANNED = [(re.compile(p), why) for p, why in (
    (r"trust_remote_code\s*=\s*True", "trust_remote_code=True is forbidden (supply chain)"),
    (r"\bpickle\.loads?\(|\bcPickle\b|\bmarshal\.loads?\(", "pickle/marshal deserialisation is forbidden"),
    (r"\btorch\.load\((?![^)]*weights_only\s*=\s*True)", "torch.load needs weights_only=True (prefer safetensors)"),
    (r"\byaml\.load\((?![^)]*SafeLoader)", "use yaml.safe_load"),
    (r"shell\s*=\s*True", "subprocess shell=True is forbidden"),
    (r"(?<![\w.])(eval|exec)\(", "eval/exec are forbidden: untrusted content is parsed, never executed (INV-5)"),
)]
ADAPTER_BANNED = [(re.compile(r"def\s+predict\s*\("),
                   "PrePostPipelineEncoder must never define predict(): mteb would route it to the cross-encoder wrapper")]

BASH_DENY = [(re.compile(p), why) for p, why in (
    (r"(^|[;&|]\s*)sudo\b", "sudo is not allowed"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*(/|~|\$HOME|\.\.|\*|\.)(\s|$)",
     "destructive rm outside a specific path"),
    (r"git\s+push\b[^;&|]*(--force\b|--force-with-lease\b|\s-f\b)", "force-push is forbidden"),
    (r"--no-verify\b", "--no-verify bypasses the quality gates"),
    (r"(curl|wget)[^|;&]*\|\s*(sudo\s+)?(ba|z)?sh\b", "piping a download into a shell is forbidden"),
    (r"\bpip3?\s+install\b|python3?\s+-m\s+pip\s+install", "use `uv add` (with an ADR) / `uv sync`, never pip install"),
)]
PROT = r"(?:runs/ledger\.jsonl|configs/splits\.lock\.json)"
_Q = "['\"]"
_WRITE_PATTERNS = [re.compile(p) for p in (
    r"(?<![&])>>?\s*" + _Q + r"?\S*" + PROT,                                          # > / >> / 2> onto the file (not `2>&1`)
    r"\btee\b(?:\s+-\S+)*\s+" + _Q + r"?\S*" + PROT,                                   # tee [-a] file
    r"\b(?:sed|perl)\s+-[a-zA-Z]*i[^;&|\n]*" + PROT,                                   # in-place edit
    r"\b(?:rm|truncate|shred)\b[^;&|\n]*" + PROT,
    r"\b(?:cp|mv|install|ln)\b(?:\s+-\S+)*(?:\s+\S+)+?\s+" + _Q + r"?\S*" + PROT + _Q + r"?\s*$",   # file is the DESTINATION
    r"\bdd\b[^;&|\n]*\bof=\S*" + PROT,
    r"\bopen\(\s*" + _Q + r"[^'\"]*" + PROT + r"[^'\"]*" + _Q + r"\s*,\s*" + _Q + r"[^'\"]*[wax+]",  # python open(..., 'a'|'w'|'x')
    r"\.(?:write_text|write_bytes)\(",                                                 # counts only if PROT is in the segment
)]


def protected_write(cmd: str) -> bool:
    """True if a shell segment WRITES to the ledger / split lock. Reads (cat, tail, jq, git add, cp FROM) are fine."""
    for seg in re.split(r"\|\||&&|;|\||\n", cmd):
        if re.search(PROT, seg) and any(p.search(seg) for p in _WRITE_PATTERNS):
            return True
    return False


# Recursive searches rooted at data/ could traverse data/sealed (a regex cannot see glob/find expansion) -> ask first.
RECURSIVE_ON_DATA = (
    r"\b(?:grep\s+-[a-zA-Z]*[rR][a-zA-Z]*|rg\s+[^|;&]*--no-ignore\S*|find|xargs|du|tree|ls\s+-[a-zA-Z]*R)\b[^|;&\n]*"
    r"(?<![\w./-])(?:\./)?data/?(?=\s|$|['\"])")

BASH_ASK = [(re.compile(p), why) for p, why in (
    (r"git\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s)", "this discards uncommitted work"),
    (r"\buv\s+(add|remove)\b", "dependency changes need an ADR (CLAUDE.md section 4)"),
    (r"\b(curl|wget)\b|\bhf\s+download\b|huggingface-cli\s+download|git\s+clone\b",
     "network fetches are only for `make fetch` (checksummed assets)"),
    (r"git\s+push\b", "pushing needs owner confirmation"),
    (RECURSIVE_ON_DATA, "a recursive search rooted at data/ may traverse data/sealed (INV-8); name a subdirectory"),
)]


def rel(path: str, cwd: str) -> str:
    p = os.path.normpath(path if os.path.isabs(path) else os.path.join(cwd, path))
    root = os.path.normpath(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    try:
        p = os.path.relpath(p, root)
    except ValueError:
        pass
    return p.replace(os.sep, "/")


def sealed(path: str, loose: bool = False) -> bool:
    """True if `path` names sealed TEST labels. `loose` (Glob patterns) ignores the extension requirement."""
    q = path.replace("\\", "/")
    if SEALED_DIR.search(q) or HF_TEST_LABELS.search(q):
        return True
    if loose:  # Glob patterns: only unmistakable label names (tests/**/test_label*.py is fine)
        return bool(LABEL_NAME_LOOSE.search(q))
    return bool(LABEL_NAME.search(q) and DATA_EXT.search(q))


def sealed_cmd(cmd: str) -> bool:
    if SEALED_DIR.search(cmd.replace("\\", "/")):
        return True
    return any(tok and sealed(tok) for tok in re.split(r"[\s'\"=<>|;&()]+", cmd))


def texts_written(tool: str, ti: dict) -> list[str]:
    if tool == "Write":
        return [ti.get("content") or ""]
    if tool == "Edit":
        return [ti.get("new_string") or ""]
    if tool == "MultiEdit":
        return [(e or {}).get("new_string") or "" for e in (ti.get("edits") or [])]
    return []


def main() -> None:
    try:
        d = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    tool = d.get("tool_name", "")
    ti = d.get("tool_input") or {}
    cwd = d.get("cwd") or os.getcwd()

    if tool in ("Read", "NotebookRead", "Grep", "Glob"):
        if tool in ("Read", "NotebookRead"):
            cands = [ti.get("file_path"), ti.get("notebook_path")]
        elif tool == "Grep":
            cands = [ti.get("path"), ti.get("glob")]
        else:
            cands = [ti.get("path"), ti.get("pattern")]
        loose = tool == "Glob"
        if tool == "Grep" and str(ti.get("path") or "").rstrip("/") in ("data", "./data"):
            ask("a recursive Grep rooted at data/ may traverse data/sealed (INV-8); name a subdirectory.")
        for c in cands:
            if c and sealed(rel(c, cwd) if os.path.isabs(str(c)) else str(c), loose=loose):
                deny("TEST labels are sealed (INV-8). Only `acis eval official` may load them; tune on DEV-H instead.")
        sys.exit(0)

    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        fp = ti.get("file_path") or ti.get("notebook_path")
        if not fp:
            sys.exit(0)
        r = rel(fp, cwd)
        if sealed(r):
            deny(f"{r} is sealed TEST data (INV-8).")
        if any(p.search(r) for p in NO_WRITE):
            deny(f"{r} is a protected file (CLAUDE.md section 4). Use the sanctioned CLI/command or ask the owner.")
        if r.startswith("src/") and r.endswith(".py"):
            rules = list(SRC_BANNED) + (ADAPTER_BANNED if r.endswith("mteb_adapter.py") or "/mteb_adapter/" in r else [])
            for chunk in texts_written(tool, ti):
                chunk = re.sub(r"(?m)#[^\n]*$", "", chunk)          # comments after # are not code
                for pat, why in rules:
                    if pat.search(chunk):
                        deny(f"{why} (writing {r}).")
        if any(p.search(r) for p in ASK_WRITE):
            ask(f"{r} is part of the project constitution/config; confirm this change (record decisions with /adr).")
        sys.exit(0)

    if tool == "Bash":
        cmd = ti.get("command") or ""
        if not FINAL_EVAL_OK.search(cmd) and sealed_cmd(cmd):
            deny("this command references sealed TEST labels (INV-8). Only `acis eval official` may load them.")
        if protected_write(cmd):
            deny("the ledger and split lock are append-only via the acis CLI")
        for pat, why in BASH_DENY:
            if pat.search(cmd):
                deny(why)
        for pat, why in BASH_ASK:
            if pat.search(cmd):
                ask(why)
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
