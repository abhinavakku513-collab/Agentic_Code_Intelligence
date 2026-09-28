"""The experiment ledger (docs/spec/03 §6, INV-14): append-only, hash-chained JSONL.

Rules this module enforces, because every number in every document points back here:

* rows are appended, never rewritten (the file is opened `"a"`, one `fsync`'d line at a time);
* each row carries `prev_hash`, so a deleted or edited row breaks the chain and `verify_chain()` says where;
* `run_id` is derived from the row's own content, so the same experiment cannot be recorded twice under two names;
* a row that touches the held-out split must declare it, and the running total is checked against the budget.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acis.core.errors import InvalidInput
from acis.core.hashing import hash_obj, short
from acis.core.paths import repo_path

LEDGER_PATH = ("runs", "ledger.jsonl")
GENESIS = "0" * 64
TEST_TOUCH_BUDGET = 6  # RC0 x1, RC1 x2 (A+B), contingency x2, post-hoc clean-pool x1 (CLAUDE.md §4)
KINDS = ("dev", "gate", "rc", "audit", "bench")


def ledger_path() -> Path:
    return repo_path(*LEDGER_PATH)


@dataclass(frozen=True, slots=True)
class LedgerRow:
    payload: Mapping[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def row_hash(self) -> str:
        return str(self.payload["row_hash"])

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_path(), capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


#: The ledger is the evidence a commit produced, not code: appending one row must not mark the next as dirty.
DIRTY_EXEMPT = ("runs/ledger.jsonl",)


def changed_paths(porcelain: str) -> list[str]:
    """Paths from `git status --porcelain`, robust to the caller having stripped the first line's status column."""
    paths = []
    for line in porcelain.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            paths.append(parts[1].strip())
    return paths


def tree_is_dirty(porcelain: str) -> bool:
    return any(path not in DIRTY_EXEMPT for path in changed_paths(porcelain))


def environment() -> dict[str, Any]:
    """The part of a row that describes *where* a number came from."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("mteb", "torch", "transformers", "numpy", "lightgbm", "bm25s", "pytrec-eval-terrier", "datasets"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "absent"
    return {
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "dirty": tree_is_dirty(_git("status", "--porcelain")),
        "versions": versions,
        "hostname_hash": short(hash_obj(platform.node()), 12),
        "thread_env": {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


@dataclass
class LedgerRowBuilder:
    """Collects the fields of one row. Unknown extras are allowed; the required ones are checked on `append`."""

    kind: str
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def with_metrics(self, metrics: Mapping[str, float]) -> LedgerRowBuilder:
        self.metrics.update({str(k): float(v) for k, v in metrics.items()})
        return self

    def with_fields(self, **fields: Any) -> LedgerRowBuilder:
        self.extra.update(fields)
        return self

    def build(self) -> dict[str, Any]:
        if self.kind not in KINDS:
            raise InvalidInput(f"unknown ledger kind {self.kind!r}; known: {KINDS}")
        row: dict[str, Any] = {
            "ts": time.time(),
            "kind": self.kind,
            "metrics": dict(sorted(self.metrics.items())),
            "environment": environment(),
            "test_touch_count": 0,
            **self.extra,
        }
        return row


def read_rows() -> list[LedgerRow]:
    path = ledger_path()
    if not path.is_file():
        return []
    rows: list[LedgerRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(LedgerRow(json.loads(line)))
    return rows


def iter_rows() -> Iterator[LedgerRow]:
    yield from read_rows()


def last_hash() -> str:
    rows = read_rows()
    return rows[-1].row_hash if rows else GENESIS


def test_touches_used() -> int:
    return sum(int(r.get("test_touch_count", 0)) for r in read_rows())


def append(row: Mapping[str, Any]) -> LedgerRow:
    """Append one row, chaining it to the previous. The only writer of `runs/ledger.jsonl`."""
    problems = verify_chain()
    if problems:
        raise InvalidInput(
            "refusing to extend a ledger whose chain is already broken; investigate before recording anything else",
            problems=problems[:3],
        )
    payload = dict(row)
    payload.setdefault("ts", time.time())
    payload.setdefault("kind", "dev")
    payload.setdefault("test_touch_count", 0)

    touches = int(payload["test_touch_count"])
    if touches < 0:
        raise InvalidInput("test_touch_count cannot be negative")
    if test_touches_used() + touches > TEST_TOUCH_BUDGET:
        raise InvalidInput(
            "this row would exceed the held-out touch budget",
            used=test_touches_used(),
            requested=touches,
            budget=TEST_TOUCH_BUDGET,
        )

    payload["prev_hash"] = last_hash()
    payload["run_id"] = payload.get("run_id") or _derive_run_id(payload)
    payload["row_hash"] = hash_obj({k: v for k, v in payload.items() if k != "row_hash"})

    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return LedgerRow(payload)


def _derive_run_id(payload: Mapping[str, Any]) -> str:
    seed = {k: v for k, v in payload.items() if k not in ("ts", "prev_hash", "run_id", "row_hash")}
    return f"{payload.get('kind', 'dev')}-{short(hash_obj(seed), 12)}"


def verify_chain() -> list[str]:
    """Re-derive every hash. Returns the list of problems (empty == intact)."""
    problems: list[str] = []
    prev = GENESIS
    for i, row in enumerate(read_rows()):
        payload = dict(row.payload)
        recorded = payload.pop("row_hash", None)
        if payload.get("prev_hash") != prev:
            problems.append(f"row {i} ({row.get('run_id')}): prev_hash does not match the previous row")
        if recorded is None:
            problems.append(f"row {i} ({row.get('run_id')}): row_hash is missing — the row is truncated")
            break
        if hash_obj(payload) != recorded:
            problems.append(f"row {i} ({row.get('run_id')}): row_hash does not match its content")
        prev = str(recorded)
    return problems


def find(run_id: str) -> LedgerRow:
    for row in read_rows():
        if row.run_id == run_id:
            return row
    raise InvalidInput(f"no ledger row with run_id {run_id!r}")


def latest(kind: str | None = None) -> LedgerRow | None:
    rows = [r for r in read_rows() if kind is None or r.get("kind") == kind]
    return rows[-1] if rows else None


__all__ = [
    "GENESIS",
    "KINDS",
    "TEST_TOUCH_BUDGET",
    "LedgerRow",
    "LedgerRowBuilder",
    "append",
    "environment",
    "find",
    "iter_rows",
    "last_hash",
    "latest",
    "ledger_path",
    "read_rows",
    "test_touches_used",
    "verify_chain",
]
