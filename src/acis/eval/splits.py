"""Split construction and the split lock (docs/spec/03 §5, D19).

Folds come from `sha256(query_text) mod k` — the query **text**, never its id (INV-4) — so the same query always lands
in the same fold on any machine, and relabelling ids changes nothing.

`configs/splits.lock.json` records the SHA-256 of the sorted id list of every set. Its `split_lock_hash` goes into
every ledger row: if a set silently changes, every later number is provably about different data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acis.appsdata import apps
from acis.appsdata.sources import APPS
from acis.core.errors import InvalidInput
from acis.core.hashing import hash_id_list, hash_obj
from acis.core.paths import repo_path

LOCK_PATH = ("configs", "splits.lock.json")
DEFAULT_FOLDS = 5
DEV_H_FOLD = 4
LOCK_VERSION = 1


@dataclass(frozen=True, slots=True)
class SplitLock:
    version: int
    dataset: Mapping[str, str]
    fold_key: str
    folds: int
    dev_h_fold: int
    sets: Mapping[str, Mapping[str, Any]]
    assets: Mapping[str, str]
    split_lock_hash: str
    built_ts: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "dataset": dict(self.dataset),
                "fold_key": self.fold_key,
                "folds": self.folds,
                "dev_h_fold": self.dev_h_fold,
                "sets": {k: dict(v) for k, v in self.sets.items()},
                "assets": dict(self.assets),
                "split_lock_hash": self.split_lock_hash,
                "built_ts": self.built_ts,
            },
            indent=2,
            sort_keys=True,
        )


def lock_path() -> Path:
    return repo_path(*LOCK_PATH)


def fold_members(folds: int = DEFAULT_FOLDS) -> dict[int, list[str]]:
    """`{fold: [query_id, …]}` over the dev split, ids sorted for stability."""
    assignment = apps.fold_assignments(folds=folds)
    out: dict[int, list[str]] = {f: [] for f in range(folds)}
    for qid, fold in assignment.items():
        out[fold].append(qid)
    return {f: sorted(ids) for f, ids in out.items()}


def dev_h_ids(folds: int = DEFAULT_FOLDS, dev_h_fold: int = DEV_H_FOLD) -> list[str]:
    """DEV-H (F4): a once-per-milestone confirmation set, never the decision set (docs/spec/03 §5)."""
    return fold_members(folds)[dev_h_fold]


def oof_pool_ids(folds: int = DEFAULT_FOLDS) -> list[str]:
    """The decision set: **all** dev queries (docs/spec/09 §2)."""
    return sorted(apps.dev_query_ids())


def holdout_query_ids() -> list[str]:
    """Query ids that carry no dev label. Ids only — no label of theirs is ever read here (INV-8)."""
    dev = set(apps.dev_query_ids())
    return sorted(qid for qid in apps.all_query_ids() if qid not in dev)


def _entry(ids: Sequence[str]) -> dict[str, Any]:
    return {"n": len(ids), "sha256": hash_id_list(ids)}


def build_lock(folds: int = DEFAULT_FOLDS, dev_h_fold: int = DEV_H_FOLD) -> SplitLock:
    from acis.appsdata.fetch import load_manifest  # noqa: PLC0415 — only needed when a lock is built

    if folds < 2:
        raise InvalidInput("at least two folds are needed for cross-fitting", folds=folds)
    if not 0 <= dev_h_fold < folds:
        raise InvalidInput("dev_h_fold must name an existing fold", folds=folds, dev_h_fold=dev_h_fold)
    members = fold_members(folds)
    sets: dict[str, dict[str, Any]] = {
        "corpus": _entry(list(apps.corpus_ids())),
        "dev_queries": _entry(list(apps.dev_query_ids())),
        "oof_pool": _entry(oof_pool_ids(folds)),
        "dev_h": _entry(members[dev_h_fold]),
        "holdout_queries": _entry(holdout_query_ids()),
    }
    for fold, ids in members.items():
        sets[f"fold_{fold}"] = _entry(ids)
    assets = {a["repo_file"]: a["sha256"] for a in load_manifest().get("assets", [])}
    return SplitLock(
        version=LOCK_VERSION,
        dataset={"repo": APPS.repo, "revision": APPS.revision, "task": APPS.task},
        fold_key="sha256_query_text",
        folds=folds,
        dev_h_fold=dev_h_fold,
        sets=sets,
        assets=assets,
        split_lock_hash=hash_obj({"sets": sets, "folds": folds, "fold_key": "sha256_query_text"}),
        built_ts=time.time(),
    )


def write_lock(lock: SplitLock | None = None) -> Path:
    """Write `configs/splits.lock.json`. Refuses to overwrite a lock whose hash differs — that needs an ADR."""
    lock = lock if lock is not None else build_lock()
    path = lock_path()
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("split_lock_hash") != lock.split_lock_hash:
            raise InvalidInput(
                "the split lock already exists with a different hash; changing the splits invalidates every recorded "
                "number and needs an ADR",
                existing=existing.get("split_lock_hash", "")[:12],
                new=lock.split_lock_hash[:12],
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lock.to_json(), encoding="utf-8")
    return path


def load_lock() -> dict[str, Any]:
    path = lock_path()
    if not path.is_file():
        raise InvalidInput(f"split lock not found: run `uv run acis eval splits --write` ({path})")
    locked: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return locked


def verify_lock() -> list[str]:
    """Rebuild every set from the data and compare against the lock. Returns the list of problems (empty == clean)."""
    locked = load_lock()
    current = build_lock(
        folds=int(locked.get("folds", DEFAULT_FOLDS)), dev_h_fold=int(locked.get("dev_h_fold", DEV_H_FOLD))
    )
    problems: list[str] = []
    for name, entry in locked.get("sets", {}).items():
        now = current.sets.get(name)
        if now is None:
            problems.append(f"set disappeared: {name}")
        elif now["sha256"] != entry["sha256"]:
            problems.append(f"set changed: {name} (n {entry['n']} -> {now['n']})")
    for repo_file, digest in locked.get("assets", {}).items():
        if current.assets.get(repo_file) != digest:
            problems.append(f"asset changed: {repo_file}")
    if locked.get("split_lock_hash") != current.split_lock_hash:
        problems.append("split_lock_hash mismatch")
    return problems


def split_lock_hash() -> str:
    return str(load_lock().get("split_lock_hash", ""))


__all__ = [
    "DEFAULT_FOLDS",
    "DEV_H_FOLD",
    "LOCK_VERSION",
    "SplitLock",
    "build_lock",
    "dev_h_ids",
    "fold_members",
    "holdout_query_ids",
    "load_lock",
    "lock_path",
    "oof_pool_ids",
    "split_lock_hash",
    "verify_lock",
    "write_lock",
]
