"""Evaluation-integrity guard (docs/spec/03 §5, INV-8, INV-3, INV-4).

The seal itself is physical — the held-out labels live outside the working tree. This module is the *programmatic*
half: it raises when held-out material reaches a training batch, a feature table or a tuning call, it counts DEV-H
touches, and it names the one module allowed to read the held-out labels.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from acis.appsdata.sources import DEV_SPLIT
from acis.core.errors import SealedDataAccess
from acis.core.paths import acis_home

#: The only module that may read held-out labels, and only from `acis eval official` (docs/spec/03 §5).
SANCTIONED_READER = "acis.eval.final"
DEV_H_COUNTER = "dev_h_touches.json"


def assert_dev_split(split: str, *, context: str = "") -> str:
    if split != DEV_SPLIT:
        raise SealedDataAccess(
            f"split {split!r} may not be used here; the dev environment loads {DEV_SPLIT!r} only (INV-8)",
            context=context,
        )
    return split


def caller_is_sanctioned(depth: int = 12) -> bool:
    """True when `acis.eval.final` is somewhere on the call stack.

    This is a sanity check, not a boundary: any code that enters that module satisfies it. The boundary is
    physical — the labels live outside the working tree and only a sealed `HF_HOME` can reach them (D19).
    """
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            frame = frame.f_back if frame else None
            if frame is None:
                return False
            if frame.f_globals.get("__name__", "") == SANCTIONED_READER:
                return True
    finally:
        del frame
    return False


def assert_sanctioned_reader(what: str = "held-out labels") -> None:
    if not caller_is_sanctioned():
        raise SealedDataAccess(f"{what} may only be read by {SANCTIONED_READER} during `acis eval official` (INV-8)")


def holdout_query_ids() -> frozenset[str]:
    """Ids of queries with no dev label. Ids only; no label is read (INV-8)."""
    from acis.eval.splits import holdout_query_ids as _ids  # noqa: PLC0415 — avoids an import cycle

    return frozenset(_ids())


def assert_no_holdout_ids(ids: Iterable[str], *, context: str) -> None:
    """Raise if any held-out query id reaches a training batch, feature table or tuning call (docs/spec/03 §5)."""
    holdout = holdout_query_ids()
    offenders = sorted({str(i) for i in ids} & holdout)
    if offenders:
        raise SealedDataAccess(
            "held-out query ids reached a fitting path",
            context=context,
            n_offenders=len(offenders),
            example=offenders[0],
        )


def assert_dev_pool(ids: Iterable[str], *, context: str) -> list[str]:
    """Every id must belong to the dev pool. Returns the checked ids (sorted)."""
    from acis.appsdata import apps  # noqa: PLC0415

    dev = set(apps.dev_query_ids())
    listed = sorted({str(i) for i in ids})
    unknown = [i for i in listed if i not in dev]
    if unknown:
        raise SealedDataAccess(
            "ids outside the dev pool reached a fitting path",
            context=context,
            n_unknown=len(unknown),
            example=unknown[0],
        )
    return listed


def assert_ids_not_features(feature_names: Iterable[str]) -> None:
    """INV-4: no feature may be derived from a document or query id — or from its position in the corpus.

    Position matters as much as identity here: on the AppsRetrieval corpus the ordinals 0–4,999 are exactly the
    training-partition documents and 5,000–8,764 exactly the held-out ones, so `ordinal < 5000` *is* the
    train-document detector docs/DESIGN_RULES.md forbids, with no error at all.
    """
    # Substring matching would reject legitimate Phase 4 features: `first_match_position` and
    # `term_position_variance` are *within-document* positions and have nothing to do with the corpus ordinal.
    # Identity names are matched as whole names; the positional hazards are matched as exact feature names.
    banned_names = frozenset(
        {
            "query_id",
            "qid",
            "doc_id",
            "docid",
            "corpus_id",
            "corpusid",
            "external_id",
            "corpus_ordinal",
            "corpus_order",
            "corpus_position",
            "corpus_index",
            "ordinal",
            "partition",
            "row_index",
        }
    )
    banned_tokens = frozenset({"qid", "docid"})
    hits = sorted(
        {n for n in feature_names if str(n).lower() in banned_names or (set(str(n).lower().split("_")) & banned_tokens)}
    )
    if hits:
        raise SealedDataAccess("external ids and corpus position may never be features (INV-4)", features=hits)


# -- DEV-H touch counter -------------------------------------------------------------------------------------------
def _counter_path() -> Path:
    return acis_home() / DEV_H_COUNTER


def dev_h_touches() -> list[Mapping[str, object]]:
    path = _counter_path()
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def record_dev_h_touch(reason: str, milestone: str) -> int:
    """DEV-H is a once-per-milestone confirmation, not a decision set. Every use is counted (docs/spec/03 §5)."""
    entries = list(dev_h_touches())
    entries.append({"ts": time.time(), "reason": reason, "milestone": milestone})
    path = _counter_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return len(entries)


def dev_h_touches_for(milestone: str) -> int:
    return sum(1 for e in dev_h_touches() if e.get("milestone") == milestone)


def official_run_env_ok() -> tuple[bool, str]:
    """The official run is the only process pointed at the sealed HF cache (docs/spec/09 §3).

    Compared as a path under `sealed_root()`, not as a substring of the directory name. The substring test refused
    a correctly configured owner who had moved the seal with `ACIS_SEALED_HOME`, and accepted any decoy directory
    whose name happened to contain those characters.
    """
    from acis.core.paths import sealed_root  # noqa: PLC0415

    home = os.environ.get("HF_HOME", "")
    if not home:
        return False, ""
    try:
        return Path(home).resolve().is_relative_to(sealed_root().resolve()), home
    except (OSError, ValueError):
        return False, home


__all__ = [
    "DEV_H_COUNTER",
    "SANCTIONED_READER",
    "assert_dev_pool",
    "assert_dev_split",
    "assert_ids_not_features",
    "assert_no_holdout_ids",
    "assert_sanctioned_reader",
    "caller_is_sanctioned",
    "dev_h_touches",
    "dev_h_touches_for",
    "holdout_query_ids",
    "official_run_env_ok",
    "record_dev_h_touch",
]
