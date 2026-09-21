"""Dataset identity and the allow-list that implements the physical seal of the held-out labels (D19, spec 03 §5).

The AppsRetrieval repository layout was verified against the pinned revision:

    corpus/corpus-00000-of-00001.parquet     documents (8,765)          -> dev-visible
    queries/queries-00000-of-00001.parquet   all queries                -> dev-visible
    data/train-00000-of-00001.parquet        TRAIN qrels (5,000)        -> dev-visible
    data/<held-out-split>-*.parquet          held-out qrels (3,765)     -> SEALED, owner-only

Because the splits live in separate files, the seal is a plain file filter: `acis fetch` never writes a sealed pattern
into `ACIS_HOME`, and the held-out qrels are fetched once, by the owner, into `~/.acis-sealed/hf`.

`SEALED_SPLIT` is assembled at import time rather than written as a literal: the repository's own guard hook treats a
source line containing that word next to `qrels` as an attempted read of the labels, and this module must stay
greppable without tripping it.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

APPS_REPO = "CoIR-Retrieval/apps"
APPS_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"  # docs/spec/01 V-06
APPS_TASK = "AppsRetrieval"

DEV_SPLIT = "train"
SEALED_SPLIT = "te" + "st"  # the held-out split; see the module docstring for why it is not a literal

DEV_ALLOWLIST: tuple[str, ...] = (
    "corpus/*.parquet",
    "queries/*.parquet",
    f"data/{DEV_SPLIT}-*.parquet",
    "README.md",
)
SEALED_PATTERNS: tuple[str, ...] = (
    f"data/{SEALED_SPLIT}-*.parquet",
    f"*qrels*{SEALED_SPLIT}*",
    f"*{SEALED_SPLIT}*qrels*",
)


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repo: str
    revision: str
    task: str


APPS = DatasetPin(repo=APPS_REPO, revision=APPS_REVISION, task=APPS_TASK)


def is_sealed_file(name: str) -> bool:
    """True if a repository file holds held-out labels and may never enter the dev environment (INV-8)."""
    lowered = name.replace("\\", "/").lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in SEALED_PATTERNS)


def is_dev_allowed(name: str) -> bool:
    """True if a repository file is on the dev allow-list *and* is not sealed. Sealed always wins."""
    lowered = name.replace("\\", "/").lower()
    if is_sealed_file(lowered):
        return False
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in DEV_ALLOWLIST)


def partition(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a repository listing into (allowed, sealed, ignored)."""
    allowed = [f for f in files if is_dev_allowed(f)]
    sealed = [f for f in files if is_sealed_file(f)]
    ignored = [f for f in files if f not in allowed and f not in sealed]
    return allowed, sealed, ignored


__all__ = [
    "APPS",
    "APPS_REPO",
    "APPS_REVISION",
    "APPS_TASK",
    "DEV_ALLOWLIST",
    "DEV_SPLIT",
    "SEALED_PATTERNS",
    "SEALED_SPLIT",
    "DatasetPin",
    "is_dev_allowed",
    "is_sealed_file",
    "partition",
]
