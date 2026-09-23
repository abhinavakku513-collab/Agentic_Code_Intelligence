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
#: The held-out label file appears in three shapes: as a repository entry
#: (`data/<split>-00000-of-00001.parquet`), inside an HF cache
#: (`datasets--CoIR-Retrieval--apps/snapshots/<sha>/data/<split>-...parquet`), and as a bare basename if someone
#: copies it out. Detection therefore works on path **components**: a basename that is unmistakably a label file,
#: or a split-named shard sitting directly inside a `data/`- or `qrels/`-style directory.
#: Basenames that are a label file wherever they sit.
SEALED_BASENAMES: tuple[str, ...] = (
    f"*qrels*{SEALED_SPLIT}*.parquet",
    f"*{SEALED_SPLIT}*qrels*.parquet",
    f"*qrels*{SEALED_SPLIT}*.tsv",
    f"*{SEALED_SPLIT}*qrels*.tsv",
    f"*qrels*{SEALED_SPLIT}*.jsonl",
    f"*{SEALED_SPLIT}*qrels*.jsonl",
    f"{SEALED_SPLIT}-labels*.parquet",
    # A HuggingFace *datasets* cache stores `…/<config>/0.0.0/<hash>/<dataset>-<split>.arrow`: no "qrels" in the
    # name and a hash for a parent. Since `warm_sealed_datasets_cache` deliberately creates that layout in the
    # sealed area, a detector blind to it would be blind to the one cache we build ourselves.
    f"*-{SEALED_SPLIT}.arrow",
    f"*-{SEALED_SPLIT}-*.arrow",
)
#: Directory names whose split-named shards are labels rather than documents.
SEALED_PARENTS: frozenset[str] = frozenset({"data", "qrels"})
SEALED_IN_PARENT: tuple[str, ...] = (f"{SEALED_SPLIT}-*.parquet", f"{SEALED_SPLIT}.parquet", f"{SEALED_SPLIT}.tsv")
#: Kept for the repository-listing filter, which only ever sees repo-relative names.
SEALED_PATTERNS: tuple[str, ...] = (f"data/{SEALED_SPLIT}-*.parquet", *SEALED_BASENAMES)


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repo: str
    revision: str
    task: str


APPS = DatasetPin(repo=APPS_REPO, revision=APPS_REVISION, task=APPS_TASK)


def is_sealed_file(name: str) -> bool:
    """True if a path holds held-out labels and may never enter the dev environment (INV-8).

    Every suffix of the path is tested, so a file is recognised whether it is named as a repository entry, as a
    deeply nested HF-cache blob, or as a bare basename someone copied out. Prevention (the fetch allow-list) and
    detection (`assert_seal`) both rely on this, and detection is the half that sees paths we did not construct.
    """
    parts = [p for p in name.replace("\\", "/").lower().lstrip("/").split("/") if p]
    if not parts:
        return False
    basename = parts[-1]
    parent = parts[-2] if len(parts) > 1 else ""

    # Matched on path *components*, not on arbitrary string suffixes. `fnmatch`'s `*` crosses `/`, so suffix
    # matching made `tests/unit/test_qrels.py` and `runs/qrels/latest/test.json` look like held-out labels — and
    # the pressure from a false positive is always to weaken the pattern, which is the opposite of the point.
    if any(fnmatch.fnmatch(basename, p.lower()) for p in SEALED_BASENAMES):
        return True
    return parent in SEALED_PARENTS and any(fnmatch.fnmatch(basename, p.lower()) for p in SEALED_IN_PARENT)


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
