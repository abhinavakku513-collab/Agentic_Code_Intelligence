"""`acis.core` — types, configuration identity, hashing, paths, numeric profiles, typed errors.

Nothing in this package imports torch, mteb or any other heavy dependency: the CLI, the hooks and the tests must be
able to use it instantly and offline.
"""

from __future__ import annotations

from acis.core.config import FrozenConfig, freeze_config, load_frozen_config
from acis.core.errors import (
    AcisError,
    IndexRequired,
    InvalidInput,
    NotFound,
    NotReady,
    ResourceLimit,
    SealedDataAccess,
    SnapshotInvalid,
    StrictViolation,
    VersionConflict,
)
from acis.core.hashing import fold_of, hash_id_list, hash_obj, sha256_text, short
from acis.core.numeric import REFERENCE_PROFILE, get_profile, resolve_threads
from acis.core.paths import acis_home, acis_root, repo_path
from acis.core.types import Hit, SearchRequest, SearchResponse, Snapshot, Snippet, Unit

__all__ = [
    "REFERENCE_PROFILE",
    "AcisError",
    "FrozenConfig",
    "Hit",
    "IndexRequired",
    "InvalidInput",
    "NotFound",
    "NotReady",
    "ResourceLimit",
    "SealedDataAccess",
    "SearchRequest",
    "SearchResponse",
    "Snapshot",
    "SnapshotInvalid",
    "Snippet",
    "StrictViolation",
    "Unit",
    "VersionConflict",
    "acis_home",
    "acis_root",
    "fold_of",
    "freeze_config",
    "get_profile",
    "hash_id_list",
    "hash_obj",
    "load_frozen_config",
    "repo_path",
    "resolve_threads",
    "sha256_text",
    "short",
]
