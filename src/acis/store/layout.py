"""The on-disk layout of the store (docs/spec/04 §1, D10).

Every path the store ever touches is computed here, and every one of them is built from a *validated* component.
Repository ids and version labels arrive from request bodies, JSONL records and archive members, so "the store's
layout" is also part of the attack surface: one unvalidated `..` and a snapshot directory lands outside
`ACIS_HOME`.

The layout itself is the P1 design in file form: an append-only CAS shared by every snapshot, immutable snapshot
directories that are only ever written under a temporary name, and a `refs/` directory of atomic pointers — which
is what makes activation and rollback a rename rather than a rebuild.
"""

from __future__ import annotations

import re
from pathlib import Path

from acis.core.errors import InvalidInput
from acis.core.paths import acis_home
from acis.sec.paths import safe_component

HASH = re.compile(r"^[0-9a-f]{64}$")
#: Repository and version identifiers: conservative on purpose — these become directory names.
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

CATALOG = "catalog.sqlite"
VALID_MARKER = "VALID"
MANIFEST = "manifest.json"
UNITS_DB = "units.sqlite"
LEXICAL_DIR = "lexical"
ACTIVE_REF = "ACTIVE"
PREV_REF = "PREV"


def safe_name(name: str, *, what: str = "identifier") -> str:
    """Validate an identifier that is about to become a directory name."""
    if not isinstance(name, str) or not NAME.match(name):
        raise InvalidInput(
            f"{what} must be 1-64 characters of [A-Za-z0-9._-] and start alphanumerically",
            value=(name[:64] if isinstance(name, str) else type(name).__name__),
        )
    return safe_component(name)


def store_root() -> Path:
    return acis_home()


def catalog_path() -> Path:
    return store_root() / CATALOG


def cas_root() -> Path:
    return store_root() / "cas"


def blob_path(digest: str) -> Path:
    """`cas/blobs/ab/cd…` — sharded so no directory holds a million entries."""
    if not isinstance(digest, str) or not HASH.match(digest):
        raise InvalidInput("a blob key is a lowercase sha256 hex digest", value=str(digest)[:80])
    return cas_root() / "blobs" / digest[:2] / digest[2:]


def segments_root(model_fingerprint: str, profile: str) -> Path:
    return cas_root() / "segments" / safe_name(model_fingerprint[:32], what="model fingerprint") / safe_name(profile)


def repo_root(repo_id: str) -> Path:
    return store_root() / "repos" / safe_name(repo_id, what="repo id")


def snapshots_root(repo_id: str) -> Path:
    return repo_root(repo_id) / "snapshots"


def snapshot_dir(repo_id: str, snapshot_id: str) -> Path:
    return snapshots_root(repo_id) / safe_name(snapshot_id, what="snapshot id")


def tmp_dir(repo_id: str, token: str) -> Path:
    """An in-flight build. The `.tmp-` prefix is what crash recovery deletes on startup (spec 04 §3)."""
    return repo_root(repo_id) / f".tmp-{safe_name(token, what='build token')}"


def refs_dir(repo_id: str) -> Path:
    return repo_root(repo_id) / "refs"


def ref_path(repo_id: str, name: str) -> Path:
    return refs_dir(repo_id) / safe_name(name, what="ref name")


__all__ = [
    "ACTIVE_REF",
    "CATALOG",
    "HASH",
    "LEXICAL_DIR",
    "MANIFEST",
    "NAME",
    "PREV_REF",
    "UNITS_DB",
    "VALID_MARKER",
    "blob_path",
    "cas_root",
    "catalog_path",
    "ref_path",
    "refs_dir",
    "repo_root",
    "safe_name",
    "segments_root",
    "snapshot_dir",
    "snapshots_root",
    "store_root",
    "tmp_dir",
]
