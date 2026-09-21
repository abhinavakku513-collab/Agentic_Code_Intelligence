"""Deterministic hashing and stable serialisation.

Every identity in ACIS (config hash, split lock, ledger chain, CAS key, fold assignment) goes through this module so
that the same inputs always produce the same digest on any machine and any Python build (`PYTHONHASHSEED` independent).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HASH_NAME = "sha256"
_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hash of the UTF-8 encoding of `text` (no normalisation — callers normalise explicitly when they mean to)."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace jitter, non-ASCII preserved. The input of every structural hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"{type(obj).__name__} is not stably serialisable")


def hash_obj(obj: Any) -> str:
    """Structural hash of any JSON-able object."""
    return sha256_text(stable_json(obj))


def short(digest: str, n: int = 12) -> str:
    return digest[:n]


def hash_id_list(ids: Iterable[str]) -> str:
    """SHA-256 of the **sorted** id list — the primitive behind `configs/splits.lock.json` (docs/spec/03 §5)."""
    return hash_obj(sorted(str(i) for i in ids))


def normalized_text_key(text: str) -> str:
    """Whitespace/unicode-insensitive key used for duplicate detection and decontamination candidates."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return sha256_text(" ".join(folded.split()))


def fold_of(text: str, folds: int = 5) -> int:
    """Deterministic fold for a query: `sha256(query_text) mod folds` (docs/spec/03 §5). Never uses the query id."""
    if folds <= 0:
        raise ValueError("folds must be positive")
    return int(sha256_text(text), 16) % folds


def merkle_of(mapping: Mapping[str, str]) -> str:
    """Order-independent digest of a {name: digest} mapping (manifests, snapshot contents)."""
    return hash_obj({str(k): str(v) for k, v in mapping.items()})


__all__ = [
    "HASH_NAME",
    "fold_of",
    "hash_id_list",
    "hash_obj",
    "merkle_of",
    "normalized_text_key",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "short",
    "stable_json",
]
