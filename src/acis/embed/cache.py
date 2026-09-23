"""Content-addressed vector cache (docs/spec/02 §3, D10, INV-2).

A cold official pass embeds ~3M tokens. Re-embedding a document whose bytes have not changed is the single
largest avoidable cost in the project, and P1 depends on the same property: a version bump re-embeds only the
units that actually changed.

The cache key is the whole identity of the computation, not just the text:

    sha256(model_fingerprint ‖ numeric_profile ‖ prep_hash ‖ embed_key)

and a query vector adds the prompt hash, because the same text under two instructions is two different vectors.
Leaving any component out is how a cache silently serves a vector from a different model, a different precision or
a different preprocessing — which would not crash, would not be noticed, and would invalidate every number taken
afterwards.

Storage is plain `.npy` per vector under a fanned-out directory, memory-mapped on read (D10: files plus mmap, no
vector database). Writes are atomic: a crashed run leaves no half-written vector for the next run to trust.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acis.core.hashing import hash_obj, sha256_text
from acis.core.paths import acis_home

CACHE_DIRNAME = "vectors"
FANOUT = 2


def embed_key(text: str) -> str:
    """Identity of the *text* being embedded. Content-addressed, so identical documents share one vector."""
    return sha256_text(text)


def vector_key(
    *,
    model_fingerprint: str,
    numeric_profile: str,
    prep_hash: str,
    text: str,
    prompt_hash: str = "",
) -> str:
    """The full cache key. Every component that can change the vector is in it (docs/spec/02 §3)."""
    return hash_obj(
        {
            "model": model_fingerprint,
            "profile": numeric_profile,
            "prep": prep_hash,
            "embed": embed_key(text),
            "prompt": prompt_hash,
        }
    )


@dataclass(slots=True)
class VectorCache:
    """A content-addressed store of embedding vectors."""

    root: Path
    dim: int | None = None
    hits: int = 0
    misses: int = 0
    writes: int = 0
    _mem: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, root: str | Path | None = None, *, dim: int | None = None) -> VectorCache:
        path = Path(root) if root is not None else acis_home() / CACHE_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return cls(root=path, dim=dim)

    def path_for(self, key: str) -> Path:
        """Fanned out by the first bytes of the key: one flat directory with 8,765+ entries is slow to list."""
        return self.root / key[:FANOUT] / key[FANOUT : FANOUT * 2] / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        cached = self._mem.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            vector = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            # A truncated or unreadable vector is a miss, never an exception at the caller: the worst case is
            # that we recompute it.
            self.misses += 1
            return None
        array = np.asarray(vector, dtype=np.float32)
        self._mem[key] = array
        self.hits += 1
        return array

    def put(self, key: str, vector: np.ndarray) -> None:
        """Write atomically: a crashed run must not leave a half-written vector for the next one to trust."""
        array = np.asarray(vector, dtype=np.float32)
        if self.dim is not None and array.shape[-1] != self.dim:
            raise ValueError(f"vector has dimension {array.shape[-1]}, cache holds {self.dim}")
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, array, allow_pickle=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._mem[key] = array
        self.writes += 1

    def get_many(self, keys: Sequence[str]) -> tuple[dict[int, np.ndarray], list[int]]:
        """`({index: vector}, [missing indices])` — the shape an encode pass wants."""
        found: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            vector = self.get(key)
            if vector is None:
                missing.append(i)
            else:
                found[i] = vector
        return found, missing

    def put_many(self, pairs: Iterable[tuple[str, np.ndarray]]) -> int:
        count = 0
        for key, vector in pairs:
            self.put(key, vector)
            count += 1
        return count

    @property
    def stats(self) -> Mapping[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hits / total, 6) if total else 0.0,
        }

    def reset_stats(self) -> None:
        self.hits = self.misses = self.writes = 0


def audit_sample(
    cache: VectorCache,
    keys: Sequence[str],
    recompute: "callable[[str], np.ndarray]",  # noqa: UP037 — kept simple for the doc
    *,
    fraction: float = 0.01,
    min_cosine: float = 0.9999,
    seed: int = 0,
) -> dict[str, float | int | list[str]]:
    """Recompute a sample of cached vectors and compare (docs/spec/02 §3: 1 %, cosine ≥ 0.9999).

    A cache that serves subtly wrong vectors produces a plausible ranking and a wrong number, so the only defence
    is to spot-check it against the thing it is caching.
    """
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    sample = [k for k in keys if rng.random() < fraction] or list(keys[:1])
    failures: list[str] = []
    worst = 1.0
    for key in sample:
        cached = cache.get(key)
        if cached is None:
            failures.append(f"{key}: absent")
            continue
        fresh = np.asarray(recompute(key), dtype=np.float32)
        denominator = float(np.linalg.norm(cached) * np.linalg.norm(fresh)) or 1.0
        cosine = float(np.dot(cached.ravel(), fresh.ravel()) / denominator)
        worst = min(worst, cosine)
        if cosine < min_cosine:
            failures.append(f"{key}: cosine {cosine:.6f}")
    return {"sampled": len(sample), "worst_cosine": worst, "failures": failures, "min_cosine": min_cosine}


__all__ = ["CACHE_DIRNAME", "FANOUT", "VectorCache", "audit_sample", "embed_key", "vector_key"]
