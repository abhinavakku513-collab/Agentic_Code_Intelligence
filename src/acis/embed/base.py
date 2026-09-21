"""The dense-channel contract (docs/spec/02 §3).

Phase 1 freezes the *interface* an encoder must satisfy; Phase 2 builds the real runtime (token-budget batching,
content-addressed vector cache, numeric profiles) behind it. Everything above this line — engine, adapter, dev task —
is written against `Encoder` and never against a specific model.

`submission_capable` is the honest switch: a stand-in encoder that exists to validate the harness must say so, and
the engine refuses to serve it in strict mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """A dense encoder. `encode` returns L2-normalised float32 vectors, one row per input, in input order."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str:
        """Hash of weights + tokenizer + adapter; part of every vector cache key (docs/spec/02 §3)."""
        ...

    @property
    def submission_capable(self) -> bool:
        """False for harness stand-ins. Strict runs refuse them."""
        ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False, batch_size: int = 64) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. A zero row stays zero rather than becoming NaN."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def exact_search(query_vectors: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Exact cosine similarity by one matmul (D5). Rows are queries, columns documents.

    No ANN, no vector database: at fewer than 250k vectors an exact matmul is microseconds and has no recall loss,
    and a batch-invariant matmul keeps INV-3 trivially true.
    """
    if query_vectors.ndim != 2 or doc_matrix.ndim != 2:
        raise ValueError("expected 2-D matrices")
    if query_vectors.shape[1] != doc_matrix.shape[1]:
        raise ValueError(f"dimension mismatch: {query_vectors.shape[1]} vs {doc_matrix.shape[1]}")
    return query_vectors.astype(np.float32, copy=False) @ doc_matrix.astype(np.float32, copy=False).T


def top_k_from_scores(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-`k` `(index, score)` for one query row, ties broken by ascending index (deterministic)."""
    k = max(1, min(int(k), scores.shape[-1]))
    row = np.asarray(scores).ravel()
    part = np.argpartition(-row, k - 1)[:k]
    order = sorted(part.tolist(), key=lambda i: (-float(row[i]), i))
    return [(int(i), float(row[i])) for i in order]


__all__ = ["Encoder", "exact_search", "l2_normalize", "top_k_from_scores"]
