"""Pseudo-relevance feedback on the dense channel (docs/spec/02 §4 stage 5, gate G4).

The first ranking is evidence about the query. If the top few documents agree with each other, their average
direction is a better description of what was asked than the query vector alone — that is the whole idea, and on
a prose-against-code task it matters more than usual, because the query is written in a vocabulary the corpus
does not use.

    q' = normalise(q + α · Σ wᵢ·dᵢ)     over the top-m documents, wᵢ from their own similarity

Three properties keep it safe:

* **It uses only this query's own results** (INV-3). Nothing is pooled across queries, so a ranking still cannot
  depend on the batch it was evaluated in.
* **It costs no embedding.** The document vectors are already in the snapshot matrix; feedback is a weighted sum
  and one more matmul.
* **It is off until a gate turns it on** (G4, default `enabled: false`). Feedback helps when the top of the
  ranking is right and hurts when it is wrong, and which of those is true here is a measurement, not a guess. The
  second-pass score and rank become LTR features either way, so the ranker can learn when to trust it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_M = 3
DEFAULT_ALPHA = 0.3
MIN_WEIGHT = 1e-6


def feedback_vector(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    scores: Sequence[float],
    *,
    m: int = DEFAULT_M,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """`normalise(q + α · Σ wᵢ·dᵢ)` over the top-`m` documents, weighted by their similarity.

    Weights are the similarities themselves, floored at zero: a document the query disagrees with should not be
    subtracted from the query, which is what a negative weight would do.
    """
    if m <= 0 or alpha <= 0 or doc_vectors.size == 0:
        return query_vector
    take = min(m, doc_vectors.shape[0], len(scores))
    weights = np.clip(np.asarray(scores[:take], dtype=np.float32), 0.0, None)
    total = float(weights.sum())
    if total <= MIN_WEIGHT:
        return query_vector

    centroid = (weights[:, None] * doc_vectors[:take]).sum(axis=0) / total
    expanded = np.asarray(query_vector, dtype=np.float32) + alpha * centroid.astype(np.float32)
    norm = float(np.linalg.norm(expanded))
    return (expanded / norm).astype(np.float32) if norm > 0 else query_vector


def expand(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    first_pass: Sequence[tuple[int, float]],
    *,
    m: int = DEFAULT_M,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """One feedback round. Returns `(expanded query, second-pass scores over the whole matrix)`.

    The second pass scores every document again rather than re-ranking the first pass: feedback exists to find
    what the first pass missed, and a re-rank of the first pass cannot do that by construction.
    """
    if not first_pass:
        return query_vector, np.asarray([], dtype=np.float32)
    rows = np.asarray([i for i, _ in first_pass[:m]], dtype=np.int64)
    scores = [s for _, s in first_pass[:m]]
    expanded = feedback_vector(query_vector, matrix[rows], scores, m=m, alpha=alpha)
    return expanded, (expanded.reshape(1, -1) @ matrix.T)[0]


__all__ = ["DEFAULT_ALPHA", "DEFAULT_M", "MIN_WEIGHT", "expand", "feedback_vector"]
