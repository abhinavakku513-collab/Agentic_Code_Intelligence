"""Score composition for Mode A (D9, INV-10, docs/spec/03 §2.6).

The engine ranks; it does not report raw similarities. Mode A emits **rank-derived** scores
`(top_k + 1 - rank) / top_k`, because the grading harness does not compare floats the way the model does:
pytrec_eval collapses scores closer than the float32 spacing at their magnitude, while mteb's MRR compares exact
floats, so a raw-score ranking can score MRR@10 = 1.0 and NDCG@10 = 0.0 at once (docs/spec/01 V-08).

Rank-derived scores are finite, strictly decreasing, gapped by `1/top_k` (1e-3 at top_k = 1000), and carry no
information beyond the order — which is all the harness consumes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from acis.core.errors import InvalidInput

Hits = Sequence[tuple[str, float]] | Sequence[str]


def _doc_ids(hits: Hits) -> list[str]:
    out: list[str] = []
    for item in hits:
        out.append(str(item[0]) if isinstance(item, tuple) else str(item))
    return out


def rank_derived_scores(hits: Hits, top_k: int) -> dict[str, float]:
    """`{doc_id: (top_k + 1 - rank) / top_k}` for the first `top_k` hits, in the order given.

    Duplicated document ids are a caller bug (the same document cannot hold two ranks), so they raise rather than
    silently collapsing into one entry and breaking the `min(top_k, N)` contract.
    """
    if top_k <= 0:
        raise InvalidInput("top_k must be positive", top_k=top_k)
    ids = _doc_ids(hits)[:top_k]
    if len(set(ids)) != len(ids):
        raise InvalidInput("duplicate document ids in a ranking", n=len(ids), unique=len(set(ids)))
    return {doc_id: (top_k + 1 - rank) / top_k for rank, doc_id in enumerate(ids, start=1)}


def assert_mode_a_contract(scores: Mapping[str, float], *, top_k: int, corpus_size: int) -> None:
    """INV-10: exactly `min(top_k, N)` finite, strictly decreasing entries per query."""
    expected = min(top_k, corpus_size)
    values = list(scores.values())
    if len(scores) != expected:
        raise InvalidInput("wrong number of ranked entries", expected=expected, got=len(scores))
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):  # noqa: PLR0124 — NaN check
        raise InvalidInput("non-finite score in a ranking")
    if any(a <= b for a, b in zip(values, values[1:], strict=False)):
        raise InvalidInput("scores are not strictly decreasing")


def order_with_duplicate_tiebreak(
    ranked: Sequence[str], *, body_hash_of: Mapping[str, str], ordinal_of: Mapping[str, int]
) -> list[str]:
    """Keep exact-content duplicates adjacent and ordered by corpus ordinal — the only use of order (INV-4).

    Documents with identical content are otherwise interchangeable; without a rule their relative order would depend
    on floating-point noise. The corpus ordinal is a deterministic, content-independent tiebreak, and it is never a
    feature or a score offset.
    """
    seen: dict[str, int] = {}
    groups: list[list[str]] = []
    for doc_id in ranked:
        key = body_hash_of.get(doc_id, doc_id)
        if key in seen:
            groups[seen[key]].append(doc_id)
        else:
            seen[key] = len(groups)
            groups.append([doc_id])
    out: list[str] = []
    for group in groups:
        out.extend(sorted(group, key=lambda d: (ordinal_of.get(d, 1 << 30), d)))
    return out


__all__ = ["assert_mode_a_contract", "order_with_duplicate_tiebreak", "rank_derived_scores"]
