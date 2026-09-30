"""Weighted dense + lexical fusion — the generic route's ranking (docs/spec/02 §4 stage 8, §6b; spec 10 R-Q2).

The learned ranker is trained on APPS problem statements and stays off everything else. Before this module, the
generic route answered with the dense order alone, although the lexical channel had already run and its ranks were
shown next to every hit: a query naming `dijkstra` could see the one document defining `dijkstra` sit at #6 behind
unrelated programs whose cosines differed from it in the third decimal.

The function is the convex combination of each channel's scores after min–max normalisation within **this
query's own candidate pool** (INV-3: nothing is normalised across queries):

    fused(d) = α · minmax(cos)(d) + (1 − α) · bm25(d) / max bm25

BM25 is normalised against its theoretical minimum, zero, rather than the pool's minimum: a document BM25 did not
retrieve has no lexical evidence, and a query whose rare identifier matches exactly one document must keep that
one match rather than see it normalised away. A channel with no signal contributes nothing, so a query BM25 cannot
match at all is ranked by the dense channel exactly. α is the single parameter,
tuned on REG data with human-written queries (spec 10 §5) — never on APPS-derived stubs, never on TEST — and read
from `rank.generic.alpha`; α = 1 is the dense order.

Pure and deterministic: it returns scores, and the engine orders them with its content-hash tie-break (INV-4).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from acis.rank.candidates import Candidate


def _minmax(values: Mapping[str, float]) -> dict[str, float]:
    finite = [v for v in values.values() if math.isfinite(v)]
    if not finite:
        return {}
    low, high = min(finite), max(finite)
    if high - low <= 1e-12:
        return {}
    return {doc: (v - low) / (high - low) for doc, v in values.items() if math.isfinite(v)}


def _max_scaled(values: Mapping[str, float]) -> dict[str, float]:
    top = max((v for v in values.values() if math.isfinite(v)), default=0.0)
    if top <= 0.0:
        return {}
    return {doc: max(0.0, v) / top for doc, v in values.items() if math.isfinite(v)}


def weighted_fusion(
    pool: Sequence[Candidate],
    *,
    alpha: float,
    beta: float = 0.0,
    defined: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fused score per candidate. `pool` must carry each candidate's cosine (the engine fills missing ones).

    `beta` adds the corpus-defined-symbol evidence (summed IDF of the query names a unit *defines*, scaled by the
    pool's maximum): 0 leaves the tuned dense + BM25 fusion exactly as it was.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"fusion weight must be in [0, 1], got {alpha}")
    dense = _minmax({c.doc_id: c.dense_score for c in pool})
    lexical = _max_scaled({c.doc_id: c.lexical_score for c in pool if c.lexical_rank})
    symbolic = _max_scaled(dict(defined or {})) if beta else {}
    return {
        c.doc_id: alpha * dense.get(c.doc_id, 0.0)
        + (1.0 - alpha) * lexical.get(c.doc_id, 0.0)
        + beta * symbolic.get(c.doc_id, 0.0)
        for c in pool
    }


#: One identifier or dotted API name, optionally called: `dijkstra`, `heapq.heappush`, `lru_cache`, `sorted()`.
#: A pattern over the query's *shape*, not a list of names (INV-15): any identifier qualifies.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")


def identifier_query(query: str) -> str | None:
    """The identifier a query consists of, or `None` (spec 02 §6b: identifier-like queries)."""
    text = query.strip()
    if len(text) > 64 or not IDENTIFIER.match(text):
        return None
    return text.removesuffix("()")


def mentions(identifier: str, text: str) -> bool:
    """Whether `text` contains `identifier` as a whole token (`heapq.heappush`, not `myheapq.heappushx`)."""
    return re.search(r"(?<![A-Za-z0-9_.])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", text) is not None


__all__ = ["IDENTIFIER", "identifier_query", "mentions", "weighted_fusion"]
