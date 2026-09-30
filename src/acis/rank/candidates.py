"""Candidate assembly and the feature vector the ranker sees (docs/spec/02 §4 stages 6–8, Appendix A).

The union is deliberately small and deliberately not a merge of scores: dense top-100 ∪ BM25 top-30, capped at
100, with reciprocal rank fusion used only to *cut* when the union overflows. Mixing raw scores from two channels
before the ranker has seen them would throw away the thing the ranker is for.

Every feature here is computed **within one query's own candidate list** (INV-3): `z_cos` is standardised over
this query's candidates, never over a batch, so a query's ranking cannot depend on the other queries it happened
to be evaluated with. Features that cannot be computed are `NaN` rather than zero, and each belongs to a named
group so training can mask whole groups (≈25 % dropout) and no single extractor becomes load-bearing (INV-15).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NAN = float("nan")
RRF_K = 60.0

#: Feature groups, in the order their columns appear. Group dropout masks one whole group at a time, so the
#: ranker is trained never to depend on any single family of evidence.
GROUPS: dict[str, tuple[str, ...]] = {
    "dense": ("cos", "rank_dense", "z_cos", "gap_top1", "q_margin"),
    "prf": ("cos_prf", "rank_prf"),
    "lexical": ("bm25", "rank_lex", "bm25_norm", "rrf_prior"),
    "bridge": ("out_literal_recall", "numeric_literal_overlap", "const_jaccard", "io_shape_compat", "tc_loop_expected"),
    "meta": ("log_doc_tokens", "log_q_tokens", "len_ratio", "is_truncated", "parse_ok", "dup_cluster_size"),
    # A second, heterogeneous dense encoder (spec 08 §3 L10): its own cosine, rank and within-pool z-score.
    "dense2": ("cos2", "rank_dense2", "z_cos2"),
    # Exact symbols the query names (`acis.lexical.symbols`); NaN when the query names none.
    "symbol": ("symbol_hits", "symbol_coverage", "symbol_idf"),
    # Agreement: in how many retrieval channels' own top lists the unit appeared.
    "agree": ("n_channels",),
}
FEATURE_NAMES: tuple[str, ...] = tuple(name for names in GROUPS.values() for name in names)
#: Monotone constraints: more similarity and more lexical match may never lower a score (spec 02 §4).
MONOTONE = {"cos": 1, "bm25": 1, "cos2": 1}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One document under consideration for one query, with where each channel placed it."""

    doc_id: str
    dense_score: float = NAN
    dense_rank: int = 0
    lexical_score: float = NAN
    lexical_rank: int = 0
    prf_score: float = NAN
    prf_rank: int = 0
    aux_score: float = NAN
    aux_rank: int = 0
    symbol_score: float = NAN
    symbol_rank: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)


def reciprocal_rank(rank: int) -> float:
    """RRF contribution of a rank. Rank 0 means "this channel did not retrieve it", which contributes nothing."""
    return 1.0 / (RRF_K + rank) if rank > 0 else 0.0


def union(
    dense: Sequence[tuple[str, float]],
    lexical: Sequence[tuple[str, float]] = (),
    *,
    dense_k: int = 100,
    lexical_k: int = 30,
    cap: int = 100,
    aux: Sequence[tuple[str, float]] = (),
    aux_k: int = 0,
    symbol: Sequence[tuple[str, float]] = (),
    symbol_k: int = 0,
) -> list[Candidate]:
    """The union of every channel's top list, cut to `cap` by reciprocal rank fusion over all channels.

    RRF appears **only** at the cut: it decides which candidates survive when the union is too large, and never
    what the final order is. That stays the ranker's job (or the generic route's fusion).
    """
    lists = {
        "dense": list(dense[:dense_k]),
        "lexical": list(lexical[:lexical_k]),
        "aux": list(aux[:aux_k]),
        "symbol": list(symbol[:symbol_k]),
    }
    ranks = {name: {doc: i for i, (doc, _) in enumerate(items, start=1)} for name, items in lists.items()}
    scores = {name: dict(items) for name, items in lists.items()}

    ordered: list[str] = []
    seen: set[str] = set()
    for name in ("dense", "lexical", "aux", "symbol"):
        for doc, _ in lists[name]:
            if doc not in seen:
                seen.add(doc)
                ordered.append(doc)
    if len(ordered) > cap:
        ordered.sort(key=lambda d: -sum(reciprocal_rank(r.get(d, 0)) for r in ranks.values()))
        ordered = ordered[:cap]

    return [
        Candidate(
            doc_id=doc,
            dense_score=scores["dense"].get(doc, NAN),
            dense_rank=ranks["dense"].get(doc, 0),
            lexical_score=scores["lexical"].get(doc, NAN),
            lexical_rank=ranks["lexical"].get(doc, 0),
            aux_score=scores["aux"].get(doc, NAN),
            aux_rank=ranks["aux"].get(doc, 0),
            symbol_score=scores["symbol"].get(doc, NAN),
            symbol_rank=ranks["symbol"].get(doc, 0),
        )
        for doc in ordered
    ]


def _standardise(values: Sequence[float]) -> list[float]:
    """z-scores over *this query's own* candidates (INV-3). A degenerate spread yields `NaN`, not a divide by zero."""
    finite = [v for v in values if v == v]
    if len(finite) < 2:
        return [NAN] * len(values)
    mean = sum(finite) / len(finite)
    variance = sum((v - mean) ** 2 for v in finite) / len(finite)
    sd = math.sqrt(variance)
    if sd <= 1e-12:
        return [NAN] * len(values)
    return [((v - mean) / sd) if v == v else NAN for v in values]


def feature_matrix(
    candidates: Sequence[Candidate],
    *,
    bridge: Mapping[str, Mapping[str, float]] | None = None,
    query_tokens: int = 0,
    doc_meta: Mapping[str, Mapping[str, Any]] | None = None,
    symbols: Mapping[str, Mapping[str, float]] | None = None,
) -> np.ndarray:
    """`(n_candidates, len(FEATURE_NAMES))` float32, in `FEATURE_NAMES` order, `NaN` where a feature is absent."""
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    dense_scores = [c.dense_score for c in candidates]
    top_dense = max((v for v in dense_scores if v == v), default=NAN)
    second = sorted((v for v in dense_scores if v == v), reverse=True)[1:2]
    q_margin = (top_dense - second[0]) if (second and top_dense == top_dense) else NAN
    z_scores = _standardise(dense_scores)

    lexical_scores = [c.lexical_score for c in candidates]
    top_lex = max((v for v in lexical_scores if v == v), default=NAN)
    z_aux = _standardise([c.aux_score for c in candidates])

    rows: list[list[float]] = []
    for index, candidate in enumerate(candidates):
        doc = dict((doc_meta or {}).get(candidate.doc_id, {}))
        bridge_row = dict((bridge or {}).get(candidate.doc_id, {}))
        doc_tokens = float(doc.get("n_tokens", 0) or 0)
        values: dict[str, float] = {
            "cos": candidate.dense_score,
            "rank_dense": float(candidate.dense_rank) if candidate.dense_rank else NAN,
            "z_cos": z_scores[index],
            "gap_top1": (top_dense - candidate.dense_score) if candidate.dense_score == candidate.dense_score else NAN,
            "q_margin": q_margin,
            "cos_prf": candidate.prf_score,
            "rank_prf": float(candidate.prf_rank) if candidate.prf_rank else NAN,
            "bm25": candidate.lexical_score,
            "rank_lex": float(candidate.lexical_rank) if candidate.lexical_rank else NAN,
            "bm25_norm": (candidate.lexical_score / top_lex)
            if (top_lex and top_lex == top_lex and top_lex > 0 and candidate.lexical_score == candidate.lexical_score)
            else NAN,
            # Fusion evidence, and only that: with one channel it is a monotone function of that channel's own
            # rank and carries nothing new. Reporting it anyway would make the lexical group look present for a
            # query BM25 never matched, and group availability is exactly what abstention reads (INV-15).
            "rrf_prior": (
                reciprocal_rank(candidate.dense_rank) + reciprocal_rank(candidate.lexical_rank)
                if candidate.lexical_rank
                else NAN
            ),
            "log_doc_tokens": math.log1p(doc_tokens) if doc_tokens else NAN,
            "log_q_tokens": math.log1p(query_tokens) if query_tokens else NAN,
            "len_ratio": (doc_tokens / query_tokens) if (doc_tokens and query_tokens) else NAN,
            "is_truncated": float(bool(doc.get("is_truncated"))) if "is_truncated" in doc else NAN,
            "parse_ok": float(bool(doc["parse_ok"])) if doc.get("parse_ok") is not None else NAN,
            "dup_cluster_size": float(doc.get("dup_cluster_size", NAN)),
            **{name: float(bridge_row.get(name, NAN)) for name in GROUPS["bridge"]},
            "cos2": candidate.aux_score,
            "rank_dense2": float(candidate.aux_rank) if candidate.aux_rank else NAN,
            "z_cos2": z_aux[index],
            **{name: float((symbols or {}).get(candidate.doc_id, {}).get(name, NAN)) for name in GROUPS["symbol"]},
            "n_channels": float(
                sum(
                    1
                    for r in (candidate.dense_rank, candidate.lexical_rank, candidate.aux_rank, candidate.symbol_rank)
                    if r
                )
            ),
        }
        rows.append([float(values.get(name, NAN)) for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32)


def group_availability(matrix: np.ndarray) -> dict[str, float]:
    """Share of each group's columns that fired, averaged over candidates — the ρ the ranker abstains on."""
    out: dict[str, float] = {}
    if matrix.size == 0:
        return dict.fromkeys(GROUPS, 0.0)
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for group, names in GROUPS.items():
        columns = [index[n] for n in names]
        present = np.isfinite(matrix[:, columns])
        out[group] = float(present.mean()) if present.size else 0.0
    return out


def mask_group(matrix: np.ndarray, group: str) -> np.ndarray:
    """Blank one whole group — training-time dropout, so no family of evidence becomes load-bearing (INV-15)."""
    if group not in GROUPS:
        raise KeyError(group)
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out = matrix.copy()
    for name in GROUPS[group]:
        out[:, index[name]] = NAN
    return out


__all__ = [
    "FEATURE_NAMES",
    "GROUPS",
    "MONOTONE",
    "NAN",
    "RRF_K",
    "Candidate",
    "feature_matrix",
    "group_availability",
    "mask_group",
    "reciprocal_rank",
    "union",
]
