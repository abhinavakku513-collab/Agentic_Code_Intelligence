"""The heterogeneous candidate union: exact symbols, a second dense encoder, exact scores, ranker compatibility."""

from __future__ import annotations

import math

import numpy as np

from acis.core.config import freeze_config
from acis.core.types import Snippet
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG
from acis.lexical.symbols import SymbolIndex, query_symbols, unit_symbols
from acis.rank import candidates as cand
from acis.rank import ltr

DOCS = [
    Snippet(handle="a", text="import heapq\ndef dijkstra(g, s):\n    h = [(0, s)]\n    heapq.heappush(h, (1, s))\n"),
    Snippet(handle="b", text="def sum_intervals(xs):\n    return sum(b - a for a, b in xs)\n"),
    Snippet(handle="c", text="class UnionFind:\n    def find(self, x):\n        return x\n"),
    Snippet(handle="d", text="n = int(input())\nprint(n * 2)\n"),
    Snippet(handle="e", text="def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"),
]


def test_query_symbols_are_found_by_shape_not_by_list():
    assert query_symbols("find the shortest path in a weighted graph") == ()
    assert query_symbols("an array (sorted) of ints") == ()
    assert query_symbols("use UnionFind with path compression") == ("UnionFind",)
    assert query_symbols("Write `sum_intervals()` for the list") == ("sum_intervals",)
    assert query_symbols("call heapq.heappush then solve(n)")[:2] == ("solve", "heapq.heappush")
    assert query_symbols("dijkstra") == ("dijkstra",)  # a query that is one identifier names it


def test_the_symbol_index_ranks_units_containing_rare_symbols_first():
    index = SymbolIndex.build([d.handle for d in DOCS], [d.text for d in DOCS])
    assert "heapq" in unit_symbols(DOCS[0].text) and "heapq.heappush" in unit_symbols(DOCS[0].text)
    assert [d for d, _ in index.search(["sum_intervals"], 5)] == ["b"]
    assert index.search([], 5) == [] and index.search(["nothing_here"], 5) == []


def test_the_union_keeps_every_channel_and_cuts_by_rrf_only_when_over_cap():
    pool = cand.union(
        [("a", 0.9), ("b", 0.8)],
        [("c", 5.0)],
        dense_k=2,
        lexical_k=1,
        cap=10,
        aux=[("d", 0.7)],
        aux_k=1,
        symbol=[("e", 3.0)],
        symbol_k=1,
    )
    assert [c.doc_id for c in pool] == ["a", "b", "c", "d", "e"]
    assert pool[3].aux_rank == 1 and pool[4].symbol_rank == 1 and pool[0].dense_rank == 1
    assert len(cand.union([("a", 0.9), ("b", 0.8)], [("c", 5.0)], dense_k=2, lexical_k=1, cap=2)) == 2


def _engine(**extra: object) -> AcisEngine:
    cfg = {
        **DEFAULT_CONFIG,
        "run": {**DEFAULT_CONFIG["run"], "channel": "hybrid"},
        "retrieve": {**DEFAULT_CONFIG["retrieve"], "symbol_k": 5, **extra.pop("retrieve", {})},  # type: ignore[dict-item]
        "rank": {"generic": {"alpha": 0.9}},
        **extra,
    }
    return AcisEngine.from_config(freeze_config(cfg), encoder=HashingEncoder(dim=64))


def test_a_symbol_hit_enters_the_pool_and_every_candidate_carries_an_exact_cosine():
    engine = _engine()
    snap = engine.build_snapshot(DOCS, source="t")
    data = engine.snapshot_data(snap)
    _, pool = engine.candidate_pool(data, "UnionFind", route="generic", want=5)
    assert any(c.doc_id == "c" and c.symbol_rank for c in pool)
    assert all(not math.isnan(c.dense_score) for c in pool)  # lexical- or symbol-only units are scored too
    matrix = engine.pool_features(data, "UnionFind", pool)
    col = cand.FEATURE_NAMES.index("symbol_hits")
    hits = {c.doc_id: matrix[i, col] for i, c in enumerate(pool)}
    assert hits["c"] == 1.0 and hits.get("d", 0.0) == 0.0


def test_no_symbol_in_the_query_means_nan_symbol_features_not_zero():
    engine = _engine()
    data = engine.snapshot_data(engine.build_snapshot(DOCS, source="t"))
    _, pool = engine.candidate_pool(data, "double a number read from input", route="generic", want=5)
    matrix = engine.pool_features(data, "double a number read from input", pool)
    assert np.isnan(matrix[:, cand.FEATURE_NAMES.index("symbol_hits")]).all()


def test_a_second_dense_encoder_adds_its_own_channel_and_features():
    engine = _engine(model={"encoder": "hashing", "aux_encoder": "hashing", "dim": 64}, retrieve={"aux_k": 3})
    data = engine.snapshot_data(engine.build_snapshot(DOCS, source="t"))
    assert data.aux_vectors is not None and data.aux_vectors.shape[0] == len(DOCS)
    _, pool = engine.candidate_pool(data, "reverse the words", route="generic", want=5)
    assert any(c.aux_rank for c in pool)
    matrix = engine.pool_features(data, "reverse the words", pool)
    assert np.isfinite(matrix[:, cand.FEATURE_NAMES.index("cos2")]).all()


def test_an_older_ranker_reads_its_own_columns_from_the_wider_matrix():
    rng = np.random.default_rng(0)
    old_names = tuple(n for n in cand.FEATURE_NAMES if n not in ("cos2", "rank_dense2", "z_cos2", "n_channels"))
    groups = []
    for q in range(40):
        full = rng.standard_normal((6, len(cand.FEATURE_NAMES))).astype(np.float32)
        labels = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        groups.append((f"q{q}", full, labels))
    model = ltr.train([ltr.TrainingGroup(q, m, y, tuple("abcdef")) for q, m, y in groups], rounds=5)
    older = ltr.Ranker(booster=model.booster, feature_names=cand.FEATURE_NAMES)
    order, _ = older.rerank(list("abcdef"), groups[0][1])
    assert sorted(order) == list("abcdef")
    # A model that names a feature the pipeline no longer produces is refused, never fed shifted columns.
    stale = ltr.Ranker(booster=model.booster, feature_names=(*old_names, "gone_feature"))
    try:
        stale.score(groups[0][1])
    except Exception as exc:  # noqa: BLE001
        assert "feature width" in str(exc)
    else:
        raise AssertionError("a stale feature list must be refused")
