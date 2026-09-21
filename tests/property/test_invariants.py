"""Property tests for the invariants that cannot be checked by example (docs/spec/06 §5).

INV-2 snapshot isolation · INV-3 batch invariance · INV-4 opaque ids · score monotonicity under float32 ·
cache-key coverage · hash stability under reformatting.
"""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from acis.core.config import freeze_config
from acis.core.hashing import hash_obj, normalized_text_key
from acis.core.types import Snippet
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG
from acis.eval.metrics import as_float32
from acis.rank.compose import rank_derived_scores

SLOW = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])

code_text = st.text(alphabet=string.ascii_letters + string.digits + " \n_().,:=+-*%[]", min_size=5, max_size=200)
query_text = st.text(alphabet=string.ascii_letters + string.digits + " \n?.,", min_size=3, max_size=120).filter(
    lambda s: s.strip()
)


def build_engine(texts: list[str]) -> tuple[AcisEngine, object]:
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    docs = [Snippet(handle=f"d{i}", text=t) for i, t in enumerate(texts)]
    return engine, engine.build_snapshot(docs, source="property")


# -- INV-3: batch invariance ---------------------------------------------------------------------------------------
@given(
    texts=st.lists(code_text, min_size=3, max_size=12, unique=True),
    queries=st.lists(query_text, min_size=2, max_size=5),
)
@SLOW
def test_a_query_ranking_never_depends_on_its_batch(texts, queries):
    engine, snapshot = build_engine(texts)
    ids = [f"q{i}" for i in range(len(queries))]
    batched = engine.search_batch(snapshot, ids, queries, top_k=min(5, len(texts)))
    for qid, text in zip(ids, queries, strict=True):
        alone = engine.search_batch(snapshot, [qid], [text], top_k=min(5, len(texts)))
        assert alone[qid] == batched[qid]


@given(texts=st.lists(code_text, min_size=3, max_size=10, unique=True), query=query_text)
@SLOW
def test_batch_order_does_not_change_a_ranking(texts, query):
    engine, snapshot = build_engine(texts)
    k = min(5, len(texts))
    first = engine.search_batch(snapshot, ["a", "b"], [query, "unrelated filler text"], top_k=k)
    second = engine.search_batch(snapshot, ["b", "a"], ["unrelated filler text", query], top_k=k)
    assert first["a"] == second["a"]


# -- INV-4: ids are opaque -----------------------------------------------------------------------------------------
@given(texts=st.lists(code_text, min_size=3, max_size=10, unique=True), query=query_text)
@SLOW
def test_relabelling_documents_permutes_ids_but_not_content(texts, query):
    """The ranking must follow the *content*, whatever the documents are called."""
    engine_a, snap_a = build_engine(texts)
    engine_b = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snap_b = engine_b.build_snapshot(
        [Snippet(handle=f"zz{i:04d}", text=t) for i, t in enumerate(texts)], source="property"
    )
    k = min(5, len(texts))
    a = engine_a.search_batch(snap_a, ["q"], [query], top_k=k)["q"]
    b = engine_b.search_batch(snap_b, ["q"], [query], top_k=k)["q"]
    rank_of_a = [int(d[1:]) for d, _ in a]
    rank_of_b = [int(d[2:]) for d, _ in b]
    assert rank_of_a == rank_of_b


@given(texts=st.lists(code_text, min_size=3, max_size=10, unique=True), query=query_text)
@SLOW
def test_query_id_does_not_reach_the_ranking(texts, query):
    engine, snapshot = build_engine(texts)
    k = min(5, len(texts))
    first = engine.search_batch(snapshot, ["q1"], [query], top_k=k)["q1"]
    second = engine.search_batch(snapshot, ["completely-different-id"], [query], top_k=k)["completely-different-id"]
    assert first == second


# -- INV-2: snapshot isolation -------------------------------------------------------------------------------------
@given(
    left=st.lists(code_text, min_size=2, max_size=6, unique=True),
    right=st.lists(code_text, min_size=2, max_size=6, unique=True),
    query=query_text,
)
@SLOW
def test_results_come_only_from_the_requested_snapshot(left, right, query):
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snap_left = engine.build_snapshot([Snippet(handle=f"L{i}", text=t) for i, t in enumerate(left)], source="left")
    snap_right = engine.build_snapshot([Snippet(handle=f"R{i}", text=t) for i, t in enumerate(right)], source="right")
    hits_left = engine.search_batch(snap_left, ["q"], [query], top_k=len(left))["q"]
    hits_right = engine.search_batch(snap_right, ["q"], [query], top_k=len(right))["q"]
    assert all(d.startswith("L") for d, _ in hits_left)
    assert all(d.startswith("R") for d, _ in hits_right)


@given(texts=st.lists(code_text, min_size=2, max_size=6, unique=True))
@SLOW
def test_the_query_vector_cache_key_covers_snapshot_and_config(texts):
    """INV-2: a cached vector may never cross a snapshot or a config boundary."""
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snap_a = engine.build_snapshot([Snippet(handle=f"a{i}", text=t) for i, t in enumerate(texts)], source="a")
    snap_b = engine.build_snapshot([Snippet(handle=f"b{i}", text=t) for i, t in enumerate(texts)], source="b")
    engine.search_batch(snap_a, ["q"], ["some query"], top_k=1)
    keys_after_first = set(engine._query_vector_cache)  # noqa: SLF001 — the cache key is the property under test
    engine.search_batch(snap_b, ["q"], ["some query"], top_k=1)
    assert len(set(engine._query_vector_cache) - keys_after_first) == 1  # noqa: SLF001


# -- score composition ---------------------------------------------------------------------------------------------
@given(n=st.integers(min_value=1, max_value=1000), top_k=st.integers(min_value=1, max_value=1000))
@settings(max_examples=50, deadline=None)
def test_rank_derived_scores_stay_distinct_in_float32(n, top_k):
    """D9: the gaps must survive the float32 comparison pytrec_eval performs."""
    ids = [f"d{i}" for i in range(n)]
    scores = rank_derived_scores(ids, top_k)
    values = [as_float32(v) for v in scores.values()]
    assert len(scores) == min(n, top_k)
    assert len(set(values)) == len(values)
    assert all(a > b for a, b in zip(values, values[1:], strict=False))


# -- hashing -------------------------------------------------------------------------------------------------------
@given(st.dictionaries(st.text(min_size=1, max_size=8), st.integers() | st.text(max_size=8) | st.none(), max_size=6))
@settings(max_examples=50, deadline=None)
def test_structural_hash_ignores_key_order(mapping):
    shuffled = dict(reversed(list(mapping.items())))
    assert hash_obj(mapping) == hash_obj(shuffled)


@given(st.text(min_size=1, max_size=120))
@settings(max_examples=50, deadline=None)
def test_normalised_text_key_is_stable_under_whitespace_reformatting(text):
    reformatted = "  \n ".join(text.split())
    if text.split():
        assert normalized_text_key(text) == normalized_text_key(reformatted)


@given(st.lists(st.text(min_size=1, max_size=6), min_size=1, max_size=20, unique=True))
@settings(max_examples=50, deadline=None)
def test_config_hash_ignores_thread_count(names):
    a = freeze_config({"model": {"names": names}, "run": {"threads": 1}})
    b = freeze_config({"model": {"names": names}, "run": {"threads": 128}})
    assert a.config_hash == b.config_hash
