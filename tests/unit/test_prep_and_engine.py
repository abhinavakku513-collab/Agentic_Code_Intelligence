"""Preprocessing, the lexical channel, the degradation counters and the engine's single-query surface."""

from __future__ import annotations

import pytest

from acis.core.config import freeze_config
from acis.core.errors import InvalidInput, NotFound, StrictViolation
from acis.core.types import SearchRequest, Snippet
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine, SearchEngine
from acis.engine.core import DEFAULT_CONFIG
from acis.lexical.bm25 import Bm25Index
from acis.lexical.tokenize import corpus_text
from acis.obs.counters import Counters, degradation
from acis.prep import normalize, truncate, views


@pytest.fixture
def engine(tiny_corpus):
    eng = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snap = eng.build_snapshot([Snippet(handle=i, text=t) for i, t in tiny_corpus], source="unit-test")
    return eng, snap


# -- normalisation ----------------------------------------------------------------------------------------------
def test_q1_is_idempotent_and_format_insensitive():
    raw = "Line one  \r\n\r\n\r\n\r\nLine two\t\n"
    once = normalize.q1(raw)
    assert normalize.q1(once) == once
    assert "\r" not in once and " " not in once
    assert once == normalize.q1("Line one\n\nLine two")


def test_q1_removes_control_characters_but_keeps_text():
    assert normalize.q1("abc\x00def\x01") == "abcdef"
    assert normalize.q1("\x1b[31mred") == "[31mred"


def test_d1_keeps_tabs_because_they_are_python_syntax():
    assert normalize.d1("def f():\n\treturn 1\n") == "def f():\n\treturn 1"
    assert normalize.d1("﻿x = 1\r\n") == "x = 1"


def test_numeric_folding_is_generic_not_a_list_of_constants():
    for spelling in ("10^9+7", "10^{9}+7", "1e9+7", "10 ^ 9 + 7"):
        assert normalize.fold_numeric_literals(spelling) == "1000000007", spelling
    assert normalize.fold_numeric_literals("10^9") == "1000000000"
    assert normalize.fold_numeric_literals("2e3-1") == "1999"
    # an absurd exponent is left alone rather than materialised
    assert normalize.fold_numeric_literals("10^99") == "10^99"


def test_lexical_view_lowercases_and_strips_latex():
    out = normalize.lexical_view("Compute $\\sum_{i}$ X modulo 10^9")
    assert "\\sum" not in out and "$" not in out
    assert out == out.lower()


def test_empty_query_detection():
    assert normalize.is_empty_query("   \n\t ")
    assert not normalize.is_empty_query("a")


# -- segmentation and views --------------------------------------------------------------------------------------
def test_segmentation_finds_generic_headings():
    text = "Solve it.\n\n-----Input-----\nOne integer n.\n\n-----Examples-----\nInput\n1\nOutput\n1"
    seg = views.segment(text)
    assert seg.segmented and seg.statement.startswith("Solve it.")
    assert "One integer n." in seg.io_spec
    assert "1" in seg.examples


def test_segmentation_handles_markdown_and_labelled_headings():
    seg = views.segment("Task text\n\n## Input\nn\n\nNotes:\nbeware")
    assert seg.markers_found >= 2
    assert "n" in seg.io_spec and "beware" in seg.note


def test_segmentation_never_fails_on_arbitrary_text():
    """INV-15: no marker list gates correctness — an unsegmentable query still has a statement."""
    for text in ("normalize", "def f(): pass", "如何预处理", "a" * 5000):
        seg = views.segment(text)
        assert seg.statement and not seg.segmented


def test_v1_drops_examples_and_falls_back_to_v0():
    text = "Body\n\n-----Input-----\nn\n\n-----Examples-----\nlots of sample io"
    assert "sample io" not in views.build_view(text, "V1")
    assert views.build_view("no markers here", "V1") == normalize.q1("no markers here")
    assert views.build_view(text, "unknown-view") == normalize.q1(text)


# -- truncation --------------------------------------------------------------------------------------------------
def test_head_tail_keeps_both_ends():
    tokens = [f"t{i}" for i in range(2000)]
    result = truncate.head_tail(" ".join(tokens), max_tokens=100, head=60, tail=40)
    kept = result.text.split()
    assert result.truncated and result.n_tokens == 2000
    assert kept[:3] == ["t0", "t1", "t2"] and kept[-3:] == ["t1997", "t1998", "t1999"]
    assert len(kept) == 100


def test_short_text_is_untouched():
    result = truncate.head_tail("a b c", max_tokens=100)
    assert not result.truncated and result.text == "a b c"


def test_truncation_rejects_nonsense_budgets():
    with pytest.raises(ValueError):
        truncate.head_tail("x", max_tokens=0)


# -- lexical channel ---------------------------------------------------------------------------------------------
def test_bm25_ranks_the_matching_document_first(tiny_corpus):
    index = Bm25Index.build([i for i, _ in tiny_corpus], [corpus_text("", t) for _, t in tiny_corpus])
    hits = index.search_one("palindrome print YES NO", k=3)
    assert hits[0][0] == "d5"


def test_bm25_single_query_equals_its_row_in_a_batch(tiny_corpus):
    """INV-3 at the channel level."""
    index = Bm25Index.build([i for i, _ in tiny_corpus], [corpus_text("", t) for _, t in tiny_corpus])
    queries = ["heapq dijkstra", "palindrome", "gcd while"]
    batch = index.retrieve(queries, k=3)
    for q, row in zip(queries, batch, strict=True):
        assert index.search_one(q, k=3) == row


def test_bm25_rejects_mismatched_inputs():
    with pytest.raises(InvalidInput, match="same length"):
        Bm25Index.build(["a"], ["x", "y"])


def test_a_corpus_that_tokenises_to_nothing_is_an_empty_channel_not_a_crash():
    """Found by the property suite: `bm25s` raises from inside its own vocabulary construction.

    A corpus of bare numbers and punctuation has no indexable term. That is a real corpus, not a bug in the
    caller, so the index matches nothing and says so — the dense channel is unaffected and the documents stay
    retrievable (spec 02 §6b: a failure never removes the baseline).
    """
    index = Bm25Index.build(["a", "b"], ["0:0.0", "0.0:0"])
    assert index.vocabulary_empty
    assert index.search_one("anything", k=5) == []
    assert index.retrieve(["one", "two"], k=5) == [[], []]


def test_a_snapshot_reports_the_lexical_channel_it_does_not_have():
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snapshot = engine.build_snapshot(
        [Snippet(handle="a", text="0:0.0"), Snippet(handle="b", text="0.0:0")], source="empty-vocab"
    )
    assert "lexical" in snapshot.missing_channels and "dense" not in snapshot.missing_channels
    hits = engine.search_batch(snapshot, ["q"], ["still has to answer"], top_k=2)["q"]
    assert len(hits) == 2  # INV-10 holds whatever the lexical channel can offer


# -- counters (INV-7) ---------------------------------------------------------------------------------------------
def test_degradation_counts_and_records():
    counters = Counters()
    degradation("dense_unavailable", "lexical only", counters=counters)
    degradation("dense_unavailable", "lexical only", counters=counters)
    assert counters.get("degradation.dense_unavailable") == 2
    assert counters.degradations() == ("dense_unavailable:lexical only",)


def test_strict_mode_turns_a_degradation_into_a_failure():
    counters = Counters()
    with pytest.raises(StrictViolation, match="strict mode forbids"):
        degradation("dense_unavailable", strict=True, counters=counters)
    assert counters.get("degradation.dense_unavailable") == 1  # still counted


# -- engine ------------------------------------------------------------------------------------------------------
def test_engine_satisfies_the_frozen_protocol():
    assert isinstance(AcisEngine.from_config(), SearchEngine)


def test_snapshot_is_content_addressed_and_stable(tiny_corpus):
    docs = [Snippet(handle=i, text=t) for i, t in tiny_corpus]
    a = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    b = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    assert a.build_snapshot(docs, source="x").snapshot_id == b.build_snapshot(docs, source="x").snapshot_id
    assert a.build_snapshot(docs, source="y").snapshot_id != b.build_snapshot(docs, source="x").snapshot_id


def test_snapshot_rejects_duplicate_handles_and_empty_input():
    eng = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    with pytest.raises(InvalidInput, match="duplicate document handles"):
        eng.build_snapshot([Snippet("d1", "a"), Snippet("d1", "b")], source="x")
    with pytest.raises(InvalidInput, match="zero documents"):
        eng.build_snapshot([], source="x")


def test_search_returns_evidence_read_from_the_store(engine, tiny_corpus):
    """INV-1: `source` is the exact stored text, re-read by body hash."""
    eng, _ = engine
    response = eng.search(SearchRequest(query="palindrome print YES", top_k=3))
    assert len(response.results) == 3
    texts = dict(tiny_corpus)
    for hit in response.results:
        from acis.prep.normalize import d1

        assert hit.source == d1(texts[hit.unit.key])
        assert hit.unit.body_hash and hit.unit.snapshot_id == response.snapshot.id


def test_search_rejects_an_empty_query(engine):
    eng, _ = engine
    with pytest.raises(InvalidInput, match="empty"):
        eng.search(SearchRequest(query="   "))


def test_search_truncates_an_over_long_query_instead_of_failing(engine):
    eng, _ = engine
    response = eng.search(SearchRequest(query="word " * 20000, top_k=2))
    assert response.interpreted_intent["query_truncated"] is True
    assert len(response.results) == 2


def test_search_top_k_is_capped_by_the_corpus(engine):
    eng, _ = engine
    response = eng.search(SearchRequest(query="gcd", top_k=1000))
    assert len(response.results) == 8


def test_unknown_version_raises_not_found(engine):
    eng, _ = engine
    with pytest.raises(NotFound):
        eng.search(SearchRequest(query="gcd", version="v99"))


def test_hybrid_mode_fuses_both_channels_and_says_what_it_fell_back_to(tiny_corpus):
    """Without a trained ranker or a tuned fusion weight, the hybrid channel serves the dense order — counted,
    never silent.

    The fallback chain is the point (INV-7): a missing ranking stage is a *degradation*, not a quiet identity, so a
    run manifest says whether the specified ranking actually served the query. No bank here, so the query takes the
    generic route, whose specified ranking is the weighted fusion.
    """
    cfg = freeze_config({**DEFAULT_CONFIG, "run": {**DEFAULT_CONFIG["run"], "channel": "hybrid"}})
    eng = AcisEngine.from_config(cfg, encoder=HashingEncoder())
    snap = eng.build_snapshot([Snippet(handle=i, text=t) for i, t in tiny_corpus], source="x")

    hits = eng.search_batch(snap, ["q"], ["gcd while loop"], top_k=3)["q"]
    assert len(hits) == 3
    assert [d for d, _ in hits] == sorted({d for d, _ in hits}, key=[d for d, _ in hits].index)
    assert eng.counters.get("degradation.fusion_untuned") >= 1


def test_hybrid_mode_refuses_to_degrade_silently_in_a_strict_run(tiny_corpus):
    cfg = freeze_config({**DEFAULT_CONFIG, "run": {**DEFAULT_CONFIG["run"], "channel": "hybrid", "strict": True}})
    eng = AcisEngine.from_config(cfg, encoder=HashingEncoder())
    snap = eng.build_snapshot([Snippet(handle=i, text=t) for i, t in tiny_corpus], source="x")
    with pytest.raises(StrictViolation):
        eng.search_batch(snap, ["q"], ["gcd"], top_k=3, strict=True)


def test_lexical_only_engine_records_a_degradation(tiny_corpus):
    cfg = freeze_config({**DEFAULT_CONFIG, "run": {**DEFAULT_CONFIG["run"], "channel": "lexical"}})
    eng = AcisEngine.from_config(cfg, encoder=None)
    snap = eng.build_snapshot([Snippet(handle=i, text=t) for i, t in tiny_corpus], source="x")
    ranked = eng.search_batch(snap, ["q"], ["heapq dijkstra priority queue"], top_k=3)
    assert ranked["q"][0][0] == "d6"
    assert "dense" in snap.missing_channels


def test_short_channel_results_are_padded_without_using_corpus_position(tiny_corpus):
    """INV-10 needs `min(top_k, N)` entries even when a channel returns fewer; P5 needs the padding to be

    independent of corpus arrangement. Both are checked here because the padding path is easy to leave untested:
    bm25s normally returns `k` hits, so it only triggers when a channel genuinely returns a short list.
    """
    forward = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    reverse = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snap_f = forward.build_snapshot([Snippet(handle=i, text=t) for i, t in tiny_corpus], source="x")
    snap_r = reverse.build_snapshot([Snippet(handle=i, text=t) for i, t in reversed(tiny_corpus)], source="x")
    data_f, data_r = forward.snapshot_data(snap_f), reverse.snapshot_data(snap_r)

    partial_f = [("d5", 2.0), ("d1", 1.0)]  # a channel that retrieved only two documents
    partial_r = [("d5", 2.0), ("d1", 1.0)]
    padded_f = AcisEngine._extend_with_unretrieved(data_f, partial_f, 6)
    padded_r = AcisEngine._extend_with_unretrieved(data_r, partial_r, 6)

    assert len(padded_f) == 6
    assert [d for d, _ in padded_f[:2]] == ["d5", "d1"]  # retrieved documents keep their order
    assert [d for d, _ in padded_f] == [d for d, _ in padded_r]  # and the tail ignores corpus arrangement
    tail_scores = {s for _, s in padded_f[2:]}
    assert len(tail_scores) == 1 and max(tail_scores) < min(s for _, s in partial_f)


def test_the_whole_frozen_surface_is_implemented(engine):
    """Phase 1 froze this interface; B1 and B2 filled it in. Every method on it is now real.

    `retrieve_evolution` needs a repository, so calling it without one is `InvalidInput` — an argument error,
    not "not built yet", which is the distinction the frozen interface exists to make visible.
    """
    eng, _ = engine
    from acis.core.types import EvolveRequest

    for name in ("ingest", "index", "update_version", "compare_versions", "activate", "rollback"):
        assert callable(getattr(eng, name)), name
    with pytest.raises(InvalidInput, match="repository"):
        eng.retrieve_evolution(EvolveRequest(query="x"))


def test_diagnostics_reports_no_agent_calls(engine):
    eng, _ = engine
    diag = eng.diagnostics()
    assert diag.agent_calls == 0  # INV-13
    assert diag.config_hash == eng.config_hash
    assert diag.snapshots and diag.snapshots[0]["units"] == 8
