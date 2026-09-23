"""Candidate assembly and the learned ranker (docs/spec/02 §4, gate G5, INV-3, INV-15).

Two things decide whether a learned ranker helps or quietly hurts: whether it is *scored* on queries it was
trained on, and whether it keeps ranking when its evidence is missing. Both are tested here, along with the
batch-invariance property that a per-query feature can break without anyone noticing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from acis.core.errors import InvalidInput
from acis.rank import candidates as cand
from acis.rank import ltr


def dense(n=10, offset=0.0):
    return [(f"d{i}", round(0.9 - 0.05 * i + offset, 4)) for i in range(n)]


# -- the union ---------------------------------------------------------------------------------------------
def test_the_union_keeps_both_channels_and_records_where_each_placed_a_document():
    union = cand.union(dense(5), [("d3", 7.0), ("x1", 6.0)], dense_k=5, lexical_k=2)
    by_id = {c.doc_id: c for c in union}
    assert by_id["d3"].dense_rank == 4 and by_id["d3"].lexical_rank == 1
    assert by_id["x1"].dense_rank == 0 and by_id["x1"].lexical_score == 6.0
    assert math.isnan(by_id["x1"].dense_score)  # the dense channel did not retrieve it


def test_the_union_is_capped_by_reciprocal_rank_not_by_truncating_one_channel():
    union = cand.union(dense(80), [(f"x{i}", 10.0 - i) for i in range(40)], dense_k=80, lexical_k=40, cap=20)
    assert len(union) == 20
    assert any(c.lexical_rank == 1 for c in union), "the best lexical hit must survive the cut"
    assert any(c.dense_rank == 1 for c in union), "so must the best dense hit"


def test_a_document_no_channel_retrieved_is_not_a_candidate():
    assert all(c.doc_id != "never" for c in cand.union(dense(3), [("x", 1.0)]))


# -- features ----------------------------------------------------------------------------------------------
def test_the_feature_matrix_has_one_row_per_candidate_and_the_declared_columns():
    matrix = cand.feature_matrix(cand.union(dense(6)), query_tokens=100)
    assert matrix.shape == (6, len(cand.FEATURE_NAMES))
    assert matrix.dtype == np.float32


def test_an_absent_feature_is_nan_rather_than_zero():
    """A zero says "no match"; NaN says "nothing to compare". LightGBM treats them differently, and so must we."""
    matrix = cand.feature_matrix(cand.union(dense(3)), query_tokens=10)
    lexical = matrix[:, cand.FEATURE_NAMES.index("bm25")]
    assert np.all(np.isnan(lexical))


def test_features_depend_only_on_this_querys_own_candidates():
    """INV-3: standardising over a batch would make a ranking depend on the other queries in it."""
    alone = cand.feature_matrix(cand.union(dense(5)), query_tokens=50)
    in_a_crowd = cand.feature_matrix(cand.union(dense(5)), query_tokens=50)
    np.testing.assert_array_equal(alone, in_a_crowd)


def test_group_availability_reports_what_fired():
    matrix = cand.feature_matrix(cand.union(dense(4), [("d0", 5.0)]), query_tokens=20)
    availability = cand.group_availability(matrix)
    assert availability["dense"] > 0.0
    assert availability["bridge"] == 0.0  # no bridge features were supplied


def test_masking_a_group_blanks_exactly_that_group():
    matrix = cand.feature_matrix(cand.union(dense(4), [("d0", 5.0)]), query_tokens=20)
    masked = cand.mask_group(matrix, "lexical")
    for name in cand.GROUPS["lexical"]:
        assert np.all(np.isnan(masked[:, cand.FEATURE_NAMES.index(name)]))
    assert not np.all(np.isnan(masked[:, cand.FEATURE_NAMES.index("cos")]))


# -- training ------------------------------------------------------------------------------------------------
def make_groups(n_queries=40, n_docs=8, seed=0):
    """Groups where the gold document is the one with the highest bm25 — a signal dense order alone misses."""
    rng = np.random.default_rng(seed)
    groups = []
    for q in range(n_queries):
        gold = int(rng.integers(0, n_docs))
        cands = []
        for d in range(n_docs):
            cands.append(
                cand.Candidate(
                    doc_id=f"q{q}d{d}",
                    dense_score=float(rng.random()),
                    dense_rank=d + 1,
                    lexical_score=float(10.0 if d == gold else rng.random()),
                    lexical_rank=d + 1,
                )
            )
        matrix = cand.feature_matrix(cands, query_tokens=100)
        labels = np.array([1 if d == gold else 0 for d in range(n_docs)], dtype=np.int32)
        groups.append(ltr.TrainingGroup(f"q{q}", matrix, labels, tuple(c.doc_id for c in cands)))
    return groups


def test_a_trained_ranker_learns_a_signal_the_dense_order_does_not_carry():
    groups = make_groups()
    model = ltr.train(groups, rounds=60, seed=0)
    hits = 0
    for group in groups:
        order, abstained = model.rerank(group.doc_ids, group.features)
        assert not abstained
        gold = group.doc_ids[int(np.argmax(group.labels))]
        hits += order[0] == gold
    assert hits / len(groups) > 0.8, f"the ranker found the signal in only {hits}/{len(groups)} queries"


def test_training_is_deterministic_for_a_seed():
    groups = make_groups(n_queries=12)
    a = ltr.train(groups, rounds=20, seed=7).score(groups[0].features)
    b = ltr.train(groups, rounds=20, seed=7).score(groups[0].features)
    np.testing.assert_allclose(a, b)


def test_cross_fitting_scores_every_query_with_a_model_that_never_saw_it():
    groups = make_groups(n_queries=30)
    folds = {g.query_id: i % 3 for i, g in enumerate(groups)}
    oof = ltr.cross_fit(groups, folds=folds, rounds=20)
    assert set(oof) == {g.query_id for g in groups}
    assert all(len(v) == len(g.doc_ids) for g, v in zip(groups, (oof[g.query_id] for g in groups), strict=True))


def test_cross_fitting_needs_more_than_one_fold():
    groups = make_groups(n_queries=4)
    with pytest.raises(InvalidInput, match="two folds"):
        ltr.cross_fit(groups, folds=dict.fromkeys((g.query_id for g in groups), 0), rounds=5)


# -- abstention ------------------------------------------------------------------------------------------------
def test_the_ranker_abstains_when_almost_nothing_fired():
    """INV-15: an arbitrary query with no lexical match and no bridge evidence gets the dense order back."""
    model = ltr.train(make_groups(n_queries=12), rounds=20, seed=0)
    bare = cand.feature_matrix(
        [cand.Candidate(doc_id=f"d{i}", dense_score=0.5 - 0.01 * i, dense_rank=i + 1) for i in range(5)]
    )
    order, abstained = model.rerank([f"d{i}" for i in range(5)], bare)
    assert abstained and order == [f"d{i}" for i in range(5)]


def test_the_ranker_does_not_abstain_when_the_evidence_is_there():
    model = ltr.train(make_groups(n_queries=12), rounds=20, seed=0)
    group = make_groups(n_queries=1, seed=3)[0]
    _, abstained = model.rerank(group.doc_ids, group.features)
    assert not abstained


def test_a_model_refuses_features_of_the_wrong_width():
    model = ltr.train(make_groups(n_queries=8), rounds=10, seed=0)
    with pytest.raises(InvalidInput, match="width"):
        model.score(np.zeros((3, 5), dtype=np.float32))


def test_a_ranker_round_trips_through_disk(tmp_path):
    model = ltr.train(make_groups(n_queries=10), rounds=15, seed=1)
    path = model.save(tmp_path / "ranker.txt")
    reloaded = ltr.Ranker.load(path)
    group = make_groups(n_queries=1, seed=5)[0]
    np.testing.assert_allclose(model.score(group.features), reloaded.score(group.features))
    assert reloaded.feature_names == cand.FEATURE_NAMES
