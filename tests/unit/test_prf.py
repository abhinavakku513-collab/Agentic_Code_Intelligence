"""Pseudo-relevance feedback (docs/spec/02 §4 stage 5, gate G4, INV-3).

Feedback is a good idea that becomes a bad one silently: when the top of the first ranking is wrong, expanding
towards it makes the second ranking worse, and nothing in the output says so. These tests pin the behaviour that
keeps it controllable — it moves towards agreement, it never subtracts, and it uses only this query's own results.
"""

from __future__ import annotations

import numpy as np
import pytest

from acis.rank import prf


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_feedback_moves_the_query_towards_documents_that_agree():
    query = unit(1.0, 0.0)
    docs = np.stack([unit(0.8, 0.6), unit(0.7, 0.7)])
    expanded = prf.feedback_vector(query, docs, [0.9, 0.8], m=2, alpha=0.5)
    assert float(expanded @ docs[0]) > float(query @ docs[0])


def test_the_expanded_vector_stays_normalised():
    expanded = prf.feedback_vector(unit(1.0, 0.0), np.stack([unit(0.0, 1.0)]), [0.5])
    assert float(np.linalg.norm(expanded)) == pytest.approx(1.0, abs=1e-6)


def test_a_document_the_query_disagrees_with_is_not_subtracted():
    """A negative weight would push the query *away* from a document, which is not what feedback means."""
    query = unit(1.0, 0.0)
    docs = np.stack([unit(-1.0, 0.0)])
    assert np.allclose(prf.feedback_vector(query, docs, [-0.9]), query)


def test_alpha_zero_and_m_zero_are_both_identities():
    query = unit(1.0, 1.0)
    docs = np.stack([unit(0.0, 1.0)])
    assert np.allclose(prf.feedback_vector(query, docs, [0.9], alpha=0.0), query)
    assert np.allclose(prf.feedback_vector(query, docs, [0.9], m=0), query)


def test_an_empty_first_pass_changes_nothing():
    query = unit(1.0, 0.0)
    expanded, scores = prf.expand(query, np.stack([unit(0.0, 1.0)]), [])
    assert np.allclose(expanded, query) and scores.size == 0


def test_the_second_pass_scores_the_whole_corpus_not_just_the_first_pass():
    """Feedback exists to find what the first pass missed; re-ranking the first pass cannot do that."""
    matrix = np.stack([unit(1.0, 0.0), unit(0.9, 0.4), unit(0.0, 1.0)])
    _, scores = prf.expand(unit(1.0, 0.0), matrix, [(0, 0.99)], m=1, alpha=0.4)
    assert scores.shape == (3,)


def test_feedback_uses_only_this_querys_own_results():
    """INV-3: the expansion is a function of the query and its own top documents, and of nothing else."""
    matrix = np.stack([unit(1.0, 0.0), unit(0.9, 0.4), unit(0.0, 1.0)])
    first = [(0, 0.99), (1, 0.93)]
    a, _ = prf.expand(unit(1.0, 0.0), matrix, first)
    b, _ = prf.expand(unit(1.0, 0.0), matrix, first)
    assert np.allclose(a, b)


def test_a_wrong_first_pass_is_visible_as_a_worse_second_pass():
    """The failure mode, made explicit: expanding towards a wrong top document moves the query away from truth."""
    gold, wrong = unit(1.0, 0.0), unit(0.0, 1.0)
    matrix = np.stack([wrong, gold])
    query = unit(0.9, 0.44)
    _, scores = prf.expand(query, matrix, [(0, 0.99)], m=1, alpha=0.8)
    assert scores[0] > scores[1]  # feedback confirmed the wrong document; a gate is the only defence
