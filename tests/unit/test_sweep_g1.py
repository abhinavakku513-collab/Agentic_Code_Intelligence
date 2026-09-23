"""Gate G1: the query-representation sweep (docs/spec/02 §2, §5; docs/spec/03 §7; INV-15).

G1 chooses, **per route**, the task string, the view and the truncation length. Like G-M it is a rule rather than
an argmax — resource use is scored, so a cell that ties the best and encodes half as much should win — and like
G-M the rule is a pure function of the measured table, so it is tested here without a model.

The failure this guards against is specific: a sweep that adopts the best cell every time will happily spend two
encodes per query for a tenth of a point of noise, and that decision then rides in the submission for good.
"""

from __future__ import annotations

import pytest

from acis.core.errors import InvalidInput
from acis.eval.sweep import Cell, Measurement, decide_g1, record_decision, render_table

QUERIES = [f"q{i}" for i in range(200)]


def measurement(cell: Cell, base: float, *, lift: float = 0.0, offset: int = 0) -> Measurement:
    """A cell whose per-query NDCG@10 is `base`, lifted on a stable share of queries.

    Deterministic on purpose: the bootstrap has to see a *consistent* difference, which is exactly what makes a
    real one distinguishable from a lucky one.
    """
    scores = {}
    for i, q in enumerate(QUERIES):
        scores[q] = min(1.0, base + (lift if (i + offset) % 2 == 0 else 0.0))
    mean = sum(scores.values()) / len(scores)
    return Measurement(cell=cell, per_query=scores, metrics={"ndcg_at_10": mean}, seconds=1.0)


DEFAULT = Cell(task="T1", view="V0", max_tokens=1024)


# -- the trade-off ---------------------------------------------------------------------------------------------
def test_a_clear_improvement_is_adopted():
    cells = [
        measurement(DEFAULT, 0.40),
        measurement(Cell("T2", "V0", 1024), 0.40, lift=0.10),  # +5 pt on the mean
    ]
    decision = decide_g1(cells, route="statement_like", baseline=DEFAULT.key)
    assert decision.adopted and decision.winner == "T2/V0/1024"


def test_a_cheaper_cell_that_ties_the_best_wins():
    """Resource use is scored (FAQ): half the tokens for the same accuracy is a better submission."""
    cells = [
        measurement(DEFAULT, 0.40),
        measurement(Cell("T1", "V1", 512), 0.40),  # identical scores, a quarter of the encoding
    ]
    decision = decide_g1(cells, route="statement_like", baseline=DEFAULT.key)
    assert decision.winner == "T1/V1/512" and decision.adopted


def test_a_costlier_cell_needs_a_gated_win_not_a_nose():
    """+0.3 pt for two encodes per query is noise wearing a result's clothes."""
    cells = [
        measurement(DEFAULT, 0.40),
        measurement(Cell("T1", "V2", 1024), 0.40, lift=0.006),  # ≈ +0.3 pt, below the 0.5 pt threshold
    ]
    decision = decide_g1(cells, route="statement_like", baseline=DEFAULT.key)
    assert decision.winner == DEFAULT.key and not decision.adopted
    assert "V2" in " ".join(decision.rejected)


def test_the_default_stands_when_everything_is_worse():
    cells = [measurement(DEFAULT, 0.40), measurement(Cell("T3", "V0", 256), 0.30)]
    decision = decide_g1(cells, route="generic", baseline=DEFAULT.key)
    assert decision.winner == DEFAULT.key and not decision.adopted


def test_an_equal_and_equally_cheap_cell_does_not_displace_the_default():
    """Churn is a cost too: an exact tie at the same price keeps the incumbent."""
    cells = [measurement(DEFAULT, 0.40), measurement(Cell("T2", "V0", 1024), 0.40)]
    decision = decide_g1(cells, route="generic", baseline=DEFAULT.key)
    assert decision.winner == DEFAULT.key and not decision.adopted


# -- cost ordering ---------------------------------------------------------------------------------------------
def test_v2_counts_as_two_encodes_and_a_short_v2_can_still_be_the_cheapest():
    assert Cell("T1", "V2", 256).encoded_tokens < Cell("T1", "V0", 1024).encoded_tokens
    assert Cell("T1", "V1", 512).encoded_tokens == Cell("T1", "V0", 512).encoded_tokens
    assert Cell("T1", "V1", 512).cost_rank < Cell("T1", "V0", 512).cost_rank  # same cap, shorter text


# -- the decision set and the route ------------------------------------------------------------------------------
def test_cells_measured_on_different_queries_cannot_be_compared():
    a = measurement(DEFAULT, 0.40)
    b = Measurement(cell=Cell("T2", "V0", 1024), per_query={"other": 0.9}, metrics={"ndcg_at_10": 0.9}, seconds=1.0)
    with pytest.raises(InvalidInput, match="same queries"):
        decide_g1([a, b], route="generic", baseline=DEFAULT.key)


def test_a_sweep_of_one_cell_decides_nothing():
    with pytest.raises(InvalidInput, match="at least two"):
        decide_g1([measurement(DEFAULT, 0.4)], route="generic", baseline=DEFAULT.key)


def test_the_baseline_has_to_be_in_the_table():
    with pytest.raises(InvalidInput, match="baseline"):
        decide_g1(
            [measurement(Cell("T2", "V0", 1024), 0.4), measurement(Cell("T3", "V0", 512), 0.4)],
            route="generic",
            baseline=DEFAULT.key,
        )


def test_the_decision_names_its_route():
    """G1 runs per route (INV-15): a decision that does not say which route it belongs to is not usable."""
    cells = [measurement(DEFAULT, 0.40), measurement(Cell("T2", "V0", 1024), 0.40, lift=0.10)]
    assert decide_g1(cells, route="statement_like", baseline=DEFAULT.key).route == "statement_like"


# -- provenance ---------------------------------------------------------------------------------------------------
def test_the_table_shows_every_cell_with_its_delta_and_its_fate():
    cells = [
        measurement(DEFAULT, 0.40),
        measurement(Cell("T2", "V0", 1024), 0.40, lift=0.10),
        measurement(Cell("T3", "V0", 256), 0.30),
    ]
    decision = decide_g1(cells, route="statement_like", baseline=DEFAULT.key)
    table = render_table(cells, decision)
    assert "**selected**" in table and "baseline" in table
    for cell in ("T1/V0/1024", "T2/V0/1024", "T3/V0/256"):
        assert cell in table


def test_a_decided_gate_is_superseded_not_edited(tmp_path):
    cells = [measurement(DEFAULT, 0.40), measurement(Cell("T2", "V0", 1024), 0.40, lift=0.10)]
    decision = decide_g1(cells, route="statement_like", baseline=DEFAULT.key)
    path = tmp_path / "G1.yaml"
    record_decision(decision, path=path)
    assert "statement_like" in path.read_text(encoding="utf-8")
    with pytest.raises(InvalidInput, match="ADR"):
        record_decision(decision, path=path)


def test_two_routes_are_two_decisions_in_one_file(tmp_path):
    """One route being decided must not freeze the other out of the file (INV-15: every route is first-class)."""
    path = tmp_path / "G1.yaml"
    cells = [measurement(DEFAULT, 0.40), measurement(Cell("T2", "V0", 1024), 0.40, lift=0.10)]
    record_decision(decide_g1(cells, route="statement_like", baseline=DEFAULT.key), path=path)
    record_decision(decide_g1(cells, route="generic", baseline=DEFAULT.key), path=path)
    text = path.read_text(encoding="utf-8")
    assert "statement_like" in text and "generic" in text
