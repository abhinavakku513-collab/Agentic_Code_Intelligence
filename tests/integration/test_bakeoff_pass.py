"""The G-M measurement pass, driven end to end on the stand-in encoder (docs/spec/02 §3, docs/spec/09 §2).

`decide()` is tested exhaustively as a pure function elsewhere. What is tested here is the half that has to
*produce* the table: build the encoder through the factory, run the engine over dev queries, score them, measure
the cost, and hand `decide()` something it can act on.

The stand-in encoder makes that runnable today, and it is also the case worth pinning: a model that cannot ship
must be measurable without ever becoming selectable. When weights land, the same command measures a real model —
that is the point of writing it now rather than the day the bake-off is due.
"""

from __future__ import annotations

import pytest

from acis.appsdata import apps
from acis.eval.bakeoff import GateRule, decide, measure_card, render_table

pytestmark = pytest.mark.skipif(not apps.is_available(), reason="dataset assets are not fetched")

LIMIT = 8  # enough to produce a ranking; far too few to decide anything, which is itself under test


@pytest.fixture(scope="module")
def stand_in():
    """One measured candidate, shared: the snapshot build embeds the whole corpus."""
    return measure_card("hashing", limit=LIMIT, top_k=100, record_row=False, sample=32)


def test_the_pass_produces_metrics_and_a_cost_for_one_candidate(stand_in):
    assert stand_in.metrics["ndcg_at_10"] >= 0.0
    assert stand_in.n_queries == LIMIT
    card = stand_in.scorecard
    assert card is not None
    assert card.cold_seconds > 0 and card.warm_seconds > 0
    assert card.projected_cold_pass_hours is not None  # a measured rate, extrapolated to the official pass
    assert card.gpu == "none"


def test_a_candidate_that_cannot_ship_is_measured_but_never_selected(stand_in):
    """The stand-in says `submission_capable = False`; that has to survive into the bake-off table."""
    assert stand_in.reference_only
    decision = decide([stand_in], rule=GateRule(), decision_set_size=LIMIT)
    assert decision.default_applied and decision.winner == "qwen3-embedding-0.6b"
    assert "reference only" in decision.rejected["hashing"]


def test_a_smoke_run_cannot_become_a_decision(stand_in):
    """8 queries cannot see a 1-point difference, so the rule must refuse them (docs/spec/09 §2)."""
    decision = decide([stand_in], rule=GateRule())
    assert "not the full decision set" in decision.rejected["hashing"]


def test_the_table_names_the_candidate_and_its_fate(stand_in):
    table = render_table([stand_in], decide([stand_in], rule=GateRule()))
    assert "hashing" in table and "not the full decision set" in table


def test_a_model_without_weights_stops_the_pass_rather_than_scoring_zero(tmp_path, monkeypatch):
    """A missing model must not be measured as a bad one: that number would enter the ledger and the table.

    `ACIS_HOME` is moved aside so this stays true after the owner fetches the real weights.
    """
    from acis.core.errors import NotReady

    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "empty-home"))
    with pytest.raises(NotReady, match="weights"):
        measure_card("qwen3-embedding-0.6b", limit=2, record_row=False)
