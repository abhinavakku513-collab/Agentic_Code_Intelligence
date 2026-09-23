"""Gate G-M: the rule that picks the encoder (D4, docs/spec/02 §3).

The rule is deliberately not "pick the best score" — resource use is scored by the organisers, so a slightly
weaker model that is far smaller and faster is often the better submission. That makes the rule easy to get
subtly wrong in a direction nobody notices, because any choice it makes looks defensible. It is a pure function
of the measured table, so it is tested exhaustively here without a single model.
"""

from __future__ import annotations

import pytest

from acis.core.errors import InvalidInput
from acis.embed.scorecard import Scorecard, percentiles, project_cold_pass_hours, render_table
from acis.eval.bakeoff import Candidate, GateRule, decide, record_decision
from acis.eval.bakeoff import render_table as render_candidates

RULE = GateRule(size_tolerance_pts=1.0, slow_pass_hours=2.0, slow_tolerance_pts=3.0, slo_cold_pass_hours=4.0)


def card(key: str, ndcg: float, params: int | None, *, hours: float | None = 1.0, **kw) -> Candidate:
    scorecard = (
        Scorecard(
            model=key,
            params=params,
            model_mb=None,
            numeric_profile="cpu-fp32",
            peak_rss_mb=1.0,
            cold_seconds=1.0,
            warm_seconds=0.5,
            latency_ms={},
            throughput_rows_per_s=10.0,
            threads=8,
            projected_cold_pass_hours=hours,
        )
        if hours is not None
        else None
    )
    return Candidate(key=key, name=key, ndcg_at_10=ndcg, params=params, scorecard=scorecard, **kw)


# -- the core trade-off ---------------------------------------------------------------------------------------
def test_the_smallest_model_within_tolerance_wins_not_the_best():
    """The whole point of D4: resource use is scored, so a 1-point sacrifice can buy a 4x smaller model."""
    decision = decide(
        [card("big", 75.0, 600_000_000), card("small", 74.3, 150_000_000), card("tiny", 60.0, 47_000_000)],
        rule=RULE,
    )
    assert decision.best == "big"
    assert decision.winner == "small"  # within 1.0 pt and far smaller
    assert "tiny" in decision.rejected  # 15 pt behind is not a trade-off, it is a different model


def test_a_candidate_outside_tolerance_is_rejected_with_the_gap_named():
    decision = decide([card("best", 75.0, 600_000_000), card("far", 73.5, 10_000_000)], rule=RULE)
    assert decision.winner == "best"
    assert "1.50 pt behind" in decision.rejected["far"]


def test_ties_on_size_are_broken_by_accuracy():
    decision = decide([card("a", 74.5, 100_000_000), card("b", 74.9, 100_000_000)], rule=RULE)
    assert decision.winner == "b"


def test_an_unknown_parameter_count_is_not_treated_as_small():
    """An unmeasured size must not win the "smallest" tie-break by default."""
    decision = decide([card("known", 74.5, 100_000_000), card("unknown", 74.6, None)], rule=RULE)
    assert decision.winner == "known"


# -- the widening rule ----------------------------------------------------------------------------------------
def test_tolerance_widens_when_the_best_candidate_is_slow():
    """If the best model's cold pass would dominate the run, a larger accuracy sacrifice is worth it."""
    decision = decide(
        [card("slow_best", 75.0, 900_000_000, hours=3.5), card("fast", 72.5, 120_000_000, hours=0.8)],
        rule=RULE,
    )
    assert decision.tolerance_widened and decision.tolerance_pts == 3.0
    assert decision.winner == "fast"  # 2.5 pt behind, which the widened tolerance allows


def test_tolerance_does_not_widen_when_the_best_is_fast():
    decision = decide(
        [card("fast_best", 75.0, 900_000_000, hours=1.0), card("fast", 72.5, 120_000_000, hours=0.8)],
        rule=RULE,
    )
    assert not decision.tolerance_widened and decision.tolerance_pts == 1.0
    assert decision.winner == "fast_best"


def test_a_candidate_over_the_slo_is_rejected_however_good_it_is():
    decision = decide(
        [card("enormous", 90.0, 700_000_000, hours=9.0), card("usable", 70.0, 200_000_000, hours=1.0)],
        rule=RULE,
    )
    assert decision.winner == "usable"
    assert "SLO" in decision.rejected["enormous"]


# -- the envelope ---------------------------------------------------------------------------------------------
def test_a_non_permissive_model_is_measured_but_never_selected():
    """D4: knowing the gap the permissive-only rule costs is exactly why it is measured."""
    decision = decide(
        [card("cc_by_nc", 86.0, 300_000_000, permissive=False), card("apache", 75.0, 600_000_000)],
        rule=RULE,
    )
    assert decision.best == "cc_by_nc"  # it is still the best, and the table says so
    assert decision.winner == "apache"
    assert "non-permissive" in decision.rejected["cc_by_nc"]


def test_an_unpinned_model_cannot_be_selected():
    decision = decide([card("unpinned", 75.0, 100_000_000, pinned=False), card("pinned", 74.5, 600_000_000)], rule=RULE)
    assert decision.winner == "pinned"
    assert "not pinned" in decision.rejected["unpinned"]


def test_a_model_over_the_parameter_envelope_is_rejected():
    decision = decide([card("huge", 80.0, 4_000_000_000), card("ok", 79.5, 600_000_000)], rule=RULE)
    assert decision.winner == "ok"
    assert "envelope" in decision.rejected["huge"]


def test_a_reference_only_candidate_is_excluded_by_request():
    decision = decide([card("ref", 90.0, 100_000_000, reference_only=True), card("real", 70.0, 600_000_000)], rule=RULE)
    assert decision.winner == "real" and "reference only" in decision.rejected["ref"]


# -- the decision set -------------------------------------------------------------------------------------------
def test_a_candidate_measured_on_a_subset_is_not_a_decision():
    """docs/spec/09 §2: gates decide on all 5,000 dev queries, because 1,000 cannot see a 1-point difference."""
    decision = decide([card("subset", 80.0, 100_000_000, n_queries=200), card("full", 70.0, 600_000_000)], rule=RULE)
    assert decision.winner == "full"
    assert "not the full decision set" in decision.rejected["subset"]


def test_the_frozen_default_stands_when_nothing_qualifies():
    """Every gate has a frozen default, and 'no candidate qualified' must land on it rather than on nothing."""
    decision = decide([card("only", 75.0, 100_000_000, n_queries=10)], rule=RULE)
    assert decision.default_applied and decision.winner == "qwen3-embedding-0.6b"
    assert decision.best is None


def test_an_empty_table_is_an_error_not_a_default():
    with pytest.raises(InvalidInput, match="at least one"):
        decide([], rule=RULE)


# -- provenance ---------------------------------------------------------------------------------------------------
def test_the_decision_explains_every_rejection():
    candidates = [
        card("best", 75.0, 600_000_000),
        card("small", 74.5, 150_000_000),
        card("far", 70.0, 10_000_000),
        card("nc", 80.0, 100_000_000, permissive=False),
    ]
    decision = decide(candidates, rule=RULE)
    for key in ("far", "nc"):
        assert decision.rejected[key]
    table = render_candidates(candidates, decision)
    assert "**selected**" in table and "non-permissive" in table


def test_a_decided_gate_is_superseded_not_edited(tmp_path):
    """`.claude/rules/mteb-eval.md`: gate decisions are written once; changing one needs an ADR."""
    path = tmp_path / "G-M.yaml"
    decision = decide([card("a", 75.0, 100_000_000)], rule=RULE)
    record_decision(decision, path=path)
    assert "decided" in path.read_text(encoding="utf-8")
    with pytest.raises(InvalidInput, match="ADR"):
        record_decision(decision, path=path)


def test_the_rule_is_read_from_the_gate_config_not_hard_coded():
    loaded = GateRule.load()
    assert loaded.size_tolerance_pts == 1.0
    assert loaded.slow_tolerance_pts == 3.0
    assert loaded.slo_cold_pass_hours == 4.0
    assert loaded.max_params_b == 1.0


# -- the scorecard ------------------------------------------------------------------------------------------------
def test_an_unknown_projection_is_not_a_passing_one():
    unknown = Scorecard(
        model="x",
        params=None,
        model_mb=None,
        numeric_profile="cpu-fp32",
        peak_rss_mb=1.0,
        cold_seconds=1.0,
        warm_seconds=1.0,
        latency_ms={},
        throughput_rows_per_s=0.0,
        threads=8,
        projected_cold_pass_hours=None,
    )
    assert unknown.within_slo is None


def test_projection_refuses_a_meaningless_rate():
    assert project_cold_pass_hours(0.0, total_rows=1000) is None
    assert project_cold_pass_hours(10.0, total_rows=36_000) == pytest.approx(1.0)


def test_percentiles_of_nothing_are_nothing_not_zero():
    assert percentiles([]) == {}
    assert percentiles([0.1, 0.2, 0.3])["p50"] == 200.0


def test_the_scorecard_table_names_every_scored_dimension():
    """Running time, model size and GPU requirement are scored (FAQ), so each must appear."""
    table = render_table(
        [
            Scorecard(
                model="m",
                params=1,
                model_mb=2.0,
                numeric_profile="cpu-fp32",
                peak_rss_mb=3.0,
                cold_seconds=4.0,
                warm_seconds=5.0,
                latency_ms={"p95": 6.0},
                throughput_rows_per_s=7.0,
                threads=8,
                projected_cold_pass_hours=9.0,
            )
        ]
    )
    for column in ("params", "MB", "peak RSS", "cold", "warm", "p95", "rows/s", "GPU"):
        assert column in table
    assert "none" in table  # GPU
