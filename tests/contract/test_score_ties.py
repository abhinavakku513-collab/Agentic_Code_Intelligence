"""V-08 as an executable contract: why Mode-A scores are rank-derived (docs/spec/01 V-08, D9, INV-10).

pytrec_eval (used by mteb for NDCG) compares scores at ~float32 resolution and breaks near-ties by reverse doc-id order,
while mteb's own MRR compares Python floats. A naive epsilon ramp can therefore give MRR@10 = 1.0 with NDCG@10 < 1.0 on the
SAME ranking. Rank-derived scores (top_k + 1 - rank) / top_k never do.
"""

from __future__ import annotations

import math

import pytest

pytrec_eval = pytest.importorskip("pytrec_eval")

GOLD = "d3"  # 'x*' sorts after 'd*' -> reverse doc-id tie-breaking puts distractors FIRST (worst case for the gold)


def _evaluator():
    return pytrec_eval.RelevanceEvaluator({"q": {GOLD: 1}}, {"ndcg_cut.10", "recip_rank"})


def _ranked_ids(n: int, gold_rank: int) -> list[str]:
    ids = [f"x{i:04d}" for i in range(n)]
    ids.insert(gold_rank - 1, GOLD)
    return ids[:n]


def rank_derived(ids: list[str], top_k: int) -> dict[str, float]:
    return {d: (top_k + 1 - r) / top_k for r, d in enumerate(ids, start=1)}


def epsilon_ramp(ids: list[str], base: float, gap: float) -> dict[str, float]:
    return {d: base - r * gap for r, d in enumerate(ids)}


def mteb_style_mrr(run: dict[str, float], k: int = 10) -> float:
    order = sorted(run.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # mteb sorts by (score, doc_id) desc
    for r, (d, _) in enumerate(order[:k], start=1):
        if d == GOLD:
            return 1.0 / r
    return 0.0


@pytest.mark.parametrize("gold_rank", [1, 2, 3, 7, 10, 11, 500])
@pytest.mark.parametrize("top_k,n", [(1000, 1000), (1000, 3000)])
def test_rank_derived_scores_are_exact(gold_rank, top_k, n):
    ids = _ranked_ids(n, gold_rank)[:top_k]
    if GOLD not in ids:
        pytest.skip("gold outside top_k")
    run = rank_derived(ids, top_k)
    res = _evaluator().evaluate({"q": run})["q"]
    expected_ndcg = 1.0 / math.log2(gold_rank + 1) if gold_rank <= 10 else 0.0
    assert res["ndcg_cut_10"] == pytest.approx(expected_ndcg, abs=1e-9)
    assert mteb_style_mrr(run) == pytest.approx(1.0 / gold_rank if gold_rank <= 10 else 0.0, abs=1e-12)
    assert list(run.values()) == sorted(run.values(), reverse=True) and len(set(run.values())) == len(run)


@pytest.mark.parametrize("base,gap", [(1.0, 1e-9), (5.0, 1e-9), (5.0, 1e-7), (100.0, 1e-6)])
def test_hazard_naive_epsilon_ramps_collapse_into_ties(base, gap):
    """Documents the hazard. If this FAILS, pytrec_eval changed behaviour: re-read V-08 before relaxing D9."""
    ids = _ranked_ids(30, 1)  # our ranking puts the gold first
    run = epsilon_ramp(ids, base, gap)
    res = _evaluator().evaluate({"q": run})["q"]
    assert mteb_style_mrr(run) == 1.0  # mteb MRR sees the exact-float order ...
    assert res["ndcg_cut_10"] < 1.0  # ... pytrec_eval NDCG sees ties and demotes the gold


def test_gap_1e5_is_not_enough_at_magnitude_200():
    """An absolute '>= 1e-5 gap' rule is unsafe for raw scores >= ~128: float32 spacing there is 1.5e-5, so two scores that are
    exactly 1e-5 apart can still round to the SAME float32 value (alignment-dependent, so this row pins a bad alignment)."""
    run = {GOLD: 200.000005, "x0000": 199.999995, "x0001": 199.99998}  # gold first, gap to the next = 1.0e-5
    assert run[GOLD] - run["x0000"] == pytest.approx(1e-5, rel=1e-3)
    assert mteb_style_mrr(run) == 1.0  # mteb MRR: gold is rank 1
    assert _evaluator().evaluate({"q": run})["q"]["ndcg_cut_10"] < 1.0  # pytrec_eval: tie -> demoted
