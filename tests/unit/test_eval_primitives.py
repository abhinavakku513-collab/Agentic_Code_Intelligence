"""Unit behaviour of the evaluation primitives: metrics, score composition, run files, ledger, bootstrap, decontam."""

from __future__ import annotations

import json
import math

import pytest

from acis.core.errors import InvalidInput
from acis.eval import bootstrap, decontam, ledger, metrics, runfile
from acis.rank.compose import assert_mode_a_contract, order_with_duplicate_tiebreak, rank_derived_scores

GOLD = "d7"
QRELS = {"q1": {GOLD: 1}}


def ranking(gold_rank: int, n: int = 50) -> dict[str, float]:
    ids = [f"x{i:03d}" for i in range(n)]
    ids.insert(gold_rank - 1, GOLD)
    return rank_derived_scores(ids[:n], n)


# -- metrics ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gold_rank", [1, 2, 3, 10, 11, 50])
def test_ndcg_matches_the_closed_form_for_one_relevant_document(gold_rank):
    scores = metrics.score_run(QRELS, {"q1": ranking(gold_rank)}, (10,))
    expected = 1.0 / math.log2(gold_rank + 1) if gold_rank <= 10 else 0.0
    assert scores["ndcg_at_10"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("gold_rank", [1, 2, 5, 10, 11])
def test_mrr_is_one_over_the_rank(gold_rank):
    scores = metrics.score_run(QRELS, {"q1": ranking(gold_rank)}, (10,))
    assert scores["mrr_at_10"] == pytest.approx(1.0 / gold_rank if gold_rank <= 10 else 0.0, abs=1e-12)


def test_recall_precision_and_hit_rate_for_one_relevant_document():
    scores = metrics.score_run(QRELS, {"q1": ranking(3)}, (1, 10))
    assert scores["recall_at_10"] == 1.0
    assert scores["recall_at_1"] == 0.0
    assert scores["precision_at_10"] == pytest.approx(0.1)
    assert scores["hit_rate_at_10"] == 1.0
    assert scores["hit_rate_at_1"] == 0.0
    assert scores["accuracy"] == scores["recall_at_1"]


def test_queries_without_a_run_entry_are_not_scored():
    """pytrec_eval scores the intersection; a query we never answered must not silently count as a zero."""
    qrels = {"q1": {GOLD: 1}, "q2": {GOLD: 1}}
    scores = metrics.score_run(qrels, {"q1": ranking(1)}, (10,))
    assert scores["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics.shared_query_ids(qrels, {"q1": {}}) == ["q1"]


def test_empty_run_scores_zero_without_raising():
    assert metrics.score_run(QRELS, {}, (10,))["ndcg_at_10"] == 0.0


def test_rank_order_breaks_ties_by_doc_id_descending():
    order = metrics.rank_order({"a": 1.0, "b": 1.0, "c": 2.0})
    assert order == ["c", "b", "a"]


def test_rank_order_is_float32_but_mrr_order_is_exact():
    """The two graders disagree inside the float32 collapse zone — the whole reason for D9 (V-08)."""
    scores = {"d1": 1.0, "d2": 1.0 - 1e-9}
    assert metrics.as_float32(1.0) == metrics.as_float32(1.0 - 1e-9)
    assert metrics.rank_order(scores) == ["d2", "d1"]  # collapsed to a tie, broken by doc id
    assert metrics.mrr_order(scores) == ["d1", "d2"]  # exact floats keep the real order


def test_map_at_k_uses_the_total_relevant_count():
    qrels = {"q1": {"a": 1, "b": 1}}
    run = {"q1": rank_derived_scores(["a", "z", "b"], 3)}
    # precision at the two relevant ranks: 1/1 and 2/3, divided by 2 relevant documents
    assert metrics.score_run(qrels, run, (3,))["map_at_3"] == pytest.approx((1.0 + 2 / 3) / 2)


# -- score composition (D9, INV-10) -------------------------------------------------------------------------------
def test_rank_derived_scores_are_strictly_decreasing_and_gapped():
    scores = rank_derived_scores([f"d{i}" for i in range(1000)], 1000)
    values = list(scores.values())
    assert values[0] == 1.0
    assert all(a - b == pytest.approx(1e-3) for a, b in zip(values, values[1:], strict=False))
    assert len(set(values)) == len(values)


def test_rank_derived_scores_survive_float32():
    """A score gap that vanishes in float32 would make NDCG and MRR disagree (V-08)."""
    scores = rank_derived_scores([f"d{i}" for i in range(1000)], 1000)
    as32 = [metrics.as_float32(v) for v in scores.values()]
    assert len(set(as32)) == len(as32)


def test_rank_derived_scores_reject_duplicates_and_bad_top_k():
    with pytest.raises(InvalidInput, match="duplicate"):
        rank_derived_scores(["d1", "d1"], 10)
    with pytest.raises(InvalidInput):
        rank_derived_scores(["d1"], 0)


def test_mode_a_contract_checks_count_and_monotonicity():
    scores = rank_derived_scores(["a", "b", "c"], 10)
    assert_mode_a_contract(scores, top_k=10, corpus_size=3)
    with pytest.raises(InvalidInput, match="wrong number"):
        assert_mode_a_contract(scores, top_k=10, corpus_size=10)
    with pytest.raises(InvalidInput, match="strictly decreasing"):
        assert_mode_a_contract({"a": 1.0, "b": 1.0}, top_k=2, corpus_size=2)
    with pytest.raises(InvalidInput, match="non-finite"):
        assert_mode_a_contract({"a": float("nan"), "b": 0.5}, top_k=2, corpus_size=2)


def test_duplicate_tiebreak_keeps_duplicates_adjacent_in_corpus_order():
    body = {"a": "h1", "b": "h2", "c": "h1"}
    ordinal = {"a": 5, "b": 1, "c": 2}
    assert order_with_duplicate_tiebreak(["a", "b", "c"], body_hash_of=body, ordinal_of=ordinal) == ["c", "a", "b"]


# -- run files -------------------------------------------------------------------------------------------------------
def test_run_files_round_trip_exactly(tmp_path):
    run = {"q1": {"d1": 1.0, "d2": 0.999}, "q2": {"d3": 0.5}}
    paths = runfile.write_run(run, tmp_path)
    assert runfile.runs_equal(run, runfile.read_trec(paths["run.trec"]))
    assert runfile.runs_equal(run, runfile.read_csv(paths["run.csv"]))


def test_trec_rows_are_written_in_grading_order(tmp_path):
    run = {"q1": {"a": 0.1, "b": 0.9, "c": 0.5}}
    path = runfile.write_trec(run, tmp_path / "run.trec")
    ranks = [line.split()[2] for line in path.read_text(encoding="utf-8").splitlines()]
    assert ranks == ["b", "c", "a"]


def test_csv_header_is_the_declared_schema(tmp_path):
    path = runfile.write_csv({"q1": {"a": 1.0}}, tmp_path / "run.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "query_id,corpus_id,rank,score"


def test_rescore_from_trec_reproduces_the_metrics(tmp_path):
    run = {"q1": ranking(2)}
    path = runfile.write_trec(run, tmp_path / "run.trec")
    direct = metrics.score_run(QRELS, run)
    rescored = runfile.rescore_trec(path, QRELS)
    assert all(rescored[k] == pytest.approx(direct[k], abs=1e-12) for k in direct)


def test_malformed_trec_raises(tmp_path):
    bad = tmp_path / "run.trec"
    bad.write_text("q1 Q0 d1\n", encoding="utf-8")
    with pytest.raises(InvalidInput, match="malformed"):
        runfile.read_trec(bad)


# -- ledger ------------------------------------------------------------------------------------------------------------
@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    return tmp_path / "ledger.jsonl"


def test_ledger_chains_rows_and_verifies(isolated_ledger):
    first = ledger.append({"kind": "dev", "metrics": {"ndcg_at_10": 0.1}})
    second = ledger.append({"kind": "dev", "metrics": {"ndcg_at_10": 0.2}})
    assert first.payload["prev_hash"] == ledger.GENESIS
    assert second.payload["prev_hash"] == first.row_hash
    assert ledger.verify_chain() == []
    assert len(ledger.read_rows()) == 2


def test_tampering_with_a_row_breaks_the_chain(isolated_ledger):
    ledger.append({"kind": "dev", "metrics": {"ndcg_at_10": 0.1}})
    ledger.append({"kind": "dev", "metrics": {"ndcg_at_10": 0.2}})
    lines = isolated_ledger.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["metrics"]["ndcg_at_10"] = 0.99
    isolated_ledger.write_text("\n".join([json.dumps(row, sort_keys=True), lines[1]]) + "\n", encoding="utf-8")
    problems = ledger.verify_chain()
    assert problems and "row_hash" in problems[0]


def test_held_out_touch_budget_is_enforced(isolated_ledger):
    for _ in range(ledger.TEST_TOUCH_BUDGET):
        ledger.append({"kind": "rc", "test_touch_count": 1})
    assert ledger.test_touches_used() == ledger.TEST_TOUCH_BUDGET
    with pytest.raises(InvalidInput, match="budget"):
        ledger.append({"kind": "rc", "test_touch_count": 1})


def test_ledger_rejects_unknown_kinds(isolated_ledger):
    with pytest.raises(InvalidInput, match="unknown ledger kind"):
        ledger.LedgerRowBuilder(kind="whatever").build()


def test_run_id_is_derived_from_content(isolated_ledger):
    row = ledger.append({"kind": "dev", "rung": "bm25_acis", "metrics": {"ndcg_at_10": 0.5}})
    assert row.run_id.startswith("dev-") and len(row.run_id) == len("dev-") + 12


# -- bootstrap ------------------------------------------------------------------------------------------------------------
def test_paired_bootstrap_detects_a_known_effect():
    base = {f"q{i}": 0.50 for i in range(2000)}
    system = {f"q{i}": 0.52 for i in range(2000)}  # +2.0 points, no variance
    result = bootstrap.paired_bootstrap(system, base, resamples=2000)
    assert result.delta == pytest.approx(2.0, abs=1e-6)
    assert result.ci_excludes_zero and result.passes()


def test_paired_bootstrap_rejects_noise():
    import random

    rng = random.Random(0)
    base = {f"q{i}": rng.random() for i in range(2000)}
    system = {f"q{i}": base[f"q{i}"] + rng.gauss(0, 0.3) for i in range(2000)}
    result = bootstrap.paired_bootstrap(system, base, resamples=2000)
    assert not result.passes()


def test_paired_bootstrap_is_seeded():
    a = {f"q{i}": i / 100 for i in range(200)}
    b = {f"q{i}": (i + 1) / 100 for i in range(200)}
    first = bootstrap.paired_bootstrap(a, b, resamples=500)
    second = bootstrap.paired_bootstrap(a, b, resamples=500)
    assert first.as_dict() == second.as_dict()


def test_paired_bootstrap_pairs_on_shared_queries_only():
    result = bootstrap.paired_bootstrap({"q1": 1.0, "q2": 1.0}, {"q1": 0.0}, resamples=100)
    assert result.n == 1


def test_gate_rule_needs_both_size_and_confidence():
    small = bootstrap.BootstrapResult(100, 0.5, 0.49, 0.1, 0.05, 0.2, 0.1, 10, 0)
    assert small.ci_excludes_zero and not small.passes()  # CI is fine, effect is too small


def test_holm_adjustment_is_monotone():
    adjusted = bootstrap.holm_adjust([0.01, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert adjusted[0] == pytest.approx(0.03)


# -- decontamination ----------------------------------------------------------------------------------------------------
def test_decontamination_drops_near_duplicates_and_keeps_the_rest():
    reference = ["Given a string s, print YES if it is a palindrome and NO otherwise."]
    train = {
        "keep": "Compute the shortest path between two vertices of a weighted graph.",
        "drop": "Given a string s, print YES if it is a palindrome and NO otherwise!!",
    }
    report = decontam.decontaminate(train, reference)
    assert report.dropped == ("drop",)
    assert report.kept == 1


def test_decontamination_is_a_no_op_without_a_reference():
    report = decontam.decontaminate({"a": "text"}, [])
    assert report.n_dropped == 0 and json.loads(report.to_json())["kept"] == 1


def test_shingles_and_jaccard_are_formatting_insensitive():
    a = decontam.shingles("hello   world")
    b = decontam.shingles("hello world")
    assert decontam.jaccard(a, b) == 1.0


def test_exact_duplicate_groups_finds_only_real_groups():
    groups = decontam.exact_duplicate_groups({"a": "x y", "b": "x  y", "c": "z"})
    assert list(groups.values()) == [["a", "b"]]
