"""End-to-end dev harness: loaders, the dev task through real `mteb.evaluate`, the split lock, the ladder.

Everything here needs the allow-listed dataset assets (`make fetch`) and is skipped without them. Nothing here
reads a held-out label: the dev task is built from the dev split and the full corpus (docs/spec/03 §5).
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.dataset

from acis.appsdata import apps, fetch  # noqa: E402
from acis.eval import ladder, splits  # noqa: E402
from acis.eval.dev_task import DEV_TASK_NAME, build_split_data, make_dev_task  # noqa: E402
from acis.eval.metrics import score_run  # noqa: E402

SMOKE_QUERIES = 40


@pytest.fixture(scope="module")
def dev_ids() -> list[str]:
    return list(apps.dev_query_ids())[:SMOKE_QUERIES]


# -- loaders (D-4, D-7) ---------------------------------------------------------------------------------------
def test_corpus_and_labels_match_the_audited_shape():
    assert len(apps.load_documents()) == 8765
    assert len(apps.load_qrels()) == 5000
    assert all(len(rels) == 1 for rels in apps.load_qrels().values())
    assert {d for rels in apps.load_qrels().values() for d in rels} <= set(apps.corpus_ids())


def test_corpus_order_is_preserved():
    ids = apps.corpus_ids()
    assert ids[0] == "d1" and len(set(ids)) == len(ids)
    assert [s.handle for s in apps.load_corpus()] == list(ids)


def test_fetched_assets_are_checksum_verifiable():
    assert fetch.verify_manifest() == []


def test_partition_metadata_does_not_leak_out_of_the_package():
    """D-5: only the audit may see which partition a document belongs to."""
    snippets = apps.load_corpus()
    assert all(not hasattr(s, "partition") for s in snippets)
    audit = apps.dataset_audit()
    assert set(audit["partitions_metadata_only"]) == {"train", "test"}


# -- split lock -------------------------------------------------------------------------------------------------
def test_split_lock_matches_the_data():
    assert splits.verify_lock() == []


def test_split_lock_records_every_decision_set():
    locked = splits.load_lock()
    assert locked["fold_key"] == "sha256_query_text"
    assert locked["sets"]["dev_queries"]["n"] == 5000
    assert locked["sets"]["oof_pool"]["n"] == 5000
    assert locked["sets"]["holdout_queries"]["n"] == 3765
    assert sum(locked["sets"][f"fold_{i}"]["n"] for i in range(5)) == 5000
    assert locked["sets"]["dev_h"]["sha256"] == locked["sets"]["fold_4"]["sha256"]


def test_folds_are_disjoint_and_text_derived():
    members = splits.fold_members()
    seen: set[str] = set()
    for ids in members.values():
        assert not seen & set(ids)
        seen |= set(ids)
    assert len(seen) == 5000
    # the same text always lands in the same fold, whatever the query is called
    from acis.core.hashing import fold_of

    queries = apps.load_queries()
    for qid, fold in list(apps.fold_assignments().items())[:50]:
        assert fold_of(queries[qid], 5) == fold


def test_writing_a_different_lock_is_refused():
    """Changing the splits invalidates every recorded number, so it needs an ADR, not a rerun."""
    from acis.core.errors import InvalidInput

    tampered = splits.build_lock(folds=4, dev_h_fold=3)
    assert tampered.split_lock_hash != splits.split_lock_hash()
    with pytest.raises(InvalidInput, match="ADR"):
        splits.write_lock(tampered)


def test_build_lock_rejects_an_impossible_fold_configuration():
    from acis.core.errors import InvalidInput

    with pytest.raises(InvalidInput, match="existing fold"):
        splits.build_lock(folds=4, dev_h_fold=4)
    with pytest.raises(InvalidInput, match="two folds"):
        splits.build_lock(folds=1, dev_h_fold=0)


# -- the dev task through real mteb ------------------------------------------------------------------------------
def test_dev_task_loads_only_the_dev_split(dev_ids):
    task = make_dev_task(dev_ids)
    task.load_data()
    assert list(task.dataset["default"]) == ["train"]
    split = task.dataset["default"]["train"]
    assert len(split["queries"]) == len(dev_ids)
    assert len(split["corpus"]) == 8765  # unlabelled documents stay in as distractors


def test_dev_task_metadata_points_at_the_pinned_dataset():
    task = make_dev_task([])
    assert task.metadata.name == DEV_TASK_NAME
    from acis.appsdata.sources import APPS

    assert task.metadata.dataset["revision"] == APPS.revision
    assert task.metadata.main_score == "ndcg_at_10"


@pytest.mark.slow
def test_mode_a_runs_through_mteb_on_the_dev_task(dev_ids, tmp_path, monkeypatch):
    """The dev path exercises the same adapter, dispatch and metric code as the official run."""
    import mteb

    monkeypatch.setenv("ACIS_MODE", "A")
    monkeypatch.setenv("ACIS_CONFIG", "configs/dev.yaml")
    monkeypatch.setenv("ACIS_RUN_DIR", str(tmp_path))
    from acis.mteb_adapter import PrePostPipelineEncoder

    model = PrePostPipelineEncoder()
    result = mteb.evaluate(
        model,
        [make_dev_task(dev_ids)],
        encode_kwargs={"batch_size": 64},
        cache=None,
        overwrite_strategy="always",
        prediction_folder=str(tmp_path / "predictions"),
        show_progress_bar=False,
    )
    task_result = list(result.task_results)[0]
    scores = task_result.scores["train"][0]
    assert 0.0 <= scores["ndcg_at_10"] <= 1.0
    assert model.adapter_invocations == 1 and model.encode_calls == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == DEV_TASK_NAME and manifest["split"] == "train"
    assert manifest["agent_calls"] == 0 and manifest["fallbacks"] == 0


@pytest.mark.slow
def test_dev_task_split_data_is_consistent(dev_ids):
    split = build_split_data(dev_ids)
    assert set(split["relevant_docs"]) == set(dev_ids)
    assert split["top_ranked"] is None
    corpus_ids = set(split["corpus"]["id"])
    assert all(next(iter(rels)) in corpus_ids for rels in split["relevant_docs"].values())


# -- the ladder ---------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("rung", ["random", "oracle", "bm25_acis"])
def test_ladder_rungs_produce_scorable_runs(rung, dev_ids):
    result = ladder.run_rung(rung, query_ids=dev_ids)
    assert result.n_queries == len(dev_ids)
    assert result.n_docs == 8765
    assert set(result.run) == set(dev_ids)
    if rung == "oracle":
        assert result.metrics["ndcg_at_10"] == pytest.approx(1.0, abs=1e-12)
    if rung == "random":
        assert result.metrics["ndcg_at_10"] < 0.05


@pytest.mark.slow
def test_ladder_results_are_reproducible(dev_ids):
    first = ladder.run_rung("bm25_acis", query_ids=dev_ids)
    second = ladder.run_rung("bm25_acis", query_ids=dev_ids)
    assert first.run == second.run
    assert first.metrics == second.metrics


@pytest.mark.slow
def test_stub_dense_rung_runs_the_whole_mode_a_pipeline(dev_ids):
    result = ladder.run_rung("stub_dense", query_ids=dev_ids[:10], top_k=100)
    assert all(len(v) == 100 for v in result.run.values())
    scores = score_run(apps.load_qrels(), result.run, (10,))
    assert 0.0 <= scores["ndcg_at_10"] <= 1.0
