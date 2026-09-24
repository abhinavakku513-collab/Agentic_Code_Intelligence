"""The physical seal and the evaluation-integrity guard (INV-8, D19, docs/spec/03 §5) — gate G0.6.

The seal is a file filter plus a separate directory, not a promise: the held-out labels live in their own file in
the dataset repository, so keeping them out of the dev environment is mechanical. These tests pin that mechanism,
the programmatic guard around it, and the fact that the dev environment is currently clean.
"""

from __future__ import annotations

import json

import pytest

from acis.appsdata import fetch
from acis.appsdata.sources import DEV_ALLOWLIST, DEV_SPLIT, SEALED_SPLIT, is_dev_allowed, is_sealed_file, partition
from acis.core.errors import SealedDataAccess
from acis.core.paths import acis_root
from acis.eval import guard

REPO_LISTING = [
    ".gitattributes",
    "README.md",
    "corpus/corpus-00000-of-00001.parquet",
    "queries/queries-00000-of-00001.parquet",
    f"data/{DEV_SPLIT}-00000-of-00001.parquet",
    f"data/{SEALED_SPLIT}-00000-of-00001.parquet",
]


# -- the allow-list -------------------------------------------------------------------------------------------
def test_the_verified_repository_listing_partitions_as_expected():
    allowed, sealed, ignored = partition(REPO_LISTING)
    assert sorted(allowed) == sorted(
        [
            "README.md",
            "corpus/corpus-00000-of-00001.parquet",
            f"data/{DEV_SPLIT}-00000-of-00001.parquet",
            "queries/queries-00000-of-00001.parquet",
        ]
    )
    assert sealed == [f"data/{SEALED_SPLIT}-00000-of-00001.parquet"]
    assert ignored == [".gitattributes"]


@pytest.mark.parametrize(
    "name",
    [
        f"data/{SEALED_SPLIT}-00000-of-00001.parquet",
        f"qrels_{SEALED_SPLIT}.tsv",
        f"{SEALED_SPLIT}_qrels.jsonl",
        f"some/dir/apps_qrels_{SEALED_SPLIT}.parquet",
    ],
)
def test_sealed_patterns_are_recognised(name):
    assert is_sealed_file(name)
    assert not is_dev_allowed(name)


def test_detection_covers_every_shape_a_real_leak_would_take():
    """The file appears as a repository entry, as an HF-cache blob and as a bare basename."""
    for name in (
        f"data/{SEALED_SPLIT}-00000-of-00001.parquet",
        f"datasets--CoIR-Retrieval--apps/snapshots/f22508f/data/{SEALED_SPLIT}-00000-of-00001.parquet",
        f"/home/u/.acis-sealed/hf/hub/datasets--CoIR-Retrieval--apps/snapshots/abc/data/{SEALED_SPLIT}-0.parquet",
        f"qrels/{SEALED_SPLIT}-00000-of-00001.parquet",
        f"apps_qrels_{SEALED_SPLIT}.tsv",
        f"{SEALED_SPLIT}_qrels.jsonl",
    ):
        assert is_sealed_file(name), name


def test_detection_does_not_fire_on_ordinary_repository_files():
    """A false positive is not harmless: it breaks the working-tree check and invites weakening the pattern.

    Matching arbitrary string suffixes made a source file named after the word `qrels` look like a label file.
    """
    for name in (
        f"data/{DEV_SPLIT}-00000-of-00001.parquet",
        "corpus/corpus-00000-of-00001.parquet",
        "queries/queries-00000-of-00001.parquet",
        "src/acis/eval/verify.py",
        f"tests/unit/{SEALED_SPLIT}_qrels_loader.py",
        f"runs/qrels/latest/{SEALED_SPLIT}.json",
        f"runs/latest/{SEALED_SPLIT}-0-of-1.parquet",
    ):
        assert not is_sealed_file(name), name


def test_sealed_always_beats_the_allow_list():
    """A file matching both lists must never be fetched — the seal is not a tie-break."""
    crafted = f"data/{DEV_SPLIT}-qrels-{SEALED_SPLIT}.parquet"
    assert any(crafted.startswith(p.split("*")[0]) for p in DEV_ALLOWLIST)
    assert is_sealed_file(crafted) and not is_dev_allowed(crafted)


# -- the dev environment is clean (G0.6) ------------------------------------------------------------------------
def test_no_held_out_label_file_inside_acis_home():
    assert fetch.scan_for_sealed() == []
    fetch.assert_seal()


def test_no_held_out_label_file_inside_the_working_tree():
    """Checked on the full relative path, not just the basename, so a nested cache cannot hide one."""
    skip = (".git/", ".venv/", ".mypy_cache/", ".ruff_cache/", ".pytest_cache/")
    leaks = [
        p.relative_to(acis_root()).as_posix()
        for p in acis_root().rglob("*")
        if p.is_file()
        and not any(part in p.as_posix() for part in skip)
        and is_sealed_file(p.relative_to(acis_root()).as_posix())
    ]
    assert leaks == []


def test_caches_live_outside_the_repository():
    """CLAUDE.md §7: data, CAS and caches sit outside git — and third-party labels never sit next to ours."""
    from acis.core.paths import acis_home, reg_home

    root = acis_root().resolve()
    assert not acis_home().resolve().is_relative_to(root)
    assert not reg_home().resolve().is_relative_to(root)


def test_assert_seal_fires_when_a_sealed_file_appears(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / f"{SEALED_SPLIT}-00000-of-00001.parquet").write_bytes(b"x")
    assert fetch.scan_for_sealed(tmp_path)
    with pytest.raises(SealedDataAccess):
        fetch.assert_seal(tmp_path)


# -- loaders refuse anything but the dev split --------------------------------------------------------------------
@pytest.mark.parametrize("split", ["test", "validation", "dev", "TRAIN", ""])
def test_loaders_refuse_every_split_but_the_dev_one(split):
    from acis.appsdata import apps

    if split == DEV_SPLIT:
        pytest.skip("that is the dev split")
    with pytest.raises(SealedDataAccess):
        apps.assert_dev_split(split)
    with pytest.raises(SealedDataAccess):
        guard.assert_dev_split(split, context="test")


def test_local_asset_refuses_a_sealed_repo_file():
    with pytest.raises(SealedDataAccess):
        fetch.local_asset(f"data/{SEALED_SPLIT}-00000-of-00001.parquet")


def test_sealed_fetch_requires_an_explicit_owner_opt_in(monkeypatch):
    monkeypatch.delenv(fetch.SEALED_OPT_IN, raising=False)
    with pytest.raises(SealedDataAccess, match="opt-in"):
        fetch.fetch_sealed()


# -- the programmatic guard ---------------------------------------------------------------------------------------
@pytest.mark.dataset
def test_held_out_query_ids_never_reach_a_fitting_path():
    holdout = sorted(guard.holdout_query_ids())
    assert len(holdout) == 3765
    guard.assert_no_holdout_ids(["q1", "q2"], context="unit test")  # dev ids are fine
    with pytest.raises(SealedDataAccess, match="fitting path"):
        guard.assert_no_holdout_ids([holdout[0]], context="unit test")


@pytest.mark.dataset
def test_assert_dev_pool_rejects_unknown_ids():
    from acis.appsdata import apps

    dev = list(apps.dev_query_ids())[:3]
    assert guard.assert_dev_pool(dev, context="unit test") == sorted(dev)
    with pytest.raises(SealedDataAccess, match="outside the dev pool"):
        guard.assert_dev_pool(["not-a-real-query-id"], context="unit test")


@pytest.mark.parametrize(
    "name", ["query_id", "qid", "doc_id", "docid", "corpus_id", "external_id", "corpus_ordinal", "partition"]
)
def test_ids_and_corpus_position_may_never_become_features(name):
    """INV-4, including corpus position: on this corpus `ordinal < 5000` is an exact train-partition detector."""
    with pytest.raises(SealedDataAccess, match="INV-4"):
        guard.assert_ids_not_features(["cos", name])


@pytest.mark.parametrize("name", ["first_match_position", "term_position_variance", "doc_id_free_feature"])
def test_legitimate_feature_names_are_not_rejected(name):
    """Substring matching rejected within-document positions, which have nothing to do with the corpus ordinal."""
    guard.assert_ids_not_features(["cos", "bm25", name])


def test_feature_names_without_ids_are_accepted():
    guard.assert_ids_not_features(["cos", "bm25", "rank_dense", "len_ratio"])


# -- only one module may read the held-out labels ------------------------------------------------------------------
def test_the_sanctioned_reader_is_a_single_named_module():
    assert guard.SANCTIONED_READER == "acis.eval.final"
    assert not guard.caller_is_sanctioned()


def test_final_refuses_without_the_official_environment(monkeypatch):
    from acis.eval import final

    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(SealedDataAccess, match="physical seal"):
        final.load_holdout_qrels()


def test_official_run_refuses_without_the_sealed_hf_home(monkeypatch, tmp_path):
    from acis.core.config import freeze_config
    from acis.eval.official import assert_official_environment

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    monkeypatch.setenv("ACIS_SEALED_HOME", str(sealed))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "somewhere-else"))
    with pytest.raises(SealedDataAccess, match="HF_HOME"):
        assert_official_environment(freeze_config({"run": {"strict": True}}))


def test_a_machine_without_a_sealed_area_can_still_reproduce(monkeypatch, tmp_path):
    """A judge has no sealed area: `make reproduce` must not be blocked by a seal that does not exist."""
    from acis.core.config import freeze_config
    from acis.eval.official import assert_official_environment

    monkeypatch.setenv("ACIS_SEALED_HOME", str(tmp_path / "no-such-seal"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "ordinary-cache"))
    assert_official_environment(freeze_config({"run": {"strict": True}}), ledgered=False)


def test_official_run_requires_strict_and_cpu(monkeypatch):
    from acis.core.config import freeze_config
    from acis.core.errors import StrictViolation
    from acis.eval.official import assert_official_environment

    # A sealed-looking HF_HOME, so the environment check passes and only the strict/device checks can fire.
    monkeypatch.setenv("HF_HOME", "/home/someone/.acis-sealed/hf")
    monkeypatch.setenv("ACIS_ALLOW_DIRTY_RC", "1")
    with pytest.raises(StrictViolation, match="strict"):
        assert_official_environment(freeze_config({"run": {"strict": False}}))
    with pytest.raises(StrictViolation, match="CPU-only"):
        assert_official_environment(freeze_config({"run": {"strict": True, "device": "cuda"}}))


# -- DEV-H is counted, not free ---------------------------------------------------------------------------------------
def test_dev_h_touches_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path))
    n = guard.record_dev_h_touch("confirming G1", milestone="phase-1")
    assert n >= 1
    assert guard.dev_h_touches_for("phase-1") >= 1
    payload = json.loads((tmp_path / guard.DEV_H_COUNTER).read_text(encoding="utf-8"))
    assert payload[-1]["milestone"] == "phase-1"


# -- the stand-in encoder may never serve a strict run ------------------------------------------------------------------
def test_the_harness_stand_in_encoder_is_refused_in_strict_mode(monkeypatch):
    from acis.core.errors import StrictViolation
    from acis.mteb_adapter import MODE_ENV, PrePostPipelineEncoder

    monkeypatch.setenv(MODE_ENV, "A")
    monkeypatch.setenv("ACIS_CONFIG", "configs/dev.yaml")
    model = PrePostPipelineEncoder()
    # The dev config names the G-M winner now, so the stand-in has to be asked for: this test is about what
    # strict mode does when it is handed one, not about what the default happens to be.
    strict_cfg = model.cfg.with_overrides(**{"run.strict": True, "model.encoder": "hashing"})
    object.__setattr__(model, "cfg", strict_cfg)
    with pytest.raises(StrictViolation, match="stand-in"):
        _ = model.engine


# -- DEV-H is a counted confirmation, not a second decision set ------------------------------------------------
def test_dev_h_requires_a_named_milestone(monkeypatch, tmp_path):
    """Defaulting the milestone would let the first unlabelled run block every later one, while any invented
    name granted a fresh touch. Neither is accounting."""
    import argparse

    from acis.cli.commands import _dev_h_milestone
    from acis.core.errors import InvalidInput

    monkeypatch.setenv("ACIS_HOME", str(tmp_path))
    monkeypatch.delenv("ACIS_MILESTONE", raising=False)
    args = argparse.Namespace(fold=4, milestone="", system="bm25_acis")
    with pytest.raises(InvalidInput, match="naming the milestone"):
        _dev_h_milestone(args, guard)


def test_dev_h_refuses_a_second_use_inside_one_milestone(monkeypatch, tmp_path):
    import argparse

    from acis.cli.commands import _dev_h_milestone
    from acis.core.errors import InvalidInput

    monkeypatch.setenv("ACIS_HOME", str(tmp_path))
    monkeypatch.delenv("ACIS_ALLOW_REPEAT_DEV_H", raising=False)
    args = argparse.Namespace(fold=4, milestone="phase-2-gm", system="bm25_acis")
    assert _dev_h_milestone(args, guard) == "phase-2-gm"
    guard.record_dev_h_touch(reason="first use", milestone="phase-2-gm")
    with pytest.raises(InvalidInput, match="already been used"):
        _dev_h_milestone(args, guard)


def test_a_dev_h_touch_is_chained_into_the_ledger(monkeypatch, tmp_path):
    """A counter in a JSON file outside git can be erased with `rm`; a ledger row cannot, silently."""
    import argparse

    from acis.cli.commands import _record_dev_h_touch
    from acis.eval import ledger

    monkeypatch.setenv("ACIS_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    args = argparse.Namespace(fold=4, milestone="phase-2-gm", system="bm25_acis")
    _record_dev_h_touch(args, guard, "phase-2-gm", run_id="dev-abc123")

    rows = ledger.read_rows()
    assert len(rows) == 1
    assert rows[0].get("decision_set") == "dev_h"
    assert rows[0].get("milestone") == "phase-2-gm"
    assert rows[0].get("confirms_run") == "dev-abc123"
    assert ledger.verify_chain() == []
    assert guard.dev_h_touches_for("phase-2-gm") == 1
