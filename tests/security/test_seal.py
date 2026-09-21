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
    leaks = [
        p.relative_to(acis_root()).as_posix()
        for p in acis_root().rglob("*")
        if p.is_file() and ".git/" not in p.as_posix() and is_sealed_file(p.name)
    ]
    assert leaks == []


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


@pytest.mark.parametrize("name", ["query_id", "qid", "doc_id", "corpus_id", "external_id_hash"])
def test_ids_may_never_become_features(name):
    """INV-4."""
    with pytest.raises(SealedDataAccess, match="INV-4"):
        guard.assert_ids_not_features(["cos", name])


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

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(SealedDataAccess, match="HF_HOME"):
        assert_official_environment(freeze_config({"run": {"strict": True}}))


def test_official_run_requires_strict_and_cpu(monkeypatch):
    from acis.core.config import freeze_config
    from acis.core.errors import StrictViolation
    from acis.eval.official import assert_official_environment

    # A sealed-looking HF_HOME, so the environment check passes and only the strict/device checks can fire.
    monkeypatch.setenv("HF_HOME", "/home/someone/.acis-sealed/hf")
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
    strict_cfg = model.cfg.with_overrides(**{"run.strict": True})
    object.__setattr__(model, "cfg", strict_cfg)
    with pytest.raises(StrictViolation, match="stand-in"):
        _ = model.engine
