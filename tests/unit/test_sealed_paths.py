"""The sealed cache's *paths*, checked without a network and without touching a single label.

The original defect here was an off-by-one-directory: files were written to `<sealed>/hf` while the offline run
resolved `<sealed>/hf/hub`, so the official pass would have failed at data loading. That is a pure path question,
and a path test would have caught it. "The sandbox has no network" explains a missing end-to-end run; it does not
excuse a missing unit test.

The second defect in the same area was an agreement failure: `make rc-official` hard-coded the seal location while
`sealed_root()` honoured `ACIS_SEALED_HOME`, so an owner who moved the seal had their own run refused. Nothing
compared the Makefile against the code, so these tests do.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from acis.appsdata import fetch
from acis.core.paths import acis_root, sealed_root
from acis.eval.guard import official_run_env_ok


@pytest.fixture
def moved_seal(tmp_path, monkeypatch):
    """An owner who put the seal somewhere other than the default — a configuration that must keep working."""
    seal = tmp_path / "elsewhere" / "seal"
    seal.mkdir(parents=True)
    monkeypatch.setenv("ACIS_SEALED_HOME", str(seal))
    return seal


# -- the two caches the official run needs ---------------------------------------------------------------------
def test_the_hub_cache_is_where_hf_home_resolves_it(moved_seal):
    """`HF_HOME=<seal>/hf` makes huggingface_hub read `<seal>/hf/hub`; downloads must land there."""
    assert fetch.sealed_hub_cache() == moved_seal / "hf" / "hub"


def test_the_datasets_cache_is_where_hf_datasets_cache_points(moved_seal):
    assert fetch.sealed_datasets_cache() == moved_seal / "hf" / "datasets"


def test_both_caches_live_under_the_seal(moved_seal):
    for path in (fetch.sealed_hub_cache(), fetch.sealed_datasets_cache()):
        assert path.is_relative_to(sealed_root())


def test_huggingface_hub_resolves_hf_home_the_way_we_assume(moved_seal, monkeypatch):
    """Pins the assumption itself against the installed library rather than against memory."""
    monkeypatch.setenv("HF_HOME", str(moved_seal / "hf"))
    import importlib

    import huggingface_hub.constants as hub_constants

    importlib.reload(hub_constants)
    try:
        assert Path(hub_constants.HF_HUB_CACHE) == fetch.sealed_hub_cache()
    finally:
        monkeypatch.undo()
        importlib.reload(hub_constants)


# -- the Makefile and the code must agree ------------------------------------------------------------------------
def _rc_official_env() -> dict[str, str]:
    """The environment `make rc-official` exports, read out of the Makefile itself."""
    text = (acis_root() / "Makefile").read_text(encoding="utf-8")
    block = text.split("rc-official:", 1)[1].split("\n\n", 1)[0]
    return dict(re.findall(r"\b(HF_HOME|HF_DATASETS_CACHE|HF_HUB_OFFLINE|HF_DATASETS_OFFLINE)=(\S+)", block))


def test_rc_official_honours_a_moved_seal():
    """A hard-coded `$HOME/.acis-sealed` refused an owner who had moved the seal, using our own guard."""
    env = _rc_official_env()
    assert "ACIS_SEALED_HOME" in env.get("HF_HOME", "") or "$SEAL" in env.get("HF_HOME", "")


def test_rc_official_points_at_both_caches_and_goes_offline():
    env = _rc_official_env()
    assert env.get("HF_HOME", "").endswith("/hf")
    assert env.get("HF_DATASETS_CACHE", "").endswith("/hf/datasets")
    assert env.get("HF_HUB_OFFLINE") == "1" and env.get("HF_DATASETS_OFFLINE") == "1"


def test_the_environment_rc_official_exports_satisfies_our_own_guard(moved_seal, monkeypatch):
    """End of the loop: what the Makefile sets must be what `official_run_env_ok()` accepts."""
    monkeypatch.setenv("HF_HOME", str(moved_seal / "hf"))
    ok, home = official_run_env_ok()
    assert ok, home

    monkeypatch.setenv("HF_HOME", str(moved_seal.parent / "not-the-seal" / "hf"))
    assert not official_run_env_ok()[0]


# -- warming the datasets cache ------------------------------------------------------------------------------------
def test_warming_sets_the_sealed_environment_and_restores_it(moved_seal, monkeypatch):
    """The warm step must point `datasets` at the sealed tree, and must not leak that setting afterwards."""
    seen: dict[str, object] = {}

    class FakeDatasets:
        @staticmethod
        def get_dataset_config_names(repo: str, revision: str | None = None) -> list[str]:
            seen["hf_home"] = os.environ.get("HF_HOME")
            seen["datasets_cache"] = os.environ.get("HF_DATASETS_CACHE")
            seen["repo"], seen["revision"] = repo, revision
            return ["corpus", "queries", "default"]

        @staticmethod
        def load_dataset(repo: str, config: str, revision: str | None = None) -> str:
            seen.setdefault("configs", []).append(config)  # type: ignore[union-attr]
            return config

    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "datasets", FakeDatasets)

    configs = fetch.warm_sealed_datasets_cache()

    assert configs == ["corpus", "queries", "default"]
    assert seen["configs"] == ["corpus", "queries", "default"]
    assert seen["hf_home"] == str(moved_seal / "hf")
    assert seen["datasets_cache"] == str(fetch.sealed_datasets_cache())
    # restored, so a warm step cannot silently redirect the rest of the process at the seal
    assert os.environ["HF_HOME"] == "/somewhere/else"
    assert "HF_DATASETS_CACHE" not in os.environ


def test_warming_materialises_every_config_the_loader_asks_for(moved_seal, monkeypatch):
    """mteb's retrieval loader requests `corpus`, `queries` and `default`/`qrels` as separate configs.

    Warming only the one it happens to ask for first would fail offline on the next call.
    """
    requested: list[str] = []

    class FakeDatasets:
        @staticmethod
        def get_dataset_config_names(repo: str, revision: str | None = None) -> list[str]:
            return ["corpus", "queries", "qrels"]

        @staticmethod
        def load_dataset(repo: str, config: str, revision: str | None = None) -> str:
            requested.append(config)
            return config

    monkeypatch.setitem(__import__("sys").modules, "datasets", FakeDatasets)
    fetch.warm_sealed_datasets_cache()
    assert requested == ["corpus", "queries", "qrels"]


# -- the cache we build ourselves must be one the detector can see -------------------------------------------------
def test_detection_sees_a_datasets_cache_layout():
    """`warm_sealed_datasets_cache` deliberately creates this layout; a detector blind to it is blind to us."""
    from acis.appsdata.sources import SEALED_SPLIT, is_sealed_file

    for name in (
        f"coir-retrieval___apps/default/0.0.0/abc123/apps-{SEALED_SPLIT}.arrow",
        f"apps-{SEALED_SPLIT}.arrow",
        f"datasets/coir-retrieval___apps/qrels/0.0.0/deadbeef/apps-{SEALED_SPLIT}-00000-of-00001.arrow",
    ):
        assert is_sealed_file(name), name


def test_detection_still_ignores_ordinary_arrow_files():
    from acis.appsdata.sources import DEV_SPLIT, is_sealed_file

    for name in (
        f"coir-retrieval___apps/corpus/0.0.0/abc123/apps-{DEV_SPLIT}.arrow",
        "runs/cache/embeddings.arrow",
        "coir-retrieval___apps/corpus/0.0.0/abc123/apps-corpus.arrow",
    ):
        assert not is_sealed_file(name), name
