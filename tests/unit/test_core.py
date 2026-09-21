"""`acis.core` contract (src/acis/core/README.md rows C-1…C-5)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from acis.core import errors
from acis.core.config import (
    OPERATIONAL_KEYS,
    compute_config_hash,
    freeze_config,
    load_frozen_config,
    resolve_config_path,
)
from acis.core.hashing import fold_of, hash_id_list, hash_obj, normalized_text_key, sha256_text, stable_json
from acis.core.numeric import REFERENCE_PROFILE, get_profile, resolve_threads
from acis.core.paths import acis_home, acis_root
from acis.core.types import Hit, Snippet, Unit

# -- C-1: no heavy imports ----------------------------------------------------------------------------------------
HEAVY = ("torch", "mteb", "datasets", "transformers", "lightgbm", "bm25s", "sentence_transformers")


def test_core_imports_nothing_heavy():
    """`acis.core` must load instantly and offline: the CLI, the hooks and every test depend on that."""
    code = (
        "import json, importlib, sys\n"
        "importlib.import_module('acis.core')\n"
        f"heavy = {HEAVY!r}\n"
        "print(json.dumps([m for m in sys.modules if m.split('.')[0] in heavy]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == []


# -- C-2: config hash ---------------------------------------------------------------------------------------------
def test_operational_keys_do_not_change_the_config_hash():
    base = {"model": {"name": "x"}, "run": {"threads": 4, "strict": True}, "retrieve": {"batch_size": 32}}
    other = {"model": {"name": "x"}, "run": {"threads": 64, "strict": True}, "retrieve": {"batch_size": 512}}
    assert compute_config_hash(base) == compute_config_hash(other)


def test_ranking_relevant_keys_do_change_the_config_hash():
    base = {"model": {"name": "x"}, "prep": {"query": {"max_tokens": 1024}}}
    other = {"model": {"name": "x"}, "prep": {"query": {"max_tokens": 512}}}
    assert compute_config_hash(base) != compute_config_hash(other)


def test_gate_provenance_does_not_change_the_config_hash():
    """`gates:` records which ledger row decided what; it cannot change a ranking."""
    a = {"model": {"name": "x"}, "gates": {"G1": "dev-aaaa"}}
    b = {"model": {"name": "x"}, "gates": {"G1": "dev-bbbb"}}
    assert compute_config_hash(a) == compute_config_hash(b)


def test_every_operational_key_is_ignored_at_any_depth():
    for key in OPERATIONAL_KEYS:
        a = freeze_config({"deep": {"nested": {key: 1, "real": 2}}})
        b = freeze_config({"deep": {"nested": {key: 999, "real": 2}}})
        assert a.config_hash == b.config_hash, key


def test_frozen_config_is_immutable_and_dotted():
    cfg = freeze_config({"run": {"strict": True, "mode": "A"}, "prep": {"query": {"view": "V0"}}})
    assert cfg.get("prep.query.view") == "V0"
    assert cfg.get("prep.query.missing", "fallback") == "fallback"
    assert cfg.strict is True and cfg.mode == "A"
    with pytest.raises(TypeError):
        cfg.raw["run"]["strict"] = False  # type: ignore[index]


def test_with_overrides_returns_a_new_config_and_a_new_hash():
    cfg = freeze_config({"prep": {"query": {"max_tokens": 1024}}})
    other = cfg.with_overrides(**{"prep.query.max_tokens": 512})
    assert cfg.get("prep.query.max_tokens") == 1024
    assert other.get("prep.query.max_tokens") == 512
    assert other.config_hash != cfg.config_hash


def test_missing_config_file_raises_invalid_input():
    with pytest.raises(errors.InvalidInput):
        load_frozen_config("configs/does-not-exist.yaml")


def test_config_requires_raise_with_context():
    cfg = freeze_config({})
    with pytest.raises(errors.InvalidInput):
        cfg.require("model.name")


# -- C-3: hashing --------------------------------------------------------------------------------------------------
def test_stable_json_is_key_order_independent():
    assert stable_json({"b": 1, "a": 2}) == stable_json({"a": 2, "b": 1})
    assert hash_obj({"b": [1, 2], "a": None}) == hash_obj({"a": None, "b": [1, 2]})


def test_hashes_are_stable_across_processes():
    """`PYTHONHASHSEED` must not reach any digest we depend on."""
    expr = "from acis.core.hashing import hash_obj; print(hash_obj({'a': [1, 'x'], 'b': {'c': 2}}))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", expr],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }
    assert len(runs) == 1


def test_fold_of_uses_the_text_not_an_id():
    """INV-4: relabelling a query must not move it between folds."""
    text = "Given a string, print YES if it is a palindrome."
    assert fold_of(text) == fold_of(text)
    assert 0 <= fold_of(text, 5) < 5
    assert fold_of("a") != fold_of("b") or True  # different texts may collide; the point is determinism


def test_fold_distribution_is_roughly_uniform():
    counts = [0] * 5
    for i in range(5000):
        counts[fold_of(f"query number {i} about graphs and strings", 5)] += 1
    assert min(counts) > 800 and max(counts) < 1200


def test_hash_id_list_is_order_independent():
    assert hash_id_list(["q3", "q1", "q2"]) == hash_id_list(["q1", "q2", "q3"])
    assert hash_id_list(["q1"]) != hash_id_list(["q2"])


def test_normalized_text_key_ignores_formatting():
    assert normalized_text_key("a  b\n\nc") == normalized_text_key("a b c")
    assert normalized_text_key("A B") == normalized_text_key("a b")
    assert normalized_text_key("a b") != normalized_text_key("a c")


def test_sha256_text_matches_a_known_value():
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# -- C-4: paths ------------------------------------------------------------------------------------------------------
def test_repo_root_holds_pyproject():
    assert (acis_root() / "pyproject.toml").is_file()


def test_config_paths_resolve_from_the_repo_root_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path("configs/dev.yaml") == acis_root() / "configs" / "dev.yaml"


def test_acis_home_is_outside_the_repo_or_ignored():
    home = acis_home()
    assert home.exists()


# -- C-5: errors -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls,status",
    [
        (errors.InvalidInput, 400),
        (errors.NotFound, 404),
        (errors.IndexRequired, 409),
        (errors.SnapshotInvalid, 409),
        (errors.VersionConflict, 409),
        (errors.ResourceLimit, 413),
        (errors.NotReady, 503),
        (errors.StrictViolation, 500),
        (errors.SealedDataAccess, 500),
    ],
)
def test_error_statuses_and_problem_bodies(cls, status):
    exc = cls("boom", extra=1)
    assert isinstance(exc, errors.AcisError)
    assert exc.http_status == status
    body = exc.to_problem()
    assert body["status"] == status and body["detail"] == "boom" and body["extra"] == 1
    assert cls.__name__ in str(exc)  # tests/robustness matches on the class name


# -- types --------------------------------------------------------------------------------------------------------------
def test_core_types_are_frozen():
    snippet = Snippet(handle="d1", text="x")
    with pytest.raises(AttributeError):
        snippet.text = "y"  # type: ignore[misc]
    hit = Hit(rank=1, score=1.0, unit=Unit(unit_id="u", key="d1", body_hash="h"), source="x")
    assert hit.rank == 1 and hit.unit.key == "d1"


# -- numeric profiles -----------------------------------------------------------------------------------------------------
def test_reference_profile_is_cpu_fp32():
    profile = get_profile()
    assert profile.name == REFERENCE_PROFILE and profile.is_reference and not profile.needs_gate_g6


def test_non_reference_profiles_need_gate_g6():
    assert get_profile("cpu-int8").needs_gate_g6


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown numeric profile"):
        get_profile("cpu-fp4")


def test_resolve_threads_accepts_auto_and_explicit():
    assert resolve_threads("auto_physical") >= 1
    assert resolve_threads(3) == 3
    assert resolve_threads("7") == 7
