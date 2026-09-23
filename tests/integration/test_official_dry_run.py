"""The official run path, executed end to end on a synthetic task (docs/spec/03 §4).

The real official run happens once, by the owner, against held-out labels. That makes it the single most
dangerous piece of code in the repository: every defect in it is latent until the moment it matters. So the
pipeline takes its task as a parameter and this suite drives the whole thing — config loading, both modes, the
result JSONs, the run files, the manifest, the ledger row, checksums and `verify-submission` — over a fixture
corpus with no held-out data anywhere near it.

Every test here corresponds to a defect that was actually found by reading this path rather than running it:
the model silently loading `configs/dev.yaml`, `run.trec` never being written, Mode B producing no manifest, and
an A+B run booking one held-out touch instead of two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mteb")

from acis.eval import ledger  # noqa: E402
from acis.eval.official import primary_mode_for, run_pipeline  # noqa: E402
from acis.eval.verify import MANIFEST_JSON, MODE_A_JSON, MODE_B_JSON, RESULT_JSON, verify_submission  # noqa: E402

pytest.importorskip("datasets")
from tests.contract.test_adapter_contract import make_toy_task  # type: ignore[import-not-found]  # noqa: E402

OFFICIAL_CONFIG = {
    "run": {"mode": "B", "primary_mode": "B", "strict": True, "device": "cpu", "threads": "auto_physical", "seed": 0},
    "model": {"encoder": "hashing", "dim": 4096, "numeric_profile": "cpu-fp32"},
    "prep": {
        "query": {"version": "q1", "view": "V0", "max_tokens": 1024, "head": 768, "tail": 256},
        "doc": {"version": "d1", "max_tokens": 1024, "head": 768, "tail": 256},
    },
    "lexical": {"k1": 1.5, "b": 0.75, "stemmer": "english"},
    "compose": {"mode_a_scores": "rank_derived"},
}


@pytest.fixture(autouse=True)
def _dry_run_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode correctly refuses the stand-in encoder, so the dry run declares itself as one.

    The flag lands in the manifest and `verify-submission` fails on it — see
    `test_a_dry_run_can_never_pass_verification`.
    """
    from acis.mteb_adapter import DRY_RUN_ENV

    monkeypatch.setenv(DRY_RUN_ENV, "1")
    monkeypatch.setenv("ACIS_ALLOW_DIRTY_RC", "1")


@pytest.fixture
def official_config(tmp_path: Path) -> str:
    """A config that is *not* `configs/dev.yaml`, so a run that ignores the path is caught."""
    import yaml

    path = tmp_path / "official_under_test.yaml"
    path.write_text(yaml.safe_dump(OFFICIAL_CONFIG), encoding="utf-8")
    return str(path)


@pytest.fixture
def isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "ledger_path", lambda: path)
    return path


def _task_factory(name: str) -> Any:
    """**One** task name for both modes, exactly as the real run has.

    Handing each mode its own name would hide the defect this fixture is meant to expose: mteb names the
    predictions file after the task, so two modes writing into one folder overwrite each other.
    """

    def make() -> Any:
        return make_toy_task(name)

    return make


def run(tmp_path: Path, config: str, mode: str, *, ledgered: bool = True, name: str = "DryRun") -> Any:
    return run_pipeline(
        rc="RC0",
        mode=mode,
        config_path=config,
        run_dir=tmp_path / f"run-{mode}",
        ledgered=ledgered,
        task_factory=_task_factory(name),
    )


# -- the whole path, both modes ------------------------------------------------------------------------------
@pytest.mark.slow
def test_single_mode_run_produces_every_artifact(tmp_path, official_config, isolated_ledger):
    result = run(tmp_path, official_config, "B")
    d = result.run_dir

    for name in (RESULT_JSON, MODE_B_JSON, MANIFEST_JSON, "run.trec", "run.csv", "SHA256SUMS", "verify_report.txt"):
        assert (d / name).is_file(), f"{name} was not written"
    assert result.primary_mode == "B"
    assert result.per_mode_seconds == {"B": pytest.approx(result.evaluation_time, abs=1e-6)}


@pytest.mark.slow
def test_the_run_uses_the_config_it_was_given_not_the_dev_default(tmp_path, official_config, isolated_ledger):
    """The model is constructed with the config path; relying on the ambient default ran dev settings."""
    from acis.core.config import load_frozen_config

    result = run(tmp_path, official_config, "B")
    manifest = json.loads((result.run_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    expected = load_frozen_config(official_config).config_hash
    dev = load_frozen_config("configs/dev.yaml").config_hash
    assert manifest["config_hash"] == expected
    assert manifest["config_hash"] != dev


@pytest.mark.slow
def test_run_files_hold_the_ranking_that_was_graded(tmp_path, official_config, isolated_ledger):
    """`run.trec` must be the primary mode's actual ranking, or the P3 re-score is meaningless."""
    from acis.eval.runfile import read_trec

    result = run(tmp_path, official_config, "B")
    run_file = read_trec(result.run_dir / "run.trec")
    assert run_file, "run.trec is empty"
    assert all(len(hits) > 0 for hits in run_file.values())


@pytest.mark.slow
def test_mode_b_writes_a_manifest_even_though_search_never_runs(tmp_path, official_config, isolated_ledger):
    """Mode B is graded by mteb's own wrapper, so nothing in our `search()` can write its manifest."""
    result = run(tmp_path, official_config, "B")
    manifest = json.loads((result.run_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    assert manifest["mode"] == "B"
    assert manifest["adapter_invocations"] == 0
    assert manifest["encode_calls"] > 0


@pytest.mark.slow
def test_mode_a_manifest_records_our_search_running_once(tmp_path, official_config, isolated_ledger):
    result = run(tmp_path, official_config, "A")
    manifest = json.loads((result.run_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    assert manifest["mode"] == "A"
    assert manifest["adapter_invocations"] == 1
    assert manifest["encode_calls"] == 0


@pytest.mark.slow
def test_each_mode_keeps_its_own_predictions(tmp_path, official_config, isolated_ledger):
    """mteb names the predictions file after the task, so an A+B run writing into one folder loses a mode.

    At RC1 with primary A, a judge re-scoring `predictions/` would otherwise get Mode B's rankings against a
    Mode A results JSON.
    """
    result = run(tmp_path, official_config, "AB")
    for mode in ("A", "B"):
        folder = result.run_dir / "predictions" / mode
        assert folder.is_dir(), f"mode {mode} has no predictions folder"
        assert list(folder.rglob("*predictions.json")), f"mode {mode} wrote no predictions"


@pytest.mark.slow
def test_an_ab_run_books_one_held_out_touch_per_mode(tmp_path, official_config, isolated_ledger):
    """CLAUDE.md §4 budgets RC1 as two touches (A+B); booking one would under-report the budget."""
    result = run(tmp_path, official_config, "AB")
    row = ledger.find(result.run_id)
    assert row.get("test_touch_count") == 2
    assert ledger.test_touches_used() == 2
    assert (result.run_dir / MODE_A_JSON).is_file() and (result.run_dir / MODE_B_JSON).is_file()
    assert set(result.per_mode_seconds) == {"A", "B"}


@pytest.mark.slow
def test_the_primary_json_is_the_mode_the_config_declares(tmp_path, official_config, isolated_ledger):
    """D8: RC0 is Mode B. Taking `modes[0]` would have made Mode A primary for an A+B run."""
    result = run(tmp_path, official_config, "AB")
    assert result.primary_mode == "B"
    primary = json.loads((result.run_dir / RESULT_JSON).read_text(encoding="utf-8"))
    mode_b = json.loads((result.run_dir / MODE_B_JSON).read_text(encoding="utf-8"))
    assert primary["scores"] == mode_b["scores"]


@pytest.mark.slow
def test_the_ledger_records_the_primary_modes_own_cold_time(tmp_path, official_config, isolated_ledger):
    """D17: `evaluation_time` is an honest cold time, not the sum of two passes."""
    result = run(tmp_path, official_config, "AB")
    row = ledger.find(result.run_id)
    assert row.get("evaluation_time") == pytest.approx(result.per_mode_seconds["B"], abs=1e-3)
    assert row.get("evaluation_time") < sum(result.per_mode_seconds.values())


@pytest.mark.slow
def test_verification_runs_on_the_produced_directory(tmp_path, official_config, isolated_ledger):
    """The checks that can pass without held-out labels must actually pass on a real run directory."""
    result = run(tmp_path, official_config, "AB")
    report = verify_submission(result.run_dir)
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["run manifest present"] == "PASS"
    assert statuses["our code ran, on the dispatch path the mode requires"] == "PASS"
    assert statuses["the JSONs match the modes that ran"] == "PASS"
    assert statuses["checksums valid"] == "PASS", report.render()
    assert statuses["held-out touches agree with the ledger"] == "PASS", report.render()
    # The toy task is not AppsRetrieval, so the pinned-revision check is expected to fail here.
    assert statuses["dataset_revision is pinned"] == "FAIL"


@pytest.mark.slow
def test_a_dry_run_can_never_pass_verification(tmp_path, official_config, isolated_ledger):
    """The escape hatch that makes this path testable must not be able to produce a submission."""
    result = run(tmp_path, official_config, "B")
    manifest = json.loads((result.run_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    assert manifest["harness_dry_run"] is True
    report = verify_submission(result.run_dir)
    assert not report.passed
    assert next(c.status for c in report.checks if c.name == "not a harness dry run") == "FAIL"


@pytest.mark.slow
def test_the_model_revision_is_recovered_from_the_predictions_artifact(tmp_path, official_config, isolated_ledger):
    """`TaskResult.to_dict()` carries no model revision in mteb 2.21.0, so the check reads the predictions file.

    It is keyed on `mteb_model_meta`, which is what mteb actually writes — reading a plausible-but-wrong key
    turned this integrity check into a permanent SKIP, which is indistinguishable from it not existing.
    """
    from acis.eval.verify import _revision_from_predictions

    result = run(tmp_path, official_config, "B")
    recovered = _revision_from_predictions(result.run_dir)
    manifest = json.loads((result.run_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    assert recovered, "no model revision could be recovered from the predictions artifact"
    assert recovered == manifest["model_revision"]
    report = verify_submission(result.run_dir)
    assert next(c.status for c in report.checks if c.name == "model revision matches manifest") == "PASS"


@pytest.mark.slow
def test_the_shipped_report_states_that_the_checksums_verify(tmp_path, official_config, isolated_ledger):
    """The report inside the run directory is what a judge reads; it must not claim SHA256SUMS is missing."""
    result = run(tmp_path, official_config, "B")
    shipped = (result.run_dir / "verify_report.txt").read_text(encoding="utf-8")
    checksum_line = next(line for line in shipped.splitlines() if "checksums valid" in line)
    assert "[PASS]" in checksum_line, shipped


@pytest.mark.slow
def test_checksums_cover_every_artifact_including_the_verify_report(tmp_path, official_config, isolated_ledger):
    result = run(tmp_path, official_config, "B")
    listed = {
        line.split("  ", 1)[1]
        for line in (result.run_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "verify_report.txt" in listed
    assert {"run.trec", "run.csv", RESULT_JSON, MANIFEST_JSON} <= listed


@pytest.mark.slow
def test_a_cache_verification_books_no_touch(tmp_path, official_config, isolated_ledger):
    result = run(tmp_path, official_config, "B", ledgered=False)
    assert result.run_id == ""
    assert ledger.test_touches_used() == 0
    assert (result.run_dir / RESULT_JSON).is_file()


# -- preconditions, without running anything ----------------------------------------------------------------
def test_primary_mode_selection_rules():
    from acis.core.config import freeze_config

    declared_b = freeze_config({"run": {"mode": "A", "primary_mode": "B"}})
    assert primary_mode_for(declared_b, ["A", "B"]) == "B"
    assert primary_mode_for(freeze_config({"run": {"mode": "A"}}), ["A", "B"]) == "A"
    assert primary_mode_for(freeze_config({"run": {"mode": "B"}}), ["A"]) == "A"  # a single-mode run is its own


def test_the_budget_is_checked_before_the_expensive_pass(tmp_path, monkeypatch, official_config):
    """An exhausted budget must not cost a real held-out touch before it is noticed."""
    from acis.core.config import load_frozen_config
    from acis.core.errors import InvalidInput
    from acis.eval.official import assert_official_environment

    path = tmp_path / "full.jsonl"
    monkeypatch.setattr(ledger, "ledger_path", lambda: path)
    monkeypatch.setenv("ACIS_ALLOW_DIRTY_RC", "1")
    for _ in range(ledger.TEST_TOUCH_BUDGET):
        ledger.append({"kind": "rc", "test_touch_count": 1})

    config = load_frozen_config(official_config).with_overrides(**{"run.strict": True})
    with pytest.raises(InvalidInput, match="budget"):
        assert_official_environment(config, ledgered=True, touches=1)


def test_a_dirty_tree_is_refused_for_a_ledgered_run(tmp_path, monkeypatch, official_config):
    from acis.core.config import load_frozen_config
    from acis.core.errors import StrictViolation
    from acis.eval import official

    monkeypatch.delenv("ACIS_ALLOW_DIRTY_RC", raising=False)
    monkeypatch.setattr(official, "_working_tree_is_dirty", lambda: True)
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "empty.jsonl")
    config = load_frozen_config(official_config).with_overrides(**{"run.strict": True})
    with pytest.raises(StrictViolation, match="uncommitted"):
        official.assert_official_environment(config, ledgered=True, touches=1)
    official.assert_official_environment(config, ledgered=False)  # a reproduction is unaffected
