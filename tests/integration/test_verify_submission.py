"""`acis eval verify-submission` v0 (docs/spec/03 §4).

The checker is only useful if it fails for the right reasons, so each test breaks exactly one thing in an otherwise
valid run directory and asserts that exactly that check flips. A skipped check is never a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acis.appsdata.sources import APPS
from acis.eval import ledger
from acis.eval.runfile import write_run
from acis.eval.verify import (
    FAIL,
    MANIFEST_JSON,
    MODE_A_JSON,
    MODE_B_JSON,
    PASS,
    RESULT_JSON,
    SKIP,
    verify_submission,
    write_checksums,
)
from acis.rank.compose import rank_derived_scores

GOLD = "d3"
QRELS = {f"q{i}": {GOLD: 1} for i in range(5)}


def _run() -> dict[str, dict[str, float]]:
    ids = [GOLD] + [f"x{i:03d}" for i in range(19)]
    return {qid: rank_derived_scores(ids, 20) for qid in QRELS}


def make_run_dir(tmp_path: Path, **overrides) -> Path:
    """A run directory that passes every check, so a test can break exactly one thing."""
    from acis.eval.metrics import score_run

    run = _run()
    scores = score_run(QRELS, run)
    directory = tmp_path / "rc0"
    directory.mkdir(parents=True, exist_ok=True)
    write_run(run, directory, top_k=20)

    result = {
        "task_name": "AppsRetrieval",
        "dataset_revision": overrides.get("dataset_revision", APPS.revision),
        "mteb_version": overrides.get("mteb_version", "2.21.0"),
        "evaluation_time": overrides.get("evaluation_time", 4242.0),
        "model_revision": overrides.get("model_revision", "abcdef123456+0123456789ab"),
        "scores": {"test": [dict(scores)]},
    }
    if "ndcg_at_10" in overrides:
        result["scores"]["test"][0]["ndcg_at_10"] = overrides["ndcg_at_10"]
    (directory / RESULT_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in (MODE_A_JSON, MODE_B_JSON):
        (directory / name).write_text(json.dumps(result, indent=2), encoding="utf-8")

    manifest = {
        "adapter_invocations": overrides.get("adapter_invocations", 1),
        "encode_calls": overrides.get("encode_calls", 0),
        "fallbacks": overrides.get("fallbacks", 0),
        "agent_calls": overrides.get("agent_calls", 0),
        "strict": overrides.get("strict", True),
        "harness_dry_run": overrides.get("harness_dry_run", False),
        "test_touch_count": overrides.get("test_touch_count", 0),
        "mode": overrides.get("mode", "A"),
        "config_hash": "0" * 64,
        "model_revision": overrides.get("manifest_model_revision", "abcdef123456+0123456789ab"),
        "split": "test",
    }
    (directory / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_checksums(directory)
    return directory


def status_of(report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


# -- the happy path ---------------------------------------------------------------------------------------------
@pytest.fixture
def booked_ledger(tmp_path, monkeypatch):
    """A ledger holding this run's row, as a real official run would have written before verification."""
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    return tmp_path / "ledger.jsonl"


def test_a_well_formed_run_directory_passes(tmp_path, booked_ledger):
    directory = make_run_dir(tmp_path)
    ledger.append({"kind": "rc", "test_touch_count": 0, "run_dir": str(directory)})
    write_checksums(directory)
    report = verify_submission(directory, qrels=QRELS)
    assert report.passed, report.render()
    assert status_of(report, "run.trec re-scores to the JSON") == PASS
    assert status_of(report, "both mode JSONs present") == PASS
    assert status_of(report, "checksums valid") == PASS


def test_rescoring_is_skipped_not_passed_without_qrels(tmp_path):
    """The held-out labels are read by `acis.eval.final` alone, so the check declares itself skipped."""
    report = verify_submission(make_run_dir(tmp_path))
    assert status_of(report, "run.trec re-scores to the JSON") == SKIP
    assert report.n_skipped >= 1


# -- one broken thing at a time ------------------------------------------------------------------------------------
def test_a_missing_run_directory_fails_immediately(tmp_path):
    report = verify_submission(tmp_path / "nope")
    assert not report.passed and report.checks[0].status == FAIL


def test_a_wrong_dataset_revision_fails(tmp_path):
    report = verify_submission(make_run_dir(tmp_path, dataset_revision="deadbeef"), qrels=QRELS)
    assert status_of(report, "dataset_revision is pinned") == FAIL
    assert not report.passed


def test_a_zero_evaluation_time_fails(tmp_path):
    """A run that took no time did not run (D17)."""
    report = verify_submission(make_run_dir(tmp_path, evaluation_time=0.0), qrels=QRELS)
    assert status_of(report, "evaluation_time above floor") == FAIL


DISPATCH_CHECK = "our code ran, on the dispatch path the mode requires"


@pytest.mark.parametrize(
    "field,value,check",
    [
        ("adapter_invocations", 0, DISPATCH_CHECK),
        ("adapter_invocations", 2, DISPATCH_CHECK),
        ("encode_calls", 3, DISPATCH_CHECK),
        ("fallbacks", 3, "manifest.fallbacks == 0"),
        ("agent_calls", 1, "manifest.agent_calls == 0"),
        ("strict", False, "manifest.strict is true"),
        ("harness_dry_run", True, "not a harness dry run"),
    ],
)
def test_manifest_expectations(tmp_path, field, value, check):
    report = verify_submission(make_run_dir(tmp_path, **{field: value}), qrels=QRELS)
    assert status_of(report, check) == FAIL
    assert not report.passed


def test_mode_b_is_verified_by_its_own_dispatch_evidence(tmp_path):
    """RC0 is Mode B (D8): there `encode()` must have run and `search()` must not — the mirror of Mode A."""
    good = make_run_dir(tmp_path / "ok", mode="B", adapter_invocations=0, encode_calls=2)
    assert status_of(verify_submission(good, qrels=QRELS), DISPATCH_CHECK) == PASS
    bad = make_run_dir(tmp_path / "bad", mode="B", adapter_invocations=0, encode_calls=0)
    assert status_of(verify_submission(bad, qrels=QRELS), DISPATCH_CHECK) == FAIL


def test_a_model_revision_mismatch_fails(tmp_path):
    report = verify_submission(make_run_dir(tmp_path, manifest_model_revision="other+revision"), qrels=QRELS)
    assert status_of(report, "model revision matches manifest") == FAIL


def test_a_json_that_disagrees_with_the_run_file_fails(tmp_path):
    """Parity P3: if the JSON's NDCG cannot be re-derived from `run.trec`, the number is not evidence."""
    report = verify_submission(make_run_dir(tmp_path, ndcg_at_10=0.5), qrels=QRELS)
    assert status_of(report, "run.trec re-scores to the JSON") == FAIL


def test_a_tampered_artifact_breaks_the_checksums(tmp_path):
    directory = make_run_dir(tmp_path)
    (directory / "run.trec").write_text("q0 Q0 d3 1 1.0 acis\n", encoding="utf-8")
    report = verify_submission(directory, qrels=QRELS)
    assert status_of(report, "checksums valid") == FAIL


def test_a_missing_mode_json_fails(tmp_path):
    directory = make_run_dir(tmp_path)
    (directory / MODE_B_JSON).unlink()
    write_checksums(directory)
    report = verify_submission(directory, qrels=QRELS)
    assert status_of(report, "both mode JSONs present") == FAIL


def test_the_ledger_touch_count_must_match_this_runs_row(tmp_path, monkeypatch):
    """A manifest declaring fewer touches than it spent must fail, whatever the ledger total happens to be."""
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    directory = make_run_dir(tmp_path, test_touch_count=2)
    ledger.append({"kind": "rc", "test_touch_count": 1, "run_dir": str(directory)})
    write_checksums(directory)
    report = verify_submission(directory, qrels=QRELS)
    assert status_of(report, "held-out touches agree with the ledger") == FAIL


def test_a_manifest_without_a_touch_count_fails_rather_than_skipping(tmp_path):
    directory = make_run_dir(tmp_path)
    manifest = json.loads((directory / MANIFEST_JSON).read_text(encoding="utf-8"))
    del manifest["test_touch_count"]
    (directory / MANIFEST_JSON).write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(directory)
    report = verify_submission(directory, qrels=QRELS)
    assert status_of(report, "held-out touches agree with the ledger") == FAIL


def test_report_renders_and_serialises(tmp_path, booked_ledger):
    directory = make_run_dir(tmp_path)
    ledger.append({"kind": "rc", "test_touch_count": 0, "run_dir": str(directory)})
    write_checksums(directory)
    report = verify_submission(directory, qrels=QRELS)
    text = report.render()
    assert "verdict: PASS" in text
    payload = report.to_dict()
    assert payload["verdict"] == PASS and len(payload["checks"]) == len(report.checks)
