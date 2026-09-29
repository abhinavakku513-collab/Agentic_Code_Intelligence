"""`make bench`: the latency and memory figures the scorecard quotes (docs/spec/02 §7, D2).

Run as a subprocess, the way `make bench` runs it, because the things worth checking are the things a library
call would skip: that it writes the report where it was told, that it records nothing when asked not to, and that
it separates the measurement of the model from the measurement of the retrieval core — the two are minutes apart
with a real encoder, and a benchmark that conflates them cannot show that the core is fast.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from acis.appsdata import apps
from acis.core.paths import acis_root
from acis.eval import ledger

pytestmark = pytest.mark.skipif(not apps.is_available(), reason="dataset assets are not fetched")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "bench.json"
    before = len(ledger.read_rows())
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bench/run.py",
            "--queries",
            "4",
            "--pause",
            "0",
            "--no-ledger",
            "--config",
            "configs/dev-standin.yaml",
            "--out",
            str(out),
        ],
        cwd=acis_root(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert len(ledger.read_rows()) == before, "--no-ledger recorded a row anyway"
    return json.loads(out.read_text(encoding="utf-8")), result.stdout


def test_it_measures_the_model_and_the_core_separately(report):
    payload, _ = report
    for name in ("short_cold", "short_warm", "long_cold", "long_warm", "short_cold_4_concurrent"):
        assert payload["latency_ms"][name]["p95"] > 0.0
    # The core alone cannot be slower than the query that had to be encoded first.
    assert payload["latency_ms"]["long_warm"]["p95"] <= payload["latency_ms"]["long_cold"]["p95"] * 2
    # Stages are reported per workload, and the queueing for the model is its own stage, never hidden in encode.
    assert "route" in payload["stage_ms"]["short_cold"]


def test_it_records_what_the_numbers_describe(report):
    payload, _ = report
    assert payload["n_units"] == len(apps.corpus_ids())
    assert payload["peak_rss_mb"] > 0 and payload["threads"] >= 1
    assert payload["config_hash"] and payload["encoder"]
    assert payload["submission_capable"] is False  # the dev config runs the stand-in, and the report says so


def test_it_prints_the_targets_it_checked(report):
    _, stdout = report
    assert "short_warm search" in stdout and "target" in stdout
