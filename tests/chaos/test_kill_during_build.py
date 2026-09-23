"""`kill -9` during a build, at every step (docs/spec/04 §5, D11).

The acceptance criterion is not "recovery works" but "recovery works *wherever* it was killed", so the child
process is a real one, killed with `SIGKILL` (which no handler can intercept, and which therefore behaves like
the power going out) at each named point in the build. After each kill the parent runs recovery and asserts the
only two acceptable outcomes: the store serves the previous snapshot, or it serves the new one — never a torn
mixture, never an exception from a half-written file.

The build steps are addressed by name through `ACIS_CRASH_AT`, so adding a step to the builder without adding it
here is caught by `test_every_crash_point_is_exercised`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from acis.core.paths import acis_root

pytestmark = pytest.mark.slow

#: Every point at which the builder can be interrupted. Ordered as the build runs.
CRASH_POINTS = ("before_blobs", "after_blobs", "after_manifest", "before_marker", "after_marker", "before_activate")

CHILD = textwrap.dedent(
    """
    import os, sys
    from acis.store import snapshots
    from acis.store.cas import BlobStore

    repo, label, crash_at = sys.argv[1], sys.argv[2], sys.argv[3]
    units = [snapshots.UnitInput(key=f"u{i}", text=f"def f{i}():\\n    return {i}\\n") for i in range(6)]
    if label == "v2":
        units.append(snapshots.UnitInput(key="new", text="def brand_new():\\n    return 1\\n"))

    os.environ["ACIS_CRASH_AT"] = crash_at
    record = snapshots.build_snapshot(
        repo_id=repo, label=label, units=units, store=BlobStore.open(), config_hash="chaos"
    )
    if crash_at == "before_activate":
        os.kill(os.getpid(), 9)
    snapshots.activate(repo, record.snapshot_id)
    print(record.snapshot_id)
    """
)


def run_child(tmp_path, repo: str, label: str, crash_at: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "ACIS_HOME": str(tmp_path / "home"), "ACIS_ROOT": str(acis_root())}
    return subprocess.run(
        [sys.executable, "-c", CHILD, repo, label, crash_at],
        capture_output=True,
        text=True,
        env=env,
        cwd=acis_root(),
        timeout=300,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A repository with one activated snapshot, built by a child process that was allowed to finish."""
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    done = run_child(tmp_path, "apps", "v1", "never")
    assert done.returncode == 0, done.stderr[-2000:]
    return tmp_path, done.stdout.strip()


@pytest.mark.parametrize("crash_at", CRASH_POINTS)
def test_a_build_killed_at_any_step_leaves_a_serveable_store(store, crash_at):
    from acis.store import snapshots

    tmp_path, first = store
    killed = run_child(tmp_path, "apps", "v2", crash_at)
    assert killed.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL, 137), (
        f"the child was supposed to be killed at {crash_at}, it exited {killed.returncode}: {killed.stderr[-500:]}"
    )

    report = snapshots.recover("apps")
    active = snapshots.active_snapshot_id("apps")
    assert active is not None, f"nothing is serveable after a crash at {crash_at}: {report}"

    # Whatever is active must be complete and readable — that is the whole invariant (INV-9).
    manifest = snapshots.open_snapshot("apps", active)
    units = snapshots.read_units("apps", active)
    assert len(units) == manifest["n_units"]
    assert snapshots.validate("apps", active) == []

    # And it must be one of the two legitimate answers, never a mixture.
    assert active == first or manifest["label"] == "v2"


@pytest.mark.parametrize("crash_at", CRASH_POINTS)
def test_recovery_never_leaves_an_unfinished_build_behind(store, crash_at):
    from acis.store import layout, snapshots

    tmp_path, _ = store
    run_child(tmp_path, "apps", "v2", crash_at)
    snapshots.recover("apps")
    assert list(layout.repo_root("apps").glob(".tmp-*")) == []


def test_an_interrupted_build_never_publishes_a_partial_snapshot(store):
    """Between the kill and recovery — the window a reader could hit — nothing partial is visible."""
    from acis.store import snapshots

    tmp_path, first = store
    run_child(tmp_path, "apps", "v2", "after_manifest")
    # No recovery yet, exactly as a reader that opened the store one millisecond after the crash would find it.
    assert snapshots.active_snapshot_id("apps") == first
    assert snapshots.validate("apps", first) == []


def test_the_content_store_is_reusable_after_a_crash(store):
    """Blobs written before the crash stay valid: that is what makes the retry cheap rather than a full rebuild."""
    from acis.store.cas import BlobStore

    tmp_path, _ = store
    run_child(tmp_path, "apps", "v2", "after_blobs")

    store_after = BlobStore.open()
    digest = store_after.put("def brand_new():\n    return 1\n")
    assert store_after.stats["exists"] == 1  # it was already there, written before the process died
    assert store_after.verify(digest)


def test_every_crash_point_is_exercised():
    """A build step added without a crash test is a step nobody has ever interrupted."""
    from acis.store import snapshots

    assert set(snapshots.CRASH_POINTS) == set(CRASH_POINTS)


def test_the_journal_survives_the_crash(store):
    from acis.store import snapshots

    tmp_path, _ = store
    run_child(tmp_path, "apps", "v2", "after_manifest")
    events = [e["event"] for e in snapshots.history("apps")]
    assert "build.started" in events  # written before the work, so the crash is visible in the record
    assert json.dumps(events)  # serialisable: the journal is evidence, and evidence gets published
