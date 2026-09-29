"""Snapshot lifecycle: build → validate → activate → roll back → recover (docs/spec/04 §3, INV-2, INV-9, D11).

This is the P1 promise in one file. A snapshot is immutable once it is `VALID`, so activation is a rename and
rollback is a rename back; nothing partial is ever visible, and a process killed at any point leaves a store that
either has the new snapshot or does not — never half of it.

The tests are written against the failure, not the feature: a build interrupted before its marker, a `VALID` file
that disagrees with the manifest, an activation racing a reader, a repository whose active snapshot has gone
missing entirely.
"""

from __future__ import annotations

import json

import pytest

from acis.core.errors import InvalidInput, NotFound, SnapshotInvalid, VersionConflict
from acis.store import catalog, layout, snapshots
from acis.store.cas import BlobStore

UNITS_V1 = [("u1", "def a():\n    return 1\n"), ("u2", "def b():\n    return 2\n")]
UNITS_V2 = [("u1", "def a():\n    return 1\n"), ("u2", "def b():\n    return 22\n"), ("u3", "def c():\n    pass\n")]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))


def build(repo="apps", label="v1", units=UNITS_V1, *, activate=True, config_hash="cfg") -> snapshots.SnapshotRecord:
    store = BlobStore.open()
    record = snapshots.build_snapshot(
        repo_id=repo,
        label=label,
        units=[snapshots.UnitInput(key=k, text=t) for k, t in units],
        store=store,
        config_hash=config_hash,
    )
    if activate:
        snapshots.activate(repo, record.snapshot_id)
    return record


# -- identity ------------------------------------------------------------------------------------------------------
def test_the_same_content_produces_the_same_snapshot_id():
    """D11: identical trees dedup. A rebuild of unchanged content must not produce a second snapshot."""
    first = build(label="v1")
    second = build(repo="other", label="whatever", units=UNITS_V1, activate=False)
    assert first.snapshot_id == second.snapshot_id


def test_changing_one_unit_changes_the_snapshot_id():
    assert build(label="v1").snapshot_id != build(label="v2", units=UNITS_V2, activate=False).snapshot_id


def test_the_configuration_is_part_of_the_snapshot_identity():
    """INV-2: a cache keyed on a snapshot id must not serve vectors built under different settings."""
    a = build(label="v1", config_hash="cfg-a")
    b = build(label="v2", units=UNITS_V1, activate=False, config_hash="cfg-b")
    assert a.snapshot_id != b.snapshot_id


def test_unit_ids_do_not_depend_on_the_order_units_arrived_in():
    forward = build(label="v1", units=UNITS_V1, activate=False)
    backward = build(label="v2", units=list(reversed(UNITS_V1)), activate=False)
    assert forward.snapshot_id == backward.snapshot_id


# -- what a built snapshot contains -------------------------------------------------------------------------------
def test_a_built_snapshot_is_valid_and_readable():
    record = build()
    directory = layout.snapshot_dir("apps", record.snapshot_id)
    assert (directory / layout.VALID_MARKER).is_file()
    assert (directory / layout.MANIFEST).is_file()

    units = snapshots.read_units("apps", record.snapshot_id)
    assert [u.key for u in units] == ["u1", "u2"]
    store = BlobStore.open()
    for unit in units:
        assert store.get(unit.body_hash)  # INV-1: every unit's body is in the CAS, by hash


def test_the_body_of_every_unit_is_in_the_content_store_once():
    """Two units with identical text share one blob — the property the whole incremental story rests on."""
    store = BlobStore.open()
    record = snapshots.build_snapshot(
        repo_id="apps",
        label="v1",
        units=[snapshots.UnitInput(key="a", text="same"), snapshots.UnitInput(key="b", text="same")],
        store=store,
        config_hash="cfg",
    )
    units = snapshots.read_units("apps", record.snapshot_id)
    assert units[0].body_hash == units[1].body_hash
    assert store.stats["writes"] == 1


def test_an_empty_source_is_refused_rather_than_activated():
    with pytest.raises(InvalidInput, match="no units"):
        build(units=[])


def test_two_units_with_the_same_key_are_refused():
    with pytest.raises(InvalidInput, match="duplicate"):
        build(units=[("dup", "a"), ("dup", "b")])


# -- activation and refs -------------------------------------------------------------------------------------------
def test_activation_points_active_at_the_new_snapshot_and_keeps_the_previous():
    first = build(label="v1")
    assert snapshots.active_snapshot_id("apps") == first.snapshot_id

    second = build(label="v2", units=UNITS_V2)
    assert snapshots.active_snapshot_id("apps") == second.snapshot_id
    assert snapshots.previous_snapshot_id("apps") == first.snapshot_id


def test_rollback_re_points_active_without_rebuilding_anything():
    first = build(label="v1")
    second = build(label="v2", units=UNITS_V2)
    restored = snapshots.rollback("apps")
    assert restored == first.snapshot_id and snapshots.active_snapshot_id("apps") == first.snapshot_id
    # Immutable snapshots: rolling back did not touch the one we rolled back *from*.
    assert (layout.snapshot_dir("apps", second.snapshot_id) / layout.VALID_MARKER).is_file()


def test_the_catalog_marks_exactly_the_active_snapshot_active_after_rollbacks():
    """The ref is the truth and the catalog's state column follows it. Rollback used to move only the ref, which
    left two rows marked ACTIVE; recovery reconciles a store written before the fix."""
    build(label="v1")
    build(label="v2", units=UNITS_V2)

    def active_rows() -> list[str]:
        with catalog.open_catalog() as db:
            return [str(r["snapshot_id"]) for r in catalog.list_snapshots(db, "apps") if r["state"] == "ACTIVE"]

    for _ in range(3):
        snapshots.rollback("apps")
        assert active_rows() == [snapshots.active_snapshot_id("apps")]
    with catalog.open_catalog() as db:  # simulate the drift an older store carries
        for row in catalog.list_snapshots(db, "apps"):
            catalog.set_snapshot_state(db, str(row["snapshot_id"]), "ACTIVE", repo_id="apps")
    snapshots.recover("apps")
    assert active_rows() == [snapshots.active_snapshot_id("apps")]


def test_rolling_back_twice_returns_to_where_it_started():
    build(label="v1")
    second = build(label="v2", units=UNITS_V2)
    snapshots.rollback("apps")
    assert snapshots.rollback("apps") == second.snapshot_id


def test_rollback_without_a_previous_snapshot_is_refused():
    build(label="v1")
    with pytest.raises(NotFound, match="no previous"):
        snapshots.rollback("apps")


def test_activating_an_unknown_snapshot_is_refused():
    build(label="v1")
    with pytest.raises(NotFound):
        snapshots.activate("apps", "s_" + "0" * 16)


def test_activation_can_require_the_active_snapshot_it_expected():
    """`update_version(expected_active=...)`: a stale writer must lose rather than overwrite (spec 04 §4)."""
    first = build(label="v1")
    second = build(label="v2", units=UNITS_V2, activate=False)
    with pytest.raises(VersionConflict):
        snapshots.activate("apps", second.snapshot_id, expected_active="s_somethingelse")
    assert snapshots.active_snapshot_id("apps") == first.snapshot_id

    snapshots.activate("apps", second.snapshot_id, expected_active=first.snapshot_id)
    assert snapshots.active_snapshot_id("apps") == second.snapshot_id


# -- validation and corruption ---------------------------------------------------------------------------------------
def test_a_snapshot_without_its_marker_is_not_searchable():
    """INV-9: the marker is written last, so its absence means the build did not finish."""
    record = build()
    (layout.snapshot_dir("apps", record.snapshot_id) / layout.VALID_MARKER).unlink()
    with pytest.raises(SnapshotInvalid):
        snapshots.open_snapshot("apps", record.snapshot_id)


def test_a_marker_that_disagrees_with_the_manifest_is_corruption():
    record = build()
    directory = layout.snapshot_dir("apps", record.snapshot_id)
    manifest = json.loads((directory / layout.MANIFEST).read_text(encoding="utf-8"))
    manifest["n_units"] = 999
    (directory / layout.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SnapshotInvalid, match="does not match"):
        snapshots.open_snapshot("apps", record.snapshot_id)


def test_validation_counts_units_against_the_manifest():
    record = build()
    problems = snapshots.validate("apps", record.snapshot_id)
    assert problems == []


# -- crash recovery ----------------------------------------------------------------------------------------------------
def test_an_interrupted_build_leaves_nothing_visible_and_is_swept_up():
    """A `.tmp-` directory is an unfinished build by construction: it is deleted, never promoted."""
    build(label="v1")
    leftover = layout.tmp_dir("apps", "abcdef123456")
    (leftover / layout.LEXICAL_DIR).mkdir(parents=True)
    (leftover / "manifest.json").write_text("{}", encoding="utf-8")

    report = snapshots.recover("apps")
    assert not leftover.exists()
    assert report["removed_partial_builds"] == 1
    assert snapshots.active_snapshot_id("apps")  # the active snapshot was untouched


def test_recovery_falls_back_to_the_previous_snapshot_when_the_active_one_is_damaged():
    first = build(label="v1")
    second = build(label="v2", units=UNITS_V2)
    (layout.snapshot_dir("apps", second.snapshot_id) / layout.VALID_MARKER).unlink()

    report = snapshots.recover("apps")
    assert snapshots.active_snapshot_id("apps") == first.snapshot_id
    assert report["active_fell_back_to_previous"] is True
    assert report["incidents"]  # never silent: a fallback is an incident


def test_recovery_reports_a_repository_with_nothing_to_serve():
    record = build(label="v1")
    (layout.snapshot_dir("apps", record.snapshot_id) / layout.VALID_MARKER).unlink()
    report = snapshots.recover("apps")
    assert snapshots.active_snapshot_id("apps") is None
    assert report["incidents"] and not report["active_fell_back_to_previous"]


def test_recovery_is_idempotent():
    build(label="v1")
    first = snapshots.recover("apps")
    second = snapshots.recover("apps")
    assert second["removed_partial_builds"] == 0 and not second["incidents"]
    assert first["active"] == second["active"]


# -- the journal tells the story ----------------------------------------------------------------------------------------
def test_every_build_and_activation_is_journalled():
    record = build()
    events = snapshots.history("apps")
    assert [e["event"] for e in events] == ["build.started", "build.validated", "build.activated"]
    assert all(e["payload"].get("snapshot_id") == record.snapshot_id for e in events)
