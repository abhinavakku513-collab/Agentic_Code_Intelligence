"""The content-addressed store and the catalog (docs/spec/04 §1, D10, INV-1, INV-9).

Everything the engine returns as evidence is read back out of the CAS by hash, and every P1 promise — incremental
builds that reuse unchanged content, snapshots that cannot be torn, rollback that is instant — rests on the store
being append-only and its writes being atomic. So the tests are about the properties a filesystem will violate
the moment it is interrupted: a half-written blob, a rewritten one, a catalog that lost its last transaction.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from acis.core.errors import InvalidInput, NotFound
from acis.store import cas, catalog, layout

TEXT = "def solve():\n\treturn 42\n"


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))


# -- layout ------------------------------------------------------------------------------------------------------
def test_every_store_path_lives_under_acis_home(tmp_path):
    home = (tmp_path / "home").resolve()
    for path in (layout.cas_root(), layout.catalog_path(), layout.repo_root("apps"), layout.refs_dir("apps")):
        assert path.is_relative_to(home)


def test_a_repository_id_cannot_walk_out_of_the_store():
    """Repository ids come from a request body. `../` in one must not become a directory anywhere."""
    for hostile in ("../escape", "/etc", "a/b", "..", ""):
        with pytest.raises(InvalidInput):
            layout.repo_root(hostile)


def test_blobs_are_sharded_by_hash_prefix():
    digest = "ab" + "c" * 62
    path = layout.blob_path(digest)
    assert path.parent.name == "ab" and path.name == "c" * 62


def test_a_blob_key_that_is_not_a_hash_is_refused():
    for hostile in ("../../etc/passwd", "short", "", "g" * 64):
        with pytest.raises(InvalidInput):
            layout.blob_path(hostile)


# -- the CAS -----------------------------------------------------------------------------------------------------
def test_text_round_trips_byte_for_byte():
    """INV-1: what comes back is what was stored — tabs, CRLF and non-ASCII included, unchanged."""
    store = cas.BlobStore.open()
    for text in (TEXT, "a\r\nb\t c", "héllo — ünïcode 🎯", "x" * 100_000, ""):
        assert store.get(store.put(text)) == text


def test_the_same_content_is_one_blob_however_often_it_is_written():
    store = cas.BlobStore.open()
    first = store.put(TEXT)
    written_at = layout.blob_path(first).stat().st_mtime_ns
    again = store.put(TEXT)
    assert again == first
    assert layout.blob_path(first).stat().st_mtime_ns == written_at  # append-only: never rewritten
    assert store.stats["writes"] == 1 and store.stats["exists"] == 1


def test_an_unknown_hash_is_not_found_rather_than_empty():
    with pytest.raises(NotFound):
        cas.BlobStore.open().get("0" * 64)


def test_a_tampered_blob_is_detected():
    """The store is the evidence path. A blob whose bytes no longer hash to its name is corruption, not content."""
    store = cas.BlobStore.open()
    digest = store.put(TEXT)
    assert store.verify(digest)

    layout.blob_path(digest).write_bytes(b"not the original bytes")
    assert not store.verify(digest)
    with pytest.raises(InvalidInput, match="does not match"):
        store.get(digest, verify=True)


def test_an_interrupted_write_leaves_no_blob():
    """A crash between opening the temporary file and renaming it must not publish a truncated blob."""
    store = cas.BlobStore.open()
    digest = cas.body_hash(TEXT)
    partial = layout.blob_path(digest).with_suffix(".tmp-crashed")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")
    assert not store.has(digest)

    assert store.put(TEXT) == digest and store.has(digest)


def test_concurrent_writers_of_the_same_content_agree():
    store = cas.BlobStore.open()
    results: list[str] = []
    errors: list[BaseException] = []

    def write() -> None:
        try:
            results.append(store.put(TEXT))
        except BaseException as exc:  # noqa: BLE001 — the point is that nothing escapes
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(set(results)) == 1
    assert store.get(results[0]) == TEXT


def test_a_blob_larger_than_the_limit_is_refused_not_truncated():
    from acis.core.errors import ResourceLimit

    store = cas.BlobStore.open(max_blob_bytes=64)
    with pytest.raises(ResourceLimit):
        store.put("x" * 1000)


def test_put_many_reports_what_was_new():
    store = cas.BlobStore.open()
    store.put(TEXT)
    digests, new = store.put_many([TEXT, "another body", "another body"])
    assert len(digests) == 3 and digests[1] == digests[2]
    assert new == 1  # only "another body" was unknown, and only once


# -- the catalog -------------------------------------------------------------------------------------------------
def test_the_catalog_is_durable_by_configuration():
    """P1 survives `kill -9`, which is a promise about SQLite's settings as much as about our code."""
    with catalog.open_catalog() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert int(db.execute("PRAGMA synchronous").fetchone()[0]) == 2  # FULL
        assert int(db.execute("PRAGMA foreign_keys").fetchone()[0]) == 1


def test_registering_a_repository_twice_is_not_an_error():
    with catalog.open_catalog() as db:
        catalog.register_repo(db, "apps", source_kind="jsonl")
        catalog.register_repo(db, "apps", source_kind="jsonl")
        assert [r["repo_id"] for r in catalog.list_repos(db)] == ["apps"]


def test_versions_keep_the_order_they_arrived_in():
    """`as_of` and `range:` selectors read this order; a set would make them meaningless."""
    with catalog.open_catalog() as db:
        catalog.register_repo(db, "apps", source_kind="jsonl")
        for label in ("v1", "v2", "v3"):
            catalog.record_version(db, "apps", label, snapshot_id=f"s_{label}")
        assert [v["label"] for v in catalog.list_versions(db, "apps")] == ["v1", "v2", "v3"]
        assert [v["ordinal"] for v in catalog.list_versions(db, "apps")] == [0, 1, 2]


def test_a_version_of_an_unknown_repository_is_refused():
    with catalog.open_catalog() as db, pytest.raises(sqlite3.IntegrityError):
        catalog.record_version(db, "never-registered", "v1", snapshot_id="s_1")


def test_the_journal_records_every_event_in_order():
    with catalog.open_catalog() as db:
        catalog.register_repo(db, "apps", source_kind="jsonl")
        catalog.journal(db, "apps", "build.started", {"snapshot": "s_1"})
        catalog.journal(db, "apps", "build.activated", {"snapshot": "s_1"})
        events = [row["event"] for row in catalog.read_journal(db, "apps")]
        assert events == ["build.started", "build.activated"]


def test_the_catalog_survives_being_reopened():
    with catalog.open_catalog() as db:
        catalog.register_repo(db, "apps", source_kind="jsonl")
        catalog.record_version(db, "apps", "v1", snapshot_id="s_1")
    with catalog.open_catalog() as db:
        assert [v["label"] for v in catalog.list_versions(db, "apps")] == ["v1"]


def test_an_older_schema_is_migrated_rather_than_guessed_at():
    path = layout.catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA user_version = 0")
        raw.execute("CREATE TABLE IF NOT EXISTS leftovers (x INTEGER)")
    with catalog.open_catalog() as db:
        assert int(db.execute("PRAGMA user_version").fetchone()[0]) == catalog.SCHEMA_VERSION
        catalog.register_repo(db, "apps", source_kind="jsonl")


def test_a_future_schema_is_refused_rather_than_downgraded():
    """A newer ACIS wrote this store. Guessing at its tables is how data is lost."""
    path = layout.catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as raw:
        raw.execute(f"PRAGMA user_version = {catalog.SCHEMA_VERSION + 1}")
    with pytest.raises(InvalidInput, match="newer"), catalog.open_catalog():
        pass
