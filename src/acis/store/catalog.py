"""The catalog: repositories, versions, refs, jobs and the journal (docs/spec/04 §1, §3, D10).

Stdlib SQLite, WAL, `synchronous=FULL`. Those three settings are the whole reason P1 can promise that a `kill -9`
loses nothing that was acknowledged: WAL lets a reader keep reading while a build commits, and FULL means a
committed transaction is on the platter rather than in a buffer. They are asserted by a test, because a default
that quietly changes would turn a durability guarantee into a hope.

The **journal** is the other half. Snapshot activation touches a ref file and a database row; a crash between
them would otherwise leave the store disagreeing with itself. Every build step appends a journal entry first, so
recovery can replay what was in flight and decide — rather than guess — what happened.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from acis.core.errors import InvalidInput
from acis.store.layout import catalog_path, safe_name

#: v2 keys snapshots by (repository, snapshot): snapshot ids are content-addressed and not repository-scoped, so
#: two repositories holding identical content share an id — under v1 the second one's row replaced the first's,
#: and the first then listed its versions with zero units.
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    repo_id     TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    created_ts  REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS versions (
    repo_id     TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    snapshot_id TEXT,
    parent      TEXT,
    state       TEXT NOT NULL DEFAULT 'BUILDING',
    created_ts  REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (repo_id, label)
);
CREATE INDEX IF NOT EXISTS versions_by_order ON versions(repo_id, ordinal);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT NOT NULL,
    repo_id     TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    state       TEXT NOT NULL,
    n_units     INTEGER NOT NULL DEFAULT 0,
    config_hash TEXT NOT NULL DEFAULT '',
    tree_hash   TEXT NOT NULL DEFAULT '',
    created_ts  REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (repo_id, snapshot_id)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    repo_id    TEXT NOT NULL,
    state      TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    repo_id TEXT NOT NULL,
    event   TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS journal_by_repo ON journal(repo_id, seq);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = FULL")  # D10: a committed build survives the power going out
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 30000")
    return db


def _migrate(db: sqlite3.Connection) -> None:
    version = int(db.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise InvalidInput(
            "this store was written by a newer ACIS; refusing to open it with an older schema",
            found=version,
            supported=SCHEMA_VERSION,
        )
    if version < SCHEMA_VERSION:
        has_snapshots = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshots'").fetchone()
        if version == 1 and has_snapshots:
            # v1 -> v2: rebuild the snapshots table under the (repository, snapshot) key, rows unchanged — one
            # script, one transaction (`executescript` would commit anything opened around it).
            db.executescript(
                "BEGIN IMMEDIATE;"
                "ALTER TABLE snapshots RENAME TO snapshots_v1;"
                + SCHEMA
                + "INSERT OR IGNORE INTO snapshots SELECT * FROM snapshots_v1;"
                "DROP TABLE snapshots_v1;"
                f"PRAGMA user_version = {SCHEMA_VERSION};"
                "COMMIT;"
            )
            return
        db.executescript(SCHEMA)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def open_catalog(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open (and migrate) the catalog. The connection is closed on the way out, committed or not."""
    db = _connect(Path(path) if path is not None else catalog_path())
    try:
        _migrate(db)
        yield db
    finally:
        db.close()


# -- repositories and versions -------------------------------------------------------------------------------------
def register_repo(
    db: sqlite3.Connection, repo_id: str, *, source_kind: str, meta: Mapping[str, Any] | None = None
) -> str:
    """Idempotent: re-registering an existing repository keeps its creation time and its history."""
    safe_name(repo_id, what="repo id")
    db.execute(
        "INSERT INTO repos (repo_id, source_kind, created_ts, meta) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(repo_id) DO UPDATE SET source_kind = excluded.source_kind",
        (repo_id, source_kind, time.time(), json.dumps(dict(meta or {}), sort_keys=True)),
    )
    return repo_id


def get_repo_exists(db: sqlite3.Connection, repo_id: str) -> bool:
    return db.execute("SELECT 1 FROM repos WHERE repo_id = ?", (repo_id,)).fetchone() is not None


def drop_repo(db: sqlite3.Connection, repo_id: str) -> None:
    """Remove every row that belongs to one repository. Explicit rather than cascaded: journal and jobs have no
    foreign key, and a row left behind here is exactly what made a dropped repository haunt the next ingest."""
    for table in ("jobs", "journal", "versions", "snapshots", "repos"):
        db.execute(f"DELETE FROM {table} WHERE repo_id = ?", (repo_id,))  # noqa: S608 — fixed table names


def list_repos(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(db.execute("SELECT * FROM repos ORDER BY repo_id"))


def record_version(
    db: sqlite3.Connection,
    repo_id: str,
    label: str,
    *,
    snapshot_id: str | None = None,
    parent: str | None = None,
    state: str = "BUILDING",
    meta: Mapping[str, Any] | None = None,
) -> int:
    """Append a version, or update the one that already carries this label. Returns its ordinal.

    The ordinal is the arrival order and it is never reused: `as_of` and `range:` selectors are defined over it,
    so a version's position in the history has to be a fact rather than a sort of its label.
    """
    safe_name(label, what="version label")
    existing = db.execute("SELECT ordinal FROM versions WHERE repo_id = ? AND label = ?", (repo_id, label)).fetchone()
    if existing is not None:
        db.execute(
            "UPDATE versions SET snapshot_id = ?, parent = COALESCE(?, parent), state = ? "
            "WHERE repo_id = ? AND label = ?",
            (snapshot_id, parent, state, repo_id, label),
        )
        return int(existing["ordinal"])

    row = db.execute("SELECT COALESCE(MAX(ordinal) + 1, 0) AS next FROM versions WHERE repo_id = ?", (repo_id,))
    ordinal = int(row.fetchone()["next"])
    db.execute(
        "INSERT INTO versions (repo_id, label, ordinal, snapshot_id, parent, state, created_ts, meta) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            repo_id,
            label,
            ordinal,
            snapshot_id,
            parent,
            state,
            time.time(),
            json.dumps(dict(meta or {}), sort_keys=True),
        ),
    )
    return ordinal


def list_versions(db: sqlite3.Connection, repo_id: str) -> list[sqlite3.Row]:
    return list(db.execute("SELECT * FROM versions WHERE repo_id = ? ORDER BY ordinal", (repo_id,)))


def get_version(db: sqlite3.Connection, repo_id: str, label: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = db.execute(
        "SELECT * FROM versions WHERE repo_id = ? AND label = ?", (repo_id, label)
    ).fetchone()
    return row


# -- snapshots -------------------------------------------------------------------------------------------------------
def record_snapshot(
    db: sqlite3.Connection,
    snapshot_id: str,
    *,
    repo_id: str,
    label: str,
    state: str,
    n_units: int = 0,
    config_hash: str = "",
    tree_hash: str = "",
    meta: Mapping[str, Any] | None = None,
) -> None:
    db.execute(
        "INSERT INTO snapshots (snapshot_id, repo_id, label, state, n_units, config_hash, tree_hash, created_ts, meta)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(repo_id, snapshot_id) DO UPDATE SET "
        "state = excluded.state, n_units = excluded.n_units, meta = excluded.meta",
        (
            snapshot_id,
            repo_id,
            label,
            state,
            n_units,
            config_hash,
            tree_hash,
            time.time(),
            json.dumps(dict(meta or {}), sort_keys=True),
        ),
    )


def set_snapshot_state(db: sqlite3.Connection, snapshot_id: str, state: str, *, repo_id: str) -> None:
    """Change a snapshot's state without touching what it *is*.

    Activation is a state change, not a rebuild: going through `record_snapshot` would upsert the row and reset
    the unit count to whatever the caller happened to pass, which is how a snapshot ends up claiming zero units.
    """
    db.execute("UPDATE snapshots SET state = ? WHERE repo_id = ? AND snapshot_id = ?", (state, repo_id, snapshot_id))


def get_snapshot(db: sqlite3.Connection, snapshot_id: str, *, repo_id: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = db.execute(
        "SELECT * FROM snapshots WHERE repo_id = ? AND snapshot_id = ?", (repo_id, snapshot_id)
    ).fetchone()
    return row


def list_snapshots(db: sqlite3.Connection, repo_id: str) -> list[sqlite3.Row]:
    return list(db.execute("SELECT * FROM snapshots WHERE repo_id = ? ORDER BY created_ts", (repo_id,)))


# -- the journal -------------------------------------------------------------------------------------------------
def journal(db: sqlite3.Connection, repo_id: str, event: str, payload: Mapping[str, Any] | None = None) -> int:
    """Append an event. Written *before* the action it describes, so recovery can tell what was in flight."""
    cursor = db.execute(
        "INSERT INTO journal (ts, repo_id, event, payload) VALUES (?, ?, ?, ?)",
        (time.time(), repo_id, event, json.dumps(dict(payload or {}), sort_keys=True, default=str)),
    )
    return int(cursor.lastrowid or 0)


def read_journal(db: sqlite3.Connection, repo_id: str | None = None, *, since: int = 0) -> list[sqlite3.Row]:
    if repo_id is None:
        return list(db.execute("SELECT * FROM journal WHERE seq > ? ORDER BY seq", (since,)))
    return list(db.execute("SELECT * FROM journal WHERE repo_id = ? AND seq > ? ORDER BY seq", (repo_id, since)))


def payload_of(row: sqlite3.Row) -> dict[str, Any]:
    data = json.loads(row["payload"] or "{}")
    return data if isinstance(data, dict) else {}


# -- jobs ---------------------------------------------------------------------------------------------------------
def record_job(db: sqlite3.Connection, job_id: str, *, repo_id: str, state: str, detail: str = "") -> None:
    now = time.time()
    db.execute(
        "INSERT INTO jobs (job_id, repo_id, state, detail, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(job_id) DO UPDATE SET state = excluded.state, detail = excluded.detail, updated_ts = ?",
        (job_id, repo_id, state, detail, now, now, now),
    )


def get_job(db: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return row


def rows_as_dicts(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "get_job",
    "drop_repo",
    "get_repo_exists",
    "get_snapshot",
    "get_version",
    "journal",
    "list_repos",
    "list_snapshots",
    "list_versions",
    "open_catalog",
    "payload_of",
    "read_journal",
    "record_job",
    "record_snapshot",
    "record_version",
    "register_repo",
    "set_snapshot_state",
    "rows_as_dicts",
]
