"""Immutable snapshots: build, validate, activate, roll back, recover (docs/spec/04 §3, D11, INV-2, INV-9).

The whole P1 story is one rule — **a snapshot is never modified after it exists** — and everything else follows
from it:

* A build writes into `.tmp-<token>/` and is invisible until it is renamed into `snapshots/<id>/`. A process
  killed at any point before the rename leaves a directory whose name says it is unfinished, and recovery deletes
  it. There is no state in which half a snapshot is searchable.
* `VALID` is written **last**, after the manifest, and holds a hash over the manifest. A snapshot without it, or
  with one that disagrees, is not opened: it is reported as corrupt (INV-9).
* Activation is `os.replace` on a ref file, so a reader either sees the old snapshot or the new one, never a torn
  pointer. The previous ref is kept, which is why rollback is instant and costs nothing: the snapshot it rolls
  back to was never touched.
* `snapshot_id` is a hash of the content and the build configuration, so the same tree built twice is the same
  snapshot — identical versions cost nothing, and a rebuild is detectable rather than assumed.

Vectors and the lexical index are built *into* a snapshot by the engine (Track B1's integration step); this module
owns the lifecycle, the content and the guarantees around them.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acis.core.errors import InvalidInput, NotFound, SnapshotInvalid, VersionConflict
from acis.core.hashing import hash_obj, sha256_text, short
from acis.obs.log import get_logger
from acis.store import catalog, layout
from acis.store.cas import BlobStore

log = get_logger("acis.store")

SNAPSHOT_PREFIX = "s_"
STATES = ("BUILDING", "READY_LEX", "READY", "ACTIVE", "RETIRED", "FAILED")

#: Named points at which a build can be interrupted, in the order the build reaches them. They exist so the chaos
#: suite can kill a real process at each one (`ACIS_CRASH_AT`, docs/spec/04 §5); nothing else reads them, and a
#: build with the variable unset never checks more than a dictionary lookup.
CRASH_POINTS = ("before_blobs", "after_blobs", "after_manifest", "before_marker", "after_marker", "before_activate")


def _crash_if_asked(point: str) -> None:
    """`kill -9` this process when the test harness asked to be interrupted here.

    SIGKILL rather than an exception on purpose: an exception unwinds and runs cleanup, which is precisely the
    behaviour a power cut does not have. Recovery has to work without it.
    """
    if os.environ.get("ACIS_CRASH_AT") == point:
        os.kill(os.getpid(), 9)


@dataclass(frozen=True, slots=True)
class UnitInput:
    """One unit as a source hands it over: an external key and its text."""

    key: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredUnit:
    """One unit as the snapshot holds it. `body_hash` is the only way its text is ever read back (INV-1)."""

    unit_id: str
    key: str
    body_hash: str
    ordinal: int
    n_bytes: int
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """What a build produced. `state` is `READY` until activation makes it `ACTIVE`."""

    snapshot_id: str
    repo_id: str
    label: str
    n_units: int
    n_new_blobs: int
    tree_hash: str
    config_hash: str
    seconds: float
    state: str = "READY"


# -- identity --------------------------------------------------------------------------------------------------
def tree_hash(pairs: Sequence[tuple[str, str]]) -> str:
    """`sha256(sorted[(key, body_hash)])` — the content of a version, independent of the order it arrived in."""
    return hash_obj(sorted((str(k), str(h)) for k, h in pairs))


def snapshot_id_for(tree: str, config_hash: str) -> str:
    """Identical content and identical build settings ⇒ identical snapshot (D11).

    Deliberately **not** keyed on the source (spec 04 §1 includes `source_id`; ADR-0006 records the deviation).
    Feeding the source in would mean the same corpus ingested from a JSONL file and from a directory produced two
    different snapshots of identical content, and re-ingesting one version of a history would not recognise the
    version it already had — which is exactly the dedup D11 promises. The source stays in the manifest as
    provenance, where it answers "where did this come from" without deciding "is this the same content".
    """
    return SNAPSHOT_PREFIX + short(hash_obj({"tree": tree, "config": config_hash}), 16)


def unit_id_for(snapshot_id: str, key: str) -> str:
    return "u_" + short(hash_obj({"snapshot": snapshot_id, "key": key}), 16)


# -- building --------------------------------------------------------------------------------------------------
def build_snapshot(
    *,
    repo_id: str,
    label: str,
    units: Sequence[UnitInput],
    store: BlobStore | None = None,
    config_hash: str = "",
    source_id: str = "",
    parent: str | None = None,
    state: str = "READY",
) -> SnapshotRecord:
    """Build one immutable snapshot. Returns as soon as it is `VALID` — activation is a separate decision.

    Content goes into the shared CAS first, so a body that already exists costs nothing: that is the difference
    between a ten-unit change and a full rebuild, and it is measured rather than assumed (spec 04 §5).
    """
    layout.safe_name(repo_id, what="repo id")
    layout.safe_name(label, what="version label")
    if not units:
        raise InvalidInput("a snapshot needs at least one unit: refusing to build from no units", repo=repo_id)
    keys = [u.key for u in units]
    if len(set(keys)) != len(keys):
        raise InvalidInput("duplicate unit keys in one version", n_units=len(keys), unique=len(set(keys)))

    started = time.perf_counter()
    blobs = store or BlobStore.open()
    _crash_if_asked("before_blobs")
    digests, new_blobs = blobs.put_many(u.text for u in units)
    _crash_if_asked("after_blobs")

    tree = tree_hash(list(zip(keys, digests, strict=True)))
    snapshot_id = snapshot_id_for(tree, config_hash)

    with catalog.open_catalog() as db:
        catalog.register_repo(db, repo_id, source_kind="units")
        catalog.journal(db, repo_id, "build.started", {"snapshot_id": snapshot_id, "label": label, "units": len(units)})

    # Ordinals follow the sorted key order, not the arrival order, so the same content always yields the same
    # unit ids — a rebuild of an unchanged version is recognisable as the same snapshot (D11).
    ordered = sorted(zip(keys, digests, units, strict=True), key=lambda item: item[0])
    stored = [
        StoredUnit(
            unit_id=unit_id_for(snapshot_id, key),
            key=key,
            body_hash=digest,
            ordinal=index,
            n_bytes=len(unit.text.encode("utf-8")),
            meta=dict(unit.meta),
        )
        for index, (key, digest, unit) in enumerate(ordered)
    ]

    target = layout.snapshot_dir(repo_id, snapshot_id)
    if target.is_dir() and (target / layout.VALID_MARKER).is_file():
        # Same content, same config: the snapshot already exists and is immutable. Nothing to do but say so.
        seconds = time.perf_counter() - started
        _record(repo_id, label, snapshot_id, stored, tree, config_hash, parent, state)
        return SnapshotRecord(snapshot_id, repo_id, label, len(stored), new_blobs, tree, config_hash, seconds, state)

    staging = layout.tmp_dir(repo_id, uuid.uuid4().hex[:12])
    staging.mkdir(parents=True, exist_ok=False)
    try:
        manifest = {
            "snapshot_id": snapshot_id,
            "repo_id": repo_id,
            "label": label,
            "n_units": len(stored),
            "tree_hash": tree,
            "config_hash": config_hash,
            "source_id": source_id,
            "parent": parent,
            "created_ts": time.time(),
            "units": [
                {
                    "unit_id": u.unit_id,
                    "key": u.key,
                    "body_hash": u.body_hash,
                    "ordinal": u.ordinal,
                    "n_bytes": u.n_bytes,
                    "meta": dict(u.meta),
                }
                for u in stored
            ],
        }
        _write_json(staging / layout.MANIFEST, manifest)
        (staging / layout.LEXICAL_DIR).mkdir(exist_ok=True)
        _crash_if_asked("after_manifest")

        problems = _validate_staged(staging, blobs)
        if problems:
            raise SnapshotInvalid("the built snapshot did not validate", problems=problems[:5])

        # The marker goes last and carries the manifest's hash: its presence is the claim that the build finished,
        # and its content is what makes that claim checkable.
        _crash_if_asked("before_marker")
        _write_text(staging / layout.VALID_MARKER, manifest_hash(manifest))
        _crash_if_asked("after_marker")
        _fsync_tree(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        _fsync_dir(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _record(repo_id, label, snapshot_id, stored, tree, config_hash, parent, state)
    with catalog.open_catalog() as db:
        catalog.journal(db, repo_id, "build.validated", {"snapshot_id": snapshot_id, "units": len(stored)})
    seconds = time.perf_counter() - started
    log.info("snapshot.built", repo=repo_id, snapshot=snapshot_id, units=len(stored), new_blobs=new_blobs)
    return SnapshotRecord(snapshot_id, repo_id, label, len(stored), new_blobs, tree, config_hash, seconds, state)


def _record(
    repo_id: str,
    label: str,
    snapshot_id: str,
    stored: Sequence[StoredUnit],
    tree: str,
    config_hash: str,
    parent: str | None,
    state: str,
) -> None:
    with catalog.open_catalog() as db:
        catalog.register_repo(db, repo_id, source_kind="units")
        catalog.record_snapshot(
            db,
            snapshot_id,
            repo_id=repo_id,
            label=label,
            state=state,
            n_units=len(stored),
            config_hash=config_hash,
            tree_hash=tree,
        )
        catalog.record_version(db, repo_id, label, snapshot_id=snapshot_id, parent=parent, state=state)


# -- reading ---------------------------------------------------------------------------------------------------
def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """What the `VALID` marker holds. Covers the unit list, so a tampered manifest cannot pass unnoticed."""
    return sha256_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str))


def read_manifest(repo_id: str, snapshot_id: str) -> dict[str, Any]:
    path = layout.snapshot_dir(repo_id, snapshot_id) / layout.MANIFEST
    if not path.is_file():
        raise NotFound(f"no snapshot {snapshot_id} for {repo_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def open_snapshot(repo_id: str, snapshot_id: str) -> dict[str, Any]:
    """Open a snapshot for reading. Refuses anything that is not `VALID` (INV-9)."""
    directory = layout.snapshot_dir(repo_id, snapshot_id)
    # A snapshot that was never built and one that failed to finish are different facts, and the API distinguishes
    # them: `NotFound` (404) is "no such snapshot", `SnapshotInvalid` (409) is "it is there and it is not usable".
    if not directory.is_dir():
        raise NotFound(f"no snapshot {snapshot_id} for {repo_id}")
    marker = directory / layout.VALID_MARKER
    if not marker.is_file():
        raise SnapshotInvalid(f"snapshot {snapshot_id} has no VALID marker: the build did not finish (INV-9)")
    manifest = read_manifest(repo_id, snapshot_id)
    if marker.read_text(encoding="utf-8").strip() != manifest_hash(manifest):
        raise SnapshotInvalid(f"snapshot {snapshot_id} does not match its VALID marker; it is corrupt")
    return manifest


def read_units(repo_id: str, snapshot_id: str) -> list[StoredUnit]:
    manifest = open_snapshot(repo_id, snapshot_id)
    return [
        StoredUnit(
            unit_id=str(u["unit_id"]),
            key=str(u["key"]),
            body_hash=str(u["body_hash"]),
            ordinal=int(u["ordinal"]),
            n_bytes=int(u.get("n_bytes", 0)),
            meta=dict(u.get("meta") or {}),
        )
        for u in manifest.get("units", [])
    ]


def validate(repo_id: str, snapshot_id: str, *, store: BlobStore | None = None, sample: int = 64) -> list[str]:
    """Re-check a snapshot on disk. Returns the problems; empty means it is exactly what it claims to be."""
    try:
        manifest = open_snapshot(repo_id, snapshot_id)
    except (SnapshotInvalid, NotFound) as exc:
        return [str(exc)]
    return _validate_manifest(manifest, store or BlobStore.open(), sample=sample)


def _validate_staged(staging: Path, store: BlobStore, *, sample: int = 64) -> list[str]:
    manifest = json.loads((staging / layout.MANIFEST).read_text(encoding="utf-8"))
    return _validate_manifest(manifest, store, sample=sample)


def _validate_manifest(manifest: Mapping[str, Any], store: BlobStore, *, sample: int) -> list[str]:
    """Counts agree, bodies exist, and a deterministic sample of them still hashes to its name (spec 04 §3.6)."""
    problems: list[str] = []
    units = list(manifest.get("units", []))
    if len(units) != int(manifest.get("n_units", -1)):
        problems.append(f"manifest says {manifest.get('n_units')} units but lists {len(units)}")

    recomputed = tree_hash([(u["key"], u["body_hash"]) for u in units])
    if recomputed != manifest.get("tree_hash"):
        problems.append("the unit list does not match the recorded tree hash")

    # Deterministic sample: the same snapshot always checks the same bodies, so a validation failure reproduces.
    step = max(1, len(units) // max(1, sample))
    for unit in units[::step][:sample]:
        digest = str(unit["body_hash"])
        if not store.has(digest):
            problems.append(f"unit {unit['key']}: body {digest[:12]} is missing from the content store")
        elif not store.verify(digest):
            problems.append(f"unit {unit['key']}: body {digest[:12]} does not match its hash")
    return problems


# -- refs, activation and rollback -------------------------------------------------------------------------------
def _read_ref(repo_id: str, name: str) -> str | None:
    path = layout.ref_path(repo_id, name)
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _write_ref(repo_id: str, name: str, snapshot_id: str) -> None:
    """Atomic: write beside the ref, fsync, rename. A reader sees one value or the other, never a torn one."""
    path = layout.ref_path(repo_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(snapshot_id + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def active_snapshot_id(repo_id: str) -> str | None:
    return _read_ref(repo_id, layout.ACTIVE_REF)


def previous_snapshot_id(repo_id: str) -> str | None:
    return _read_ref(repo_id, layout.PREV_REF)


def activate(repo_id: str, snapshot_id: str, *, expected_active: str | None = None) -> str:
    """Point `ACTIVE` at a validated snapshot, keeping the old one as `PREV`.

    `expected_active` is the compare-and-swap `update_version` needs: a writer working from a stale view must
    lose rather than silently overwrite a version someone else activated meanwhile (spec 04 §4).
    """
    open_snapshot(repo_id, snapshot_id)  # refuses anything that is not VALID
    current = active_snapshot_id(repo_id)
    if expected_active is not None and current != expected_active:
        raise VersionConflict(
            "the active snapshot is not the one this update expected",
            expected=expected_active,
            actual=current,
        )
    if current == snapshot_id:
        return snapshot_id

    if current:
        _write_ref(repo_id, layout.PREV_REF, current)
    _write_ref(repo_id, layout.ACTIVE_REF, snapshot_id)
    with catalog.open_catalog() as db:
        catalog.set_snapshot_state(db, snapshot_id, "ACTIVE")
        if current:
            catalog.set_snapshot_state(db, current, "READY")
        catalog.journal(db, repo_id, "build.activated", {"snapshot_id": snapshot_id, "previous": current})
    log.info("snapshot.activated", repo=repo_id, snapshot=snapshot_id, previous=current)
    return snapshot_id


def rollback(repo_id: str) -> str:
    """Swap `ACTIVE` and `PREV`. Instant, because both snapshots exist and neither is modified."""
    current = active_snapshot_id(repo_id)
    previous = previous_snapshot_id(repo_id)
    if not previous:
        raise NotFound(f"{repo_id} has no previous snapshot to roll back to")
    open_snapshot(repo_id, previous)
    _write_ref(repo_id, layout.ACTIVE_REF, previous)
    if current:
        _write_ref(repo_id, layout.PREV_REF, current)
    with catalog.open_catalog() as db:
        catalog.journal(db, repo_id, "rollback", {"snapshot_id": previous, "from": current})
    log.info("snapshot.rolled_back", repo=repo_id, snapshot=previous, previous=current)
    return previous


# -- recovery ------------------------------------------------------------------------------------------------------
def recover(repo_id: str) -> dict[str, Any]:
    """Startup recovery (spec 04 §3): sweep unfinished builds, then make sure what is ACTIVE is really serveable.

    A damaged active snapshot falls back to `PREV`, and the fallback is an **incident**: it is journalled and
    returned, never absorbed quietly, because a repository silently serving an older version is exactly the kind
    of thing that is discovered at the worst moment.
    """
    root = layout.repo_root(repo_id)
    removed = 0
    for path in sorted(root.glob(".tmp-*")):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1

    incidents: list[str] = []
    fell_back = False
    active = active_snapshot_id(repo_id)
    if active and not _is_serveable(repo_id, active):
        incidents.append(f"active snapshot {active} is not valid")
        previous = previous_snapshot_id(repo_id)
        if previous and _is_serveable(repo_id, previous):
            _write_ref(repo_id, layout.ACTIVE_REF, previous)
            fell_back = True
            incidents.append(f"fell back to {previous}")
        else:
            layout.ref_path(repo_id, layout.ACTIVE_REF).unlink(missing_ok=True)
            incidents.append("no serveable snapshot remains; this repository needs a rebuild")
        active = active_snapshot_id(repo_id)

    if removed or incidents:
        with catalog.open_catalog() as db:
            catalog.register_repo(db, repo_id, source_kind="units")
            catalog.journal(
                db,
                repo_id,
                "recovery",
                {"removed_partial_builds": removed, "incidents": incidents, "active": active},
            )
        log.warning("store.recovered", repo=repo_id, removed=removed, incidents=len(incidents))

    return {
        "repo_id": repo_id,
        "removed_partial_builds": removed,
        "active_fell_back_to_previous": fell_back,
        "incidents": incidents,
        "active": active,
    }


def _is_serveable(repo_id: str, snapshot_id: str) -> bool:
    try:
        open_snapshot(repo_id, snapshot_id)
    except (SnapshotInvalid, NotFound, InvalidInput, OSError):
        return False
    return True


def history(repo_id: str) -> list[dict[str, Any]]:
    """The journal for one repository, oldest first — what happened, in the order it happened."""
    with catalog.open_catalog() as db:
        return [
            {
                "seq": int(row["seq"]),
                "ts": float(row["ts"]),
                "event": str(row["event"]),
                "payload": catalog.payload_of(row),
            }
            for row in catalog.read_journal(db, repo_id)
        ]


# -- filesystem helpers ---------------------------------------------------------------------------------------------
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    """Everything on the platter before the rename that publishes it (spec 04 §3.7)."""
    for child in sorted(path.rglob("*")):
        if child.is_file():
            fd = os.open(child, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_dir(path)


__all__ = [
    "SNAPSHOT_PREFIX",
    "STATES",
    "SnapshotRecord",
    "StoredUnit",
    "UnitInput",
    "activate",
    "active_snapshot_id",
    "build_snapshot",
    "history",
    "manifest_hash",
    "open_snapshot",
    "previous_snapshot_id",
    "read_manifest",
    "read_units",
    "recover",
    "rollback",
    "snapshot_id_for",
    "tree_hash",
    "unit_id_for",
    "validate",
]
