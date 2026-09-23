"""The engine's version surface: ingest, index, update, compare (docs/spec/04 §3–§4, D11, INV-2).

This is where the store and the retrieval engine meet. The store owns durability; the engine owns ranking; this
module is the seam, and it is deliberately thin:

* **building** a version = read a source → hand the units to `acis.store.snapshots` → embed what is not already
  embedded → write the dense matrix beside the snapshot and seal it;
* **opening** a version = resolve a selector *once*, load that snapshot's units, texts and vectors, build its
  lexical index, and hand the search path the same `SnapshotData` it has always used.

Incrementality comes from content addressing rather than from bookkeeping: a body that has been embedded before
hits the vector cache and costs no forward pass, and a version whose tree is unchanged is the *same snapshot*, so
it costs nothing at all. That is why `units_new` in a build report is a measurement rather than an estimate.

Every request resolves its selector once and pins the snapshot for the whole request (INV-2). Results therefore
cannot mix versions, even when a build activates halfway through a query.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from acis.core.errors import IndexRequired, InvalidInput, NotFound
from acis.core.types import (
    BuildMode,
    BuildReport,
    JobHandle,
    Limits,
    RefSpec,
    Snapshot,
    SourceSpec,
    VersionComparison,
)
from acis.ingest.sources import VersionUnits, open_source
from acis.obs.log import get_logger
from acis.store import catalog, layout, selectors, snapshots
from acis.store.cas import BlobStore

if TYPE_CHECKING:
    from acis.engine.core import SnapshotData

log = get_logger("acis.engine")

VECTORS_FILE = "vectors.npy"


class VersionedEngineMixin:
    """The Track B1 half of `AcisEngine`. Mixed in rather than inlined: `core.py` is the P0 path and stays readable."""

    # -- registration ------------------------------------------------------------------------------------------
    def register(self, repo_id: str, *, source_kind: str = "units") -> str:
        """Record a repository. Idempotent, and the only way one comes into existence."""
        with catalog.open_catalog() as db:
            return catalog.register_repo(db, repo_id, source_kind=source_kind)

    def versions(self, repo_id: str) -> list[dict[str, Any]]:
        """Every recorded version, oldest first, with the snapshot that answers it (or `None`).

        The unit count comes from the snapshot rather than the version row: a version is a name, a snapshot is
        the content, and only one of the two knows how much there is.
        """
        with catalog.open_catalog() as db:
            if not catalog.get_repo_exists(db, repo_id):
                raise NotFound(f"no repository {repo_id!r}")
            rows = catalog.rows_as_dicts(catalog.list_versions(db, repo_id))
            sizes = {str(s["snapshot_id"]): int(s["n_units"]) for s in catalog.list_snapshots(db, repo_id)}
        for row in rows:
            row["n_units"] = sizes.get(str(row.get("snapshot_id") or ""), 0)
        return rows

    def build_reports(self, repo_id: str) -> list[BuildReport]:
        """What each build of this repository cost, in build order. Kept in-process, for the demo and the tests."""
        return list(self._reports.get(repo_id, ()))  # type: ignore[attr-defined]

    # -- building --------------------------------------------------------------------------------------------------
    def ingest(self, source: SourceSpec, *, repo_id: str, limits: Limits | None = None) -> JobHandle:
        """Read every version of a source and build them in history order. Synchronous: the job is done when it returns.

        A background queue is the spec's answer for a repository with a long history (spec 04 §3); it is not needed
        for a demo-sized corpus, and a `JobHandle` that is already `done` keeps the API shape honest either way.
        """
        layout.safe_name(repo_id, what="repo id")
        job = JobHandle(job_id="job_" + uuid.uuid4().hex[:12], repo_id=repo_id, state="running")
        self.register(repo_id, source_kind=source.kind)
        with catalog.open_catalog() as db:
            catalog.record_job(db, job.job_id, repo_id=repo_id, state="running")

        built = 0
        try:
            for version in open_source(source, limits=limits).versions():
                self._build_version(repo_id, version)
                built += 1
        except BaseException as exc:
            with catalog.open_catalog() as db:
                catalog.record_job(db, job.job_id, repo_id=repo_id, state="failed", detail=str(exc)[:200])
            raise

        with catalog.open_catalog() as db:
            catalog.record_job(db, job.job_id, repo_id=repo_id, state="done", detail=f"{built} version(s)")
        return JobHandle(job_id=job.job_id, repo_id=repo_id, state="done", detail=f"{built} version(s)")

    def index(self, repo_id: str, *, refs: RefSpec | None = None, mode: BuildMode = "eager_heads") -> BuildReport:
        """Build the versions that have no snapshot yet. `eager_heads` builds the newest; `eager_all` builds all."""
        rows = self.versions(repo_id)
        pending = [r for r in rows if not r["snapshot_id"]]
        if refs and refs.refs:
            wanted = set(refs.refs)
            pending = [r for r in pending if r["label"] in wanted]
        if mode == "eager_heads":
            pending = pending[-1:]
        if not pending:
            raise IndexRequired(f"{repo_id} has nothing left to build", mode=mode)
        report: BuildReport | None = None
        for row in pending:
            report = self._rebuild_recorded_version(repo_id, str(row["label"]))
        assert report is not None
        return report

    def update_version(self, repo_id: str, delta: SourceSpec, *, expected_active: str | None = None) -> BuildReport:
        """Add (or rebuild) versions from `delta`, refusing if the active snapshot is not the expected one.

        `expected_active` is the whole concurrency story for P1: two writers working from the same view cannot both
        win, and the loser is told rather than silently overwritten (spec 04 §4).
        """
        self.register(repo_id, source_kind=delta.kind)
        report: BuildReport | None = None
        for version in open_source(delta).versions():
            report = self._build_version(repo_id, version, expected_active=expected_active)
            expected_active = None  # only the first activation of a batch checks the caller's view
        if report is None:
            raise InvalidInput("the delta contained no versions", repo=repo_id)
        return report

    def activate(self, repo_id: str, version: str) -> str:
        """Point a repository at a named version. The snapshot exists already, so this is a ref rename."""
        resolution = selectors.resolve(repo_id, version)
        snapshot_id = snapshots.activate(repo_id, resolution.snapshot_id)
        self._forget(repo_id)
        return snapshot_id

    def rollback(self, repo_id: str) -> str:
        """Re-point the active version at the previous snapshot. Instant: nothing is rebuilt (D11)."""
        restored = snapshots.rollback(repo_id)
        self._forget(repo_id)
        return restored

    def recover(self, repo_id: str) -> dict[str, Any]:
        """Startup recovery for one repository (spec 04 §3). Safe to call at any time."""
        self._forget(repo_id)
        return snapshots.recover(repo_id)

    # -- comparison --------------------------------------------------------------------------------------------------
    def compare_versions(self, repo_id: str, a: str, b: str, *, query: str | None = None) -> VersionComparison:
        """Unit-level difference between two versions, by content hash.

        Hash set-difference is the *certain* half of the answer (spec 04 §4): a key present on both sides with
        different bodies changed, one present on a single side was added or removed. Renames, moves and
        replacements need evidence and belong to the lineage cascade in Track B2 — guessing at them here would
        produce confident answers nobody could check.
        """
        left, right = self._units_by_key(repo_id, a), self._units_by_key(repo_id, b)
        added = tuple(sorted(set(right) - set(left)))
        removed = tuple(sorted(set(left) - set(right)))
        changed = tuple(sorted(k for k in set(left) & set(right) if left[k] != right[k]))

        effect: dict[str, Any] = {}
        if query:
            effect = {
                "a": self._ranking_for(repo_id, a, query),
                "b": self._ranking_for(repo_id, b, query),
                "query_len": len(query),
            }
        return VersionComparison(
            repo_id=repo_id, a=a, b=b, added=added, removed=removed, changed=changed, query_effect=effect
        )

    # -- opening a version -------------------------------------------------------------------------------------------
    def open_version(self, repo_id: str, selector: str | None = None) -> SnapshotData:
        """Resolve a selector and load that snapshot.

        Cached per process: opening one is real work, and a snapshot is immutable, so the cache can never be
        stale for a given id. Activation and rollback drop the cache for that repository, because what `latest`
        *means* has changed even though no snapshot has.
        """
        resolution = selectors.resolve(repo_id, selector)
        snapshot_id = resolution.snapshot_id
        cached: SnapshotData | None = self._loaded.get((repo_id, snapshot_id))  # type: ignore[attr-defined]
        if cached is not None:
            return cached
        data = self._load_snapshot(repo_id, snapshot_id, label=resolution.labels[0])
        self._loaded[(repo_id, snapshot_id)] = data  # type: ignore[attr-defined]
        return data

    # -- internals ---------------------------------------------------------------------------------------------------
    def _build_version(self, repo_id: str, version: VersionUnits, *, expected_active: str | None = None) -> BuildReport:
        started = time.perf_counter()
        store = BlobStore.open()
        record = snapshots.build_snapshot(
            repo_id=repo_id,
            label=version.label,
            units=list(version.units),
            store=store,
            config_hash=self.config_hash,  # type: ignore[attr-defined]
            source_id=version.source_id,
        )
        self._ensure_vectors(repo_id, record.snapshot_id, store)
        snapshots.activate(repo_id, record.snapshot_id, expected_active=expected_active)
        self._forget(repo_id)

        # "New" means content this store had never seen, counted by the CAS itself rather than by an encoder's
        # bookkeeping: it is true for every encoder, including one with no cache at all, and it is the partition
        # the embedding work actually follows (an already-stored body is an already-embedded body).
        new_units = record.n_new_blobs
        report = BuildReport(
            snapshot=self._snapshot_of(repo_id, record),
            units_total=record.n_units,
            units_new=new_units,
            units_reused=record.n_units - new_units,
            seconds=round(time.perf_counter() - started, 4),
        )
        self._reports.setdefault(repo_id, []).append(report)  # type: ignore[attr-defined]
        log.info(
            "version.built",
            repo=repo_id,
            label=version.label,
            snapshot=record.snapshot_id,
            units=record.n_units,
            new=new_units,
            seconds=report.seconds,
        )
        return report

    def _rebuild_recorded_version(self, repo_id: str, label: str) -> BuildReport:
        raise IndexRequired(
            f"version {label!r} of {repo_id} has no stored source to rebuild from; re-ingest it",
            hint="`acis ingest` records the units; `acis index` only builds versions it already has",
        )

    def _ensure_vectors(self, repo_id: str, snapshot_id: str, store: BlobStore) -> int:
        """Embed a snapshot's units and write the matrix beside it. Returns how many needed a forward pass.

        The count is the honest one: it comes from the encoder's own counter, so a body that was embedded for an
        earlier version — or for another repository — is reported as reused rather than as work.
        """
        encoder = getattr(self, "encoder", None)
        path = layout.snapshot_dir(repo_id, snapshot_id) / VECTORS_FILE
        if encoder is None or path.is_file():
            return 0

        units = snapshots.read_units(repo_id, snapshot_id)
        texts = [store.get(u.body_hash) for u in units]
        before = int(getattr(encoder, "rows_encoded", 0))
        vectors = self._embed_documents(texts)  # type: ignore[attr-defined]
        after = int(getattr(encoder, "rows_encoded", 0))

        # Written atomically beside an already-sealed snapshot: the manifest and the marker cover the *content*,
        # and a half-written matrix would otherwise be indistinguishable from a complete one.
        tmp = path.with_name(f".{VECTORS_FILE}.tmp-{uuid.uuid4().hex[:8]}")
        # An open handle rather than a path: `np.save` appends `.npy` to a name that lacks it, which would leave
        # the temporary file under a name the rename never touches.
        with open(tmp, "wb") as fh:
            np.save(fh, np.asarray(vectors, dtype=np.float32))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        return (after - before) if after > before else (0 if after else len(units))

    def _load_snapshot(self, repo_id: str, snapshot_id: str, *, label: str) -> SnapshotData:
        from acis.engine.core import SnapshotData  # noqa: PLC0415 — the seam, not a cycle in practice
        from acis.lexical.bm25 import Bm25Index  # noqa: PLC0415
        from acis.lexical.tokenize import corpus_text  # noqa: PLC0415

        manifest = snapshots.open_snapshot(repo_id, snapshot_id)  # refuses anything not VALID (INV-9)
        units = snapshots.read_units(repo_id, snapshot_id)
        store = BlobStore.open()
        texts = {u.body_hash: store.get(u.body_hash) for u in units}

        doc_ids = [u.key for u in units]
        body_hashes = [u.body_hash for u in units]
        lexical = Bm25Index.build(
            doc_ids,
            [corpus_text("", texts[h]) for h in body_hashes],
            k1=float(self.config.get("lexical.k1", 1.5)),  # type: ignore[attr-defined]
            b=float(self.config.get("lexical.b", 0.75)),  # type: ignore[attr-defined]
            stemmer_language=self.config.get("lexical.stemmer", "english"),  # type: ignore[attr-defined]
        )

        vectors = None
        path = layout.snapshot_dir(repo_id, snapshot_id) / VECTORS_FILE
        if path.is_file():
            loaded = np.load(path, mmap_mode="r")
            if loaded.shape[0] != len(units):
                raise IndexRequired(
                    "this snapshot's vectors do not match its units; rebuild it",
                    snapshot=snapshot_id,
                    rows=int(loaded.shape[0]),
                    units=len(units),
                )
            vectors = np.asarray(loaded, dtype=np.float32)

        missing: tuple[str, ...] = () if vectors is not None else ("dense",)
        if lexical.vocabulary_empty:
            missing = (*missing, "lexical")
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            repo_id=repo_id,
            version_id=label,
            n_units=len(units),
            config_hash=str(manifest.get("config_hash", "")),
            source=str(manifest.get("source_id", "")),
            state="VALID",
            missing_channels=missing,
            created_ts=float(manifest.get("created_ts", 0.0)),
        )
        return SnapshotData(
            snapshot=snapshot,
            doc_ids=tuple(doc_ids),
            body_hashes=tuple(body_hashes),
            store=texts,
            ordinal={key: i for i, key in enumerate(doc_ids)},
            hash_of=dict(zip(doc_ids, body_hashes, strict=True)),
            lexical=lexical,
            vectors=vectors,
            missing=missing,
        )

    def _snapshot_of(self, repo_id: str, record: snapshots.SnapshotRecord) -> Snapshot:
        return Snapshot(
            snapshot_id=record.snapshot_id,
            repo_id=repo_id,
            version_id=record.label,
            n_units=record.n_units,
            config_hash=record.config_hash,
            source=record.tree_hash,
            state="VALID",
            created_ts=time.time(),
        )

    def _units_by_key(self, repo_id: str, selector: str) -> Mapping[str, str]:
        resolution = selectors.resolve(repo_id, selector)
        return {u.key: u.body_hash for u in snapshots.read_units(repo_id, resolution.snapshot_id)}

    def _ranking_for(self, repo_id: str, selector: str, query: str, *, top_k: int = 10) -> Sequence[tuple[str, float]]:
        data = self.open_version(repo_id, selector)
        ranked: list[tuple[str, float]] = self._rank_one(data, query, top_k=top_k, strict=False)  # type: ignore[attr-defined]
        return ranked

    def _forget(self, repo_id: str) -> None:
        """Drop cached snapshots for one repository: what `latest` means has changed."""
        for key in [k for k in self._loaded if k[0] == repo_id]:  # type: ignore[attr-defined]
            self._loaded.pop(key, None)  # type: ignore[attr-defined]


__all__ = ["VECTORS_FILE", "VersionedEngineMixin"]
