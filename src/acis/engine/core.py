"""`AcisEngine` — the one engine behind every surface (docs/spec/06 §1, docs/spec/02 §4).

Phase 1 builds the skeleton that P0 needs and that Track B compiles against: content-addressed snapshots, an exact
dense channel, a per-snapshot BM25 channel, the degradation ladder, and the batch surface the mteb adapter uses.
Fusion, features and LTR are Phase 4 and are *absent*, not stubbed: `mode="hybrid"` raises rather than inventing an
ungated ranking.

Invariants implemented here:

* **INV-1** every `Hit.source` is re-read from the content store by `body_hash`; nothing is generated.
* **INV-2** results come only from the requested snapshot; cache keys contain `snapshot_id` and `config_hash`.
* **INV-3** a query's ranking never depends on the other queries in its batch: no per-batch statistics anywhere.
* **INV-4** external ids are opaque; only exact-duplicate ordering uses the corpus ordinal.
* **INV-6** same (snapshot, config, profile, threads) ⇒ same ranking.
* **INV-7** every fallback increments a counter and shows up in `degradations`; strict mode aborts instead.
* **INV-9** only VALID snapshots are searchable unless `allow_partial=True`.
* **INV-13** `agent_calls == 0` — there is no agent on this path.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from acis.core.config import FrozenConfig, freeze_config
from acis.core.errors import InvalidInput, NotFound, NotReady, SnapshotInvalid
from acis.core.hashing import hash_obj, sha256_text, short
from acis.core.numeric import resolve_threads
from acis.core.types import (
    BuildMode,
    BuildReport,
    Confidence,
    Diagnostics,
    EvalReport,
    EvalSpec,
    EvolveRequest,
    EvolveResponse,
    Hit,
    JobHandle,
    Limits,
    RefSpec,
    Route,
    SearchRequest,
    SearchResponse,
    Snapshot,
    SnapshotRef,
    Snippet,
    SourceSpec,
    Unit,
    VersionComparison,
)
from acis.embed.base import Encoder, exact_search
from acis.lexical.bm25 import Bm25Index
from acis.lexical.tokenize import corpus_text
from acis.obs.counters import Counters, degradation
from acis.prep.normalize import d1, is_empty_query, lexical_view, q1
from acis.prep.truncate import head_tail
from acis.prep.views import build_view, weak_marker_count

MAX_QUERY_CHARS = 16_000
MAX_TOP_K = 1000


@dataclass(slots=True)
class SnapshotData:
    """One immutable, content-addressed corpus view. Track B1 persists this; Phase 1 keeps it in memory."""

    snapshot: Snapshot
    doc_ids: tuple[str, ...]
    body_hashes: tuple[str, ...]
    store: dict[str, str]  # body_hash -> exact bytes we return as evidence (INV-1)
    ordinal: dict[str, int]
    hash_of: dict[str, str]
    lexical: Bm25Index | None = None
    vectors: np.ndarray | None = None
    missing: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def text_of(self, doc_id: str) -> str:
        """Evidence is always re-read from the store by hash — never carried along from the ranking (INV-1)."""
        body_hash = self.hash_of.get(doc_id)
        if body_hash is None:
            raise NotFound(f"document {doc_id!r} is not in snapshot {self.snapshot.snapshot_id}")
        return self.store[body_hash]

    def unit_of(self, doc_id: str) -> Unit:
        return Unit(
            unit_id=f"u_{short(self.hash_of[doc_id], 16)}",
            key=doc_id,
            body_hash=self.hash_of[doc_id],
            repo_id=self.snapshot.repo_id,
            version_id=self.snapshot.version_id,
            snapshot_id=self.snapshot.snapshot_id,
            n_bytes=len(self.store[self.hash_of[doc_id]].encode("utf-8")),
        )


DEFAULT_CONFIG: dict[str, object] = {
    "run": {"mode": "B", "strict": False, "device": "cpu", "threads": "auto_physical", "seed": 0},
    "prep": {
        "query": {"version": "q1", "view": "V0", "max_tokens": 1024, "head": 768, "tail": 256},
        "doc": {"version": "d1", "max_tokens": 1024, "head": 768, "tail": 256},
    },
    "lexical": {"k1": 1.5, "b": 0.75, "stemmer": "english"},
    "retrieve": {"dense_k": 100, "lexical_k": 30, "union_cap": 100, "top_k_out": 1000},
}


class AcisEngine:
    """The engine. Construct with `AcisEngine.from_config(...)`; everything else is a method on the frozen surface."""

    def __init__(self, config: FrozenConfig, *, encoder: Encoder | None = None) -> None:
        self.config = config
        self.encoder = encoder
        self.counters = Counters()
        self.threads = resolve_threads(config.get("run.threads", "auto_physical"))
        self._snapshots: dict[str, SnapshotData] = {}
        self._query_vector_cache: dict[str, np.ndarray] = {}

    # -- construction ------------------------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls, config: FrozenConfig | Mapping[str, object] | None = None, *, encoder: Encoder | None = None
    ) -> AcisEngine:
        if config is None:
            config = freeze_config(DEFAULT_CONFIG, source_path="<default>")
        elif not isinstance(config, FrozenConfig):
            config = freeze_config(config)
        return cls(config, encoder=encoder)

    @property
    def config_hash(self) -> str:
        return self.config.config_hash

    @property
    def model_fingerprint(self) -> str:
        return self.encoder.fingerprint if self.encoder is not None else "none"

    def _channels(self) -> tuple[bool, bool]:
        return self.encoder is not None, True

    # -- snapshots ---------------------------------------------------------------------------------------------
    def build_snapshot(self, docs: Sequence[Snippet], *, source: str) -> Snapshot:
        """Content-address the documents, build the lexical index and (when an encoder exists) the dense matrix."""
        if not docs:
            raise InvalidInput("cannot build a snapshot from zero documents")
        started = time.perf_counter()

        doc_ids: list[str] = []
        body_hashes: list[str] = []
        store: dict[str, str] = {}
        for snippet in docs:
            text = d1(snippet.text)
            body_hash = sha256_text(text)
            doc_ids.append(str(snippet.handle))
            body_hashes.append(body_hash)
            store.setdefault(body_hash, text)
        if len(set(doc_ids)) != len(doc_ids):
            raise InvalidInput("duplicate document handles in a snapshot", n=len(doc_ids), unique=len(set(doc_ids)))

        snapshot_id = "s_" + short(
            hash_obj({"config": self.config_hash, "docs": body_hashes, "ids": doc_ids, "source": source}), 16
        )
        hash_of = dict(zip(doc_ids, body_hashes, strict=True))
        ordinal = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        lexical = Bm25Index.build(
            doc_ids,
            [corpus_text("", store[h]) for h in body_hashes],
            k1=float(self.config.get("lexical.k1", 1.5)),
            b=float(self.config.get("lexical.b", 0.75)),
            stemmer_language=self.config.get("lexical.stemmer", "english"),
        )
        vectors = self._embed_documents([store[h] for h in body_hashes]) if self.encoder is not None else None
        missing: tuple[str, ...] = () if vectors is not None else ("dense",)

        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            repo_id="-",
            version_id="v0",
            n_units=len(doc_ids),
            config_hash=self.config_hash,
            source=source,
            state="VALID",
            missing_channels=missing,
            created_ts=time.time(),
        )
        self._snapshots[snapshot_id] = SnapshotData(
            snapshot=snapshot,
            doc_ids=tuple(doc_ids),
            body_hashes=tuple(body_hashes),
            store=store,
            ordinal=ordinal,
            hash_of=hash_of,
            lexical=lexical,
            vectors=vectors,
            missing=missing,
        )
        self.counters.incr("snapshot.builds")
        self.counters.incr("snapshot.build_ms", int((time.perf_counter() - started) * 1000))
        return snapshot

    def snapshot_data(self, snap: Snapshot | str) -> SnapshotData:
        snapshot_id = snap if isinstance(snap, str) else snap.snapshot_id
        data = self._snapshots.get(snapshot_id)
        if data is None:
            raise NotFound(f"unknown snapshot {snapshot_id!r}")
        return data

    def _embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prep = self.config.section("prep").get("doc", {})
        prepared = [
            head_tail(
                t,
                max_tokens=int(prep.get("max_tokens", 1024)),
                head=int(prep.get("head", 768)),
                tail=int(prep.get("tail", 256)),
            ).text
            for t in texts
        ]
        assert self.encoder is not None
        return self.encoder.encode(prepared, is_query=False)

    # -- query preparation --------------------------------------------------------------------------------------
    def normalise_query(self, text: str) -> tuple[str, bool]:
        """q1 plus the hard input rules of spec 02 §6b. Returns `(normalised, truncated)`."""
        if text is None or not isinstance(text, str):
            raise InvalidInput("query must be a string")
        if is_empty_query(text):
            raise InvalidInput("query is empty")
        truncated = len(text) > MAX_QUERY_CHARS
        if truncated:
            keep = MAX_QUERY_CHARS // 2
            text = text[:keep] + text[-keep:]
        return q1(text), truncated

    def route(self, query: str) -> Route:
        """Routing v1.1 (spec 10 §4). Without the TRAIN-query OOD bank every query takes the generic path.

        That is the specified fallback, not a shortcut: `statement_like` requires an OOD score *and* bridge-feature
        availability, neither of which exists before the dense channel and the features do (Phases 2 and 4).
        """
        _ = weak_marker_count(query)  # a routing signal, recorded for Phase 4; never a filter on its own
        return "generic"

    def _query_vector(self, snapshot_id: str, query: str) -> np.ndarray:
        assert self.encoder is not None
        prep = self.config.section("prep").get("query", {})
        view = build_view(query, str(prep.get("view", "V0")))
        prepared = head_tail(
            view,
            max_tokens=int(prep.get("max_tokens", 1024)),
            head=int(prep.get("head", 768)),
            tail=int(prep.get("tail", 256)),
        ).text
        # INV-2: the cache key carries the snapshot and the config, so a vector can never cross either boundary.
        key = hash_obj(
            {
                "snapshot": snapshot_id,
                "config": self.config_hash,
                "model": self.model_fingerprint,
                "profile": self.config.numeric_profile,
                "text": prepared,
            }
        )
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            self.counters.incr("cache.qemb.hit")
            return cached
        self.counters.incr("cache.qemb.miss")
        vector: np.ndarray = np.asarray(self.encoder.encode([prepared], is_query=True)[0], dtype=np.float32)
        self._query_vector_cache[key] = vector
        return vector

    # -- channels ------------------------------------------------------------------------------------------------
    def _dense_ranking(self, data: SnapshotData, query: str) -> list[tuple[str, float]]:
        if data.vectors is None or self.encoder is None:
            raise NotReady("the dense channel is not available for this snapshot")
        vector = self._query_vector(data.snapshot.snapshot_id, query)
        scores = exact_search(vector.reshape(1, -1), data.vectors)[0]
        return self._stable_order(data, [(doc_id, float(scores[i])) for i, doc_id in enumerate(data.doc_ids)])

    @staticmethod
    def _stable_order(data: SnapshotData, ranking: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
        """Deterministic order that does not depend on how the corpus happens to be arranged.

        Equal scores are broken by **content hash** — derived from the document's bytes, not from its id or its
        position (INV-4) — so shuffling the corpus or relabelling documents cannot move a result. Only documents
        with *identical* content reach the last key, the corpus ordinal, which is the one place order may be used
        (D9): it gives exact duplicates distinct adjacent ranks.
        """
        return sorted(
            ranking,
            key=lambda item: (-item[1], data.hash_of.get(item[0], item[0]), data.ordinal.get(item[0], 1 << 30)),
        )

    def _lexical_ranking(self, data: SnapshotData, query: str, k: int) -> list[tuple[str, float]]:
        if data.lexical is None:
            raise NotReady("the lexical channel is not available for this snapshot")
        return data.lexical.search_one(lexical_view(query), k)

    # -- the batch surface the adapter uses -----------------------------------------------------------------------
    def search_batch(
        self,
        snap: Snapshot,
        ids: Sequence[str],
        texts: Sequence[str],
        *,
        top_k: int,
        restrict_to: Mapping[str, Sequence[str]] | None = None,
        strict: bool = False,
    ) -> dict[str, list[tuple[str, float]]]:
        """Rank every query independently. Batch composition never changes a single ranking (INV-3)."""
        if len(ids) != len(texts):
            raise InvalidInput("ids and texts must have the same length", n_ids=len(ids), n_texts=len(texts))
        data = self.snapshot_data(snap)
        if data.snapshot.state != "VALID":
            raise SnapshotInvalid(f"snapshot {data.snapshot.snapshot_id} is {data.snapshot.state}")
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        out: dict[str, list[tuple[str, float]]] = {}
        for qid, text in zip(ids, texts, strict=True):
            allowed = restrict_to.get(str(qid)) if restrict_to else None
            if allowed is None:
                out[str(qid)] = self._rank_one(data, text, top_k=top_k, strict=strict)
                continue
            # A reranking task hands us the candidate set: rank the whole corpus, then keep the candidates, so the
            # result still holds min(top_k, |candidates|) entries rather than whatever survived an early cut.
            allowed_set = {str(a) for a in allowed}
            full = self._rank_one(data, text, top_k=data.size, strict=strict)
            out[str(qid)] = [(d, s) for d, s in full if d in allowed_set][:top_k]
        return out

    def _rank_one(self, data: SnapshotData, text: str, *, top_k: int, strict: bool) -> list[tuple[str, float]]:
        query, _truncated = self.normalise_query(text)
        mode = str(self.config.get("run.channel", "auto"))
        dense_available, _ = self._channels()

        if mode in ("auto", "dense") and dense_available:
            ranking = self._dense_ranking(data, query)
        elif mode == "dense":
            raise NotReady("dense channel requested but no encoder is configured")
        elif mode == "lexical":
            ranking = self._lexical_ranking(data, query, data.size)
        elif mode == "hybrid":
            raise NotReady("channel fusion is decided by gate G2 in Phase 4; it does not exist yet")
        else:
            degradation("dense_unavailable", "lexical-only ranking", strict=strict, counters=self.counters)
            ranking = self._lexical_ranking(data, query, data.size)

        if len(ranking) < min(top_k, data.size):
            ranking = self._extend_with_corpus_order(data, ranking, min(top_k, data.size))
        return self._stable_order(data, ranking)[:top_k]

    def _extend_with_corpus_order(
        self, data: SnapshotData, ranking: Sequence[tuple[str, float]], want: int
    ) -> list[tuple[str, float]]:
        """A channel may return fewer than `min(top_k, N)` hits (BM25 with no matching term). The tail is the

        remaining documents in corpus order, scored below every retrieved document, so INV-10 still holds.
        """
        seen = {d for d, _ in ranking}
        out = list(ranking)
        floor = min((s for _, s in ranking), default=0.0)
        for doc_id in data.doc_ids:
            if len(out) >= want:
                break
            if doc_id not in seen:
                floor -= 1.0
                out.append((doc_id, floor))
        return out

    # -- the single-query surface ---------------------------------------------------------------------------------
    def search(self, req: SearchRequest) -> SearchResponse:
        """One query against one pinned snapshot, with evidence re-read from the content store (INV-1)."""
        started = time.perf_counter()
        if not self._snapshots:
            raise NotReady("no snapshot has been built yet")
        data = self._resolve_snapshot(req)
        if data.snapshot.state != "VALID" and not req.allow_partial:
            raise SnapshotInvalid(f"snapshot {data.snapshot.snapshot_id} is {data.snapshot.state} (INV-9)")

        t_norm = time.perf_counter()
        query, truncated = self.normalise_query(req.query)
        route = self.route(query)
        top_k = max(1, min(int(req.top_k), MAX_TOP_K))
        strict = bool(self.config.strict)

        t_rank = time.perf_counter()
        ranked = self._rank_one(data, req.query, top_k=top_k, strict=strict)
        hits = [
            Hit(
                rank=rank,
                score=float(score),
                unit=data.unit_of(doc_id),
                source=data.text_of(doc_id),  # INV-1: re-read by hash
                signals={"channel_score": float(score), "corpus_ordinal": float(data.ordinal[doc_id])},
            )
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]
        confidence, no_strong_match = self._confidence(ranked)
        total_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            snapshot=SnapshotRef(
                id=data.snapshot.snapshot_id,
                version=data.snapshot.version_id,
                complete=data.snapshot.complete,
                missing_channels=data.missing,
            ),
            results=hits,
            confidence=confidence,
            no_strong_match=no_strong_match,
            route=route,
            interpreted_intent={"query_truncated": truncated},
            timings_ms={
                "normalize": round((t_rank - t_norm) * 1000, 3),
                "rank": round((time.perf_counter() - t_rank) * 1000, 3),
                "total": round(total_ms, 3),
            },
            degradations=self.counters.degradations(),
        )

    def _resolve_snapshot(self, req: SearchRequest) -> SnapshotData:
        if req.version not in ("latest", "", None):
            for data in self._snapshots.values():
                if req.version in (data.snapshot.version_id, data.snapshot.snapshot_id):
                    return data
            raise NotFound(f"no snapshot for version {req.version!r}")
        return next(reversed(list(self._snapshots.values())))

    def _confidence(self, ranked: Sequence[tuple[str, float]]) -> tuple[Confidence, bool]:
        """A routing signal only; it never reorders anything (spec 02 §4 stage 11).

        Phase 1 reports a margin-derived band and marks it uncalibrated in `diagnostics`; the isotonic calibration
        of spec 02 §4 is fitted out-of-fold in Phase 4.
        """
        if len(ranked) < 2:
            return "low", not ranked
        top, second = float(ranked[0][1]), float(ranked[1][1])
        spread = abs(top) + 1e-9
        margin = (top - second) / spread
        if margin >= 0.25:
            return "high", False
        if margin >= 0.05:
            return "medium", False
        return "low", True

    # -- introspection ---------------------------------------------------------------------------------------------
    def diagnostics(self, repo_id: str | None = None) -> Diagnostics:
        from acis.cli.doctor import collect  # noqa: PLC0415 — hardware probing is not on the hot path

        hardware = collect(with_gemm=False)["hardware"]
        return Diagnostics(
            config_hash=self.config_hash,
            model_fingerprint=self.model_fingerprint,
            numeric_profile=self.config.numeric_profile,
            hardware=hardware,
            tier=str(collect(with_gemm=False)["tier"]),
            degradations=dict(self.counters.snapshot()),
            snapshots=tuple(
                {
                    "id": d.snapshot.snapshot_id,
                    "version": d.snapshot.version_id,
                    "units": d.size,
                    "state": d.snapshot.state,
                    "missing_channels": list(d.missing),
                }
                for d in self._snapshots.values()
                if repo_id is None or d.snapshot.repo_id == repo_id
            ),
            agent_calls=0,  # INV-13: there is no agent on this path
        )

    def evaluate(self, spec: EvalSpec) -> EvalReport:
        """CLI-only, never exposed on the network API (docs/spec/06 §1)."""
        from acis.eval import ladder  # noqa: PLC0415

        return ladder.run_spec(self, spec)

    # -- Track B surface (frozen signatures; built in B1/B2) -------------------------------------------------------
    def ingest(self, source: SourceSpec, *, repo_id: str, limits: Limits | None = None) -> JobHandle:
        raise NotReady("ingest is built in Track B1 (docs/spec/04)")

    def index(self, repo_id: str, *, refs: RefSpec | None = None, mode: BuildMode = "eager_heads") -> BuildReport:
        raise NotReady("index is built in Track B1 (docs/spec/04)")

    def update_version(self, repo_id: str, delta: SourceSpec, *, expected_active: str | None = None) -> BuildReport:
        raise NotReady("update_version is built in Track B1 (docs/spec/04)")

    def compare_versions(self, repo_id: str, a: str, b: str, *, query: str | None = None) -> VersionComparison:
        raise NotReady("compare_versions is built in Track B1 (docs/spec/04)")

    def search_version(self, repo_id: str, version: str, query: str, **kw: object) -> SearchResponse:
        return self.search(SearchRequest(query=query, repo_id=repo_id, version=version, **kw))  # type: ignore[arg-type]

    def retrieve_evolution(self, req: EvolveRequest) -> EvolveResponse:
        raise NotReady("evolution-aware retrieval is built in Track B2 (docs/spec/04 §6)")


__all__ = ["DEFAULT_CONFIG", "MAX_QUERY_CHARS", "MAX_TOP_K", "AcisEngine", "SnapshotData"]
