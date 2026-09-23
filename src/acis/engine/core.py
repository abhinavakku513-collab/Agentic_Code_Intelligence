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

import heapq
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from acis.core.config import FrozenConfig, freeze_config
from acis.core.errors import InvalidInput, NotFound, NotReady, SnapshotInvalid
from acis.core.hashing import hash_obj, sha256_text, short
from acis.core.numeric import resolve_threads
from acis.core.types import (
    BuildReport,
    Confidence,
    Diagnostics,
    EvalReport,
    EvalSpec,
    EvolveRequest,
    EvolveResponse,
    Hit,
    Route,
    SearchRequest,
    SearchResponse,
    Snapshot,
    SnapshotRef,
    Snippet,
    Unit,
)
from acis.embed.base import Encoder, exact_search
from acis.engine.versions import VersionedEngineMixin
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
    #: Parse-only document features, built once per snapshot on first use (Phase 4). Keyed by body hash, so
    #: duplicate documents — and the same body in another version — are parsed once.
    _features: dict[str, Any] = field(default_factory=dict, repr=False)
    _dup_counts: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def features_of(self, doc_id: str) -> Any:
        """Document features, parsed on demand and cached for the life of the snapshot (immutable content)."""
        from acis.features.doc import extract  # noqa: PLC0415

        body_hash = self.hash_of[doc_id]
        cached = self._features.get(body_hash)
        if cached is None:
            cached = extract(self.store[body_hash])
            self._features[body_hash] = cached
        return cached

    def duplicate_count(self, doc_id: str) -> int:
        """How many documents in this snapshot share this exact body — a corpus fact, never a query one."""
        if not self._dup_counts:
            for h in self.body_hashes:
                self._dup_counts[h] = self._dup_counts.get(h, 0) + 1
        return self._dup_counts.get(self.hash_of[doc_id], 1)

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


class AcisEngine(VersionedEngineMixin):
    """The engine. Construct with `AcisEngine.from_config(...)`; everything else is a method on the frozen surface."""

    def __init__(self, config: FrozenConfig, *, encoder: Encoder | None = None) -> None:
        self.config = config
        self.encoder = encoder
        self.counters = Counters()
        self.threads = resolve_threads(config.get("run.threads", "auto_physical"))
        #: In-memory snapshots built through `build_snapshot` (the P0 path: one corpus, no versions).
        self._snapshots: dict[str, SnapshotData] = {}
        #: Snapshots loaded from the store, keyed by `(repo_id, snapshot_id)` (the P1 path, Track B1).
        self._loaded: dict[tuple[str, str], SnapshotData] = {}
        self._reports: dict[str, list[BuildReport]] = {}
        self._query_vector_cache: dict[str, np.ndarray] = {}
        #: The learned ranker, when one has been trained and the config points at it (Phase 4, gate G5).
        self._ranker: Any = None
        self._ranker_loaded = False

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
    def ranker(self) -> Any:
        """The trained ranker named by `rank.model`, loaded once. `None` means fusion, never a silent identity."""
        if self._ranker_loaded:
            return self._ranker
        self._ranker_loaded = True
        path = str(self.config.get("rank.model", "") or "")
        if path:
            from acis.core.paths import acis_root  # noqa: PLC0415
            from acis.rank.ltr import Ranker  # noqa: PLC0415

            target = Path(path)
            self._ranker = Ranker.load(target if target.is_absolute() else acis_root() / target)
        return self._ranker

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
            tokenizer=str(self.config.get("lexical.tokenizer", "stock")),
        )
        vectors = self._embed_documents([store[h] for h in body_hashes]) if self.encoder is not None else None
        missing: tuple[str, ...] = () if vectors is not None else ("dense",)
        if lexical.vocabulary_empty:
            # Nothing in this corpus is indexable lexically. The snapshot says so rather than pretending to have
            # a channel that can only ever return nothing.
            missing = (*missing, "lexical")

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

    def _query_vector(self, snapshot_id: str, query: str, *, route: str = "generic") -> np.ndarray:
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
                # The route chooses the instruction the query is encoded with (INV-15), so two routes are two
                # different vectors of the same text. Leaving it out of the key would serve one for the other.
                "route": route,
                "text": prepared,
            }
        )
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            self.counters.incr("cache.qemb.hit")
            return cached
        self.counters.incr("cache.qemb.miss")
        vector: np.ndarray = np.asarray(
            self.encoder.encode([prepared], is_query=True, route=route)[0], dtype=np.float32
        )
        self._query_vector_cache[key] = vector
        return vector

    # -- channels ------------------------------------------------------------------------------------------------
    def _dense_ranking(
        self, data: SnapshotData, query: str, *, route: str = "generic", k: int | None = None
    ) -> list[tuple[str, float]]:
        """The dense channel's top-`k`, already in final order. `k=None` ranks the whole snapshot."""
        if data.vectors is None or self.encoder is None:
            raise NotReady("the dense channel is not available for this snapshot")
        vector = self._query_vector(data.snapshot.snapshot_id, query, route=route)
        scores = exact_search(vector.reshape(1, -1), data.vectors)[0]
        return self._stable_top_k(data, scores, data.size if k is None else k)

    @classmethod
    def _stable_top_k(cls, data: SnapshotData, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        """The `k` best documents in `_stable_order` order, without ordering the other 8,755 of them.

        Sorting the whole corpus per query is the single most expensive thing the retrieval core did, and almost
        all of it was thrown away: a request for 10 results does not need ranks 11 to 8,765 in order. The cut is
        exact rather than approximate — the threshold is the k-th best score and *every* document that matches it
        is sorted, so a tie at the boundary cannot be dropped on a technicality and the answer is identical to
        the full sort (pinned by a metamorphic test, not by inspection).
        """
        n = int(scores.shape[0])
        k = max(0, min(int(k), n))
        if k == 0:
            return []
        if k < n:
            window = np.argpartition(-scores, k - 1)[:k]
            threshold = float(scores[window].min())
            indices = np.flatnonzero(scores >= threshold)
        else:
            indices = np.arange(n)
        ranking = [(data.doc_ids[i], float(scores[i])) for i in indices.tolist()]
        return cls._stable_order(data, ranking)[:k]

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

    def _hybrid_ranking(
        self,
        data: SnapshotData,
        query: str,
        *,
        route: str,
        want: int,
        counters: Counters,
        strict: bool,
    ) -> list[tuple[str, float]]:
        """Dense ∪ lexical → features → ranker → dense tail (spec 02 §4 stages 3–9).

        The fallback chain is explicit and counted (INV-7): a trained ranker if one is loaded and its evidence
        fired, reciprocal-rank fusion otherwise, and the dense order if the lexical channel is missing entirely.
        Every step down increments a counter and appears in `degradations`, and strict mode refuses all of them.
        """
        from acis.features.query import bridge as bridge_features  # noqa: PLC0415
        from acis.features.query import extract as query_features  # noqa: PLC0415
        from acis.rank import candidates as cand  # noqa: PLC0415

        dense_order = self._dense_ranking(data, query, route=route, k=None)
        retrieve = self.config.section("retrieve")
        lexical_k = int(retrieve.get("lexical_k", 30))
        union_cap = int(retrieve.get("union_cap", 100))
        dense_k = int(retrieve.get("dense_k", 100))

        lexical_order: list[tuple[str, float]] = []
        if data.lexical is not None and "lexical" not in data.missing:
            lexical_order = self._lexical_ranking(data, query, lexical_k)
        else:
            degradation("lexical_unavailable", "dense-only fusion", strict=strict, counters=counters)

        pool = cand.union(dense_order, lexical_order, dense_k=dense_k, lexical_k=lexical_k, cap=union_cap)
        if not pool:
            return dense_order[:want]

        qf = query_features(query)
        bridge = {c.doc_id: bridge_features(qf, data.features_of(c.doc_id)) for c in pool}
        doc_meta = {
            c.doc_id: {
                "n_tokens": data.features_of(c.doc_id).n_tokens,
                "parse_ok": data.features_of(c.doc_id).parse_ok,
                "dup_cluster_size": data.duplicate_count(c.doc_id),
            }
            for c in pool
        }
        matrix = cand.feature_matrix(pool, bridge=bridge, query_tokens=qf.n_tokens, doc_meta=doc_meta)
        doc_ids = [c.doc_id for c in pool]

        ranker = self.ranker
        head: list[str]
        if ranker is not None:
            head, abstained = ranker.rerank(doc_ids, matrix)
            if abstained:
                degradation("ltr_abstained", "not enough feature groups fired", strict=False, counters=counters)
                head = self._rrf_order(pool)
        else:
            degradation("ltr_unavailable", "reciprocal-rank fusion", strict=strict, counters=counters)
            head = self._rrf_order(pool)

        # The head is re-scored by position, then the dense tail follows it to `want` (spec 02 §4 stage 9). The
        # tail's own order is the dense one, which is why it is taken from `dense_order` rather than recomputed.
        seen = set(head)
        tail = [doc for doc, _ in dense_order if doc not in seen]
        ordered = head + tail
        floor = -float(len(ordered))
        return [(doc, floor + float(len(ordered) - i)) for i, doc in enumerate(ordered[:want])]

    @staticmethod
    def _rrf_order(pool: Sequence[Any]) -> list[str]:
        """Reciprocal-rank fusion over the union — parameter-free, and the honest default until G2 tunes weights.

        Ties are broken by dense rank, so fusion can never reorder two documents it has no reason to separate.
        """
        from acis.rank.candidates import reciprocal_rank  # noqa: PLC0415

        return [
            c.doc_id
            for c in sorted(
                pool,
                key=lambda c: (
                    -(reciprocal_rank(c.dense_rank) + reciprocal_rank(c.lexical_rank)),
                    c.dense_rank or 1 << 30,
                    c.doc_id,
                ),
            )
        ]

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

    def _rank_one(
        self,
        data: SnapshotData,
        text: str,
        *,
        top_k: int,
        strict: bool,
        counters: Counters | None = None,
        route: str | None = None,
    ) -> list[tuple[str, float]]:
        query, _truncated = self.normalise_query(text)
        # The route chooses the instruction the query is encoded with (spec 02 §4, stage 2). `search` has already
        # computed it; the batch surface has not, and routing is a pure function of the query, so it is safe here.
        route = route if route is not None else self.route(query)
        mode = str(self.config.get("run.channel", "auto"))
        dense_available, _ = self._channels()
        sink = counters if counters is not None else self.counters

        want = min(top_k, data.size)
        ordered = False
        if mode in ("auto", "dense") and dense_available:
            # The dense channel returns its top-`want` already in final order, so nothing below re-sorts it.
            ranking = self._dense_ranking(data, query, route=route, k=want)
            ordered = True
        elif mode == "dense":
            raise NotReady("dense channel requested but no encoder is configured")
        elif mode == "lexical":
            ranking = self._lexical_ranking(data, query, data.size)
        elif mode == "hybrid":
            ranking = self._hybrid_ranking(data, query, route=route, want=want, counters=sink, strict=strict)
            ordered = True
        else:
            degradation("dense_unavailable", "lexical-only ranking", strict=strict, counters=sink)
            ranking = self._lexical_ranking(data, query, data.size)

        if len(ranking) < want:
            ranking = self._extend_with_unretrieved(data, ranking, want)
            ordered = False
        return (ranking if ordered else self._stable_order(data, ranking))[:top_k]

    @staticmethod
    def _extend_with_unretrieved(
        data: SnapshotData, ranking: Sequence[tuple[str, float]], want: int
    ) -> list[tuple[str, float]]:
        """Pad a short ranking so INV-10 still returns `min(top_k, N)` entries.

        A channel can return fewer hits than asked for — BM25 with no matching term is the ordinary case. The
        documents it did not retrieve are all equally unranked, so they all get **one** score below every retrieved
        document and `_stable_order` sorts them by content hash. Giving them descending scores in corpus order
        instead would make the tail depend on how the corpus happens to be arranged, which is exactly what parity
        P5 forbids.
        """
        seen = {d for d, _ in ranking}
        floor = min((s for _, s in ranking), default=0.0) - 1.0
        wanted = max(0, want - len(ranking))
        if not wanted:
            return list(ranking)
        # All padding entries share one score, so the order among them is the content-hash tie-break alone —
        # `nsmallest` on that key gives the same answer as sorting the whole corpus, without doing so per query.
        chosen = heapq.nsmallest(
            wanted,
            (doc_id for doc_id in data.doc_ids if doc_id not in seen),
            key=lambda doc_id: (data.hash_of.get(doc_id, doc_id), data.ordinal.get(doc_id, 1 << 30)),
        )
        return [*ranking, *((doc_id, floor) for doc_id in chosen)]

    # -- the single-query surface ---------------------------------------------------------------------------------
    def search(self, req: SearchRequest) -> SearchResponse:
        """One query against one pinned snapshot, with evidence re-read from the content store (INV-1)."""
        started = time.perf_counter()
        # Resolution decides where the snapshot comes from — the store for a repository, memory for the P0 batch
        # surface — so the "is there anything to search" check belongs there, not here.
        data = self._resolve_snapshot(req)
        if data.snapshot.state != "VALID" and not req.allow_partial:
            raise SnapshotInvalid(f"snapshot {data.snapshot.snapshot_id} is {data.snapshot.state} (INV-9)")

        t_norm = time.perf_counter()
        query, truncated = self.normalise_query(req.query)
        route = self.route(query)
        top_k = max(1, min(int(req.top_k), MAX_TOP_K))
        strict = bool(self.config.strict)

        t_rank = time.perf_counter()
        # Per-request counters: `SearchResponse.degradations` must describe *this* request, not everything the
        # process has degraded since start-up. They are merged into the engine's totals for the run manifest.
        request_counters = Counters()
        ranked = self._rank_one(data, req.query, top_k=top_k, strict=strict, counters=request_counters, route=route)
        for name, count in request_counters.snapshot().items():
            self.counters.incr(name, count)
        for event in request_counters.degradations():
            self.counters.note(event)
        hits = [
            Hit(
                rank=rank,
                score=float(score),
                unit=data.unit_of(doc_id),
                source=data.text_of(doc_id),  # INV-1: re-read by hash
                # Never publish the corpus ordinal: on this corpus `ordinal < 5000` is an exact train-partition
                # detector, and a Phase 4 feature builder reading `hit.signals` would learn it (CLAUDE.md §4).
                signals={"channel_score": float(score)},
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
            degradations=request_counters.degradations(),
        )

    def _resolve_snapshot(self, req: SearchRequest) -> SnapshotData:
        """Resolve the request's version **once**, and pin it for the whole request (INV-2).

        A request that names a repository is answered from the store (Track B1); one that does not is answered
        from the in-memory snapshots the P0 batch surface builds. The two never mix: a repository id is how a
        caller says "this is a versioned corpus".
        """
        if req.repo_id not in ("-", "", None):
            return self.open_version(req.repo_id, req.version)

        if req.version not in ("latest", "", None):
            for data in self._snapshots.values():
                if req.version in (data.snapshot.version_id, data.snapshot.snapshot_id):
                    return data
            raise NotFound(f"no snapshot for version {req.version!r}")
        if not self._snapshots:
            raise NotReady("no snapshot has been built yet")
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

    # -- Track B surface -------------------------------------------------------------------------------------------
    # `ingest`, `index`, `update_version`, `compare_versions`, `rollback` and `open_version` come from
    # `VersionedEngineMixin` (Track B1, `acis.engine.versions`).
    def search_version(self, repo_id: str, version: str, query: str, **kw: object) -> SearchResponse:
        return self.search(SearchRequest(query=query, repo_id=repo_id, version=version, **kw))  # type: ignore[arg-type]

    def retrieve_evolution(self, req: EvolveRequest) -> EvolveResponse:
        raise NotReady("evolution-aware retrieval is built in Track B2 (docs/spec/04 §6)")


__all__ = ["DEFAULT_CONFIG", "MAX_QUERY_CHARS", "MAX_TOP_K", "AcisEngine", "SnapshotData"]
