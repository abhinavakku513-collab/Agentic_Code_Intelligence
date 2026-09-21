"""Reference skeleton: MTEB adapter for the ACI retrieval engine.

Design rules (all verified against mteb 2.0.5 / 2.12.30 / 2.21.0):
  * class name + base class + zero-arg constructor match the hackathon sample;
  * index()+search() BOTH defined  -> mteb uses this object directly as the search model;
  * NEVER define predict()         -> that would route to the cross-encoder wrapper;
  * encode()+similarity() stay valid -> dense-only fallback if a harness forces the encoder path;
  * mteb_model_meta is a real ModelMeta (never None);
  * search() returns strictly decreasing scores (no ties) for every query id.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


class Engine(Protocol):
    """What the adapter needs from the (MTEB-agnostic) engine."""
    revision: str
    def embed(self, texts: Sequence[str], *, role: str, batch_size: int) -> np.ndarray: ...
    def build_snapshot(self, ids: Sequence[str], texts: Sequence[str], *, batch_size: int) -> Any: ...
    def search_batch(self, snapshot: Any, queries: Sequence[str], *, top_k: int,
                     restrict: Mapping[str, Sequence[str]] | None = None) -> list[list[tuple[str, float]]]: ...


def make_model_meta(name: str, revision: str, **extra: Any) -> ModelMeta:
    """mteb>=2.12 has ModelMeta.create_empty(); 2.0.x does not (all 17 fields are required there)."""
    overwrites = {"name": name, "revision": revision, "similarity_fn_name": "cosine", **extra}
    if hasattr(ModelMeta, "create_empty"):
        return ModelMeta.create_empty(overwrites=overwrites)
    base = dict(loader=None, release_date=None, languages=["eng-Latn"], n_parameters=None, memory_usage_mb=None,
                max_tokens=None, embed_dim=None, license=None, open_weights=True, public_training_code=None,
                public_training_data=None, framework=[], use_instructions=False, training_datasets=None)
    base.update(overwrites)
    return ModelMeta(**base)


def strictly_decreasing(ranked: list[tuple[str, float]], gap: float = 1e-5) -> dict[str, float]:
    """Emit scores whose order is OUR order under every scorer mteb uses.

    Verified pitfall: pytrec_eval (NDCG) treats score gaps below ~1e-6 as ties (float32-like resolution) and breaks
    ties by reverse doc-id order, while mteb's own MRR compares Python floats -> NDCG and MRR can silently disagree.
    Fix: squash to (1, 2) (float32 spacing ~1.2e-7 there, no underflow) and enforce an absolute gap >= 1e-5.
    Drift over 1000 results is <= 0.01, so the scale stays positive and the order is exact.
    """
    out: dict[str, float] = {}
    prev = float("inf")
    for doc_id, raw in ranked:
        raw = float(raw) if np.isfinite(raw) else -1e9
        s = 1.0 + 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))      # order-preserving map to (1, 2)
        s = min(s, prev - gap) if prev != float("inf") else s
        out[doc_id] = float(s)
        prev = s
    return out


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, engine: Engine | None = None) -> None:          # zero-arg friendly
        self._engine = engine
        self._snapshot: Any = None
        rev = getattr(engine, "revision", "unbuilt")
        self._meta = make_model_meta("aci/hybrid-retriever", rev)

    @property
    def mteb_model_meta(self) -> ModelMeta:                             # type: ignore[override]
        return self._meta

    @property
    def engine(self) -> Engine:
        if self._engine is None:                                        # lazy: heavy imports/model loads live in the engine
            raise RuntimeError("engine not configured; wire aci.engine.build_default_engine() here")
        return self._engine

    # ---- (1) SearchProtocol: the primary path --------------------------------------------------
    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None,
              encode_kwargs: Mapping[str, Any] | None = None, num_proc=None, **kwargs) -> None:
        ids = [str(i) for i in corpus["id"]]
        titles = corpus["title"] if "title" in corpus.column_names else [""] * len(ids)
        texts = [(f"{t}\n{x}" if t else x) for t, x in zip(titles, corpus["text"])]
        bs = int((encode_kwargs or {}).get("batch_size", 64))
        self._snapshot = self.engine.build_snapshot(ids, texts, batch_size=bs)

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None, top_k: int = 1000,
               encode_kwargs: Mapping[str, Any] | None = None, top_ranked=None, num_proc=None, **kwargs):
        qids = [str(i) for i in queries["id"]]
        ranked = self.engine.search_batch(self._snapshot, list(queries["text"]), top_k=top_k, restrict=top_ranked)
        return {qid: strictly_decreasing(r) for qid, r in zip(qids, ranked)}

    # ---- (2) dense-only fallback: only used if a harness forces SearchEncoderWrapper -------------
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type: PromptType | None = None, **kwargs) -> np.ndarray:
        texts: list[str] = []
        for batch in inputs:
            texts.extend(batch["text"])
        role = "query" if prompt_type == PromptType.query else "document"
        return np.asarray(self.engine.embed(texts, role=role, batch_size=int(kwargs.get("batch_size", 64))), dtype=np.float32)


# ---- artifact helpers ----------------------------------------------------------------------------
def _json_default(o: Any):
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_official_json(task_result, path: str | Path) -> Path:
    """The hackathon sample does json.dump(task_result.to_dict()); that raises on mteb 2.12+/2.21 (datetime)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=_json_default)
    return path