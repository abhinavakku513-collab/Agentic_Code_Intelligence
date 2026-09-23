"""The mteb adapter — translation only, no ranking logic (INV-11, docs/spec/03 §1–§2).

Two surfaces, one class:

* **Mode A** — `PrePostPipelineEncoder` defines `index()` and `search()`, so mteb's dispatch
  (`EncoderProtocol ∧ ¬SearchProtocol → wrapper; CrossEncoderProtocol → cross-encoder; SearchProtocol → direct`)
  uses the object **directly** and the full engine pipeline runs.
* **Mode B** — with `ACIS_MODE=B` the constructor returns `_EncoderSurface`, which has no `index`/`search`, so mteb
  wraps it in its own `SearchEncoderWrapper` and grades honest embeddings.

`predict()` is **never** defined: a class that has it is routed to the cross-encoder branch and our `search()` would
never run (docs/spec/01 V-02). The repository's guard hook blocks writing one into this file.

Mode A scores are rank-derived (D9), exactly `min(top_k, N)` per query (INV-10), and `search()` writes a run manifest
so a reader can prove our code ran: `adapter_invocations` must be 1.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder

from acis.core.config import FrozenConfig, load_frozen_config
from acis.core.errors import StrictViolation
from acis.core.paths import acis_root
from acis.core.types import Snippet
from acis.engine import AcisEngine
from acis.mteb_meta import make_model_meta, revision_for
from acis.rank.compose import assert_mode_a_contract, rank_derived_scores

MODE_ENV = "ACIS_MODE"
CONFIG_ENV = "ACIS_CONFIG"
RUN_DIR_ENV = "ACIS_RUN_DIR"
TOUCH_ENV = "ACIS_TEST_TOUCH_COUNT"
DRY_RUN_ENV = "ACIS_HARNESS_DRY_RUN"
MANIFEST_NAME = "manifest.json"


def doc_text(title: str | None, text: str) -> str:
    """mteb's document convention: `"{title} {text}"` when a title exists, else the text (docs/spec/03 §2)."""
    return f"{title} {text}".strip() if title else text


def _json_default(obj: Any) -> Any:
    """Datetime-safe JSON. `TaskResult.to_dict()` carries a `date`, which plain `json.dump` cannot serialise (V-07)."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    raise TypeError(f"{type(obj).__name__} is not JSON-serialisable")


def write_official_json(task_result: Any, path: str | Path) -> Path:
    """The official writer. The PDF sample's bare `json.dump` crashes on the result's `date` field (V-07)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(task_result.to_dict(), fh, indent=2, default=_json_default)
    return p


class _EncoderSurface(AbsEncoder):
    """Mode B: an honest dense encoder over the same model and the same preprocessing as Mode A."""

    def __init__(self, config_path: str | None = None) -> None:
        self.cfg: FrozenConfig = load_frozen_config(config_path or os.environ.get(CONFIG_ENV, "configs/dev.yaml"))
        self._engine: AcisEngine | None = None
        self._meta = make_model_meta(self.cfg)
        self.adapter_invocations = 0
        self.encode_calls = 0
        self._snapshot = None
        self._run_started = time.time()
        #: Held-out touches this run spends. The official script sets it; a dev run leaves it at zero.
        self.test_touch_count = int(os.environ.get(TOUCH_ENV, "0"))

    # `mteb_model_meta` is a class attribute on AbsEncoder; a property overrides it cleanly.
    @property
    def mteb_model_meta(self) -> Any:
        return self._meta

    @property
    def engine(self) -> AcisEngine:
        if self._engine is None:
            self._engine = AcisEngine.from_config(self.cfg, encoder=self._build_encoder())
        return self._engine

    def _build_encoder(self) -> Any:
        """Phase 1 wires the harness stand-in; Phase 2 replaces this with the real encoder runtime."""
        from acis.embed.hashing import HashingEncoder  # noqa: PLC0415

        name = str(self.cfg.get("model.encoder", "hashing"))
        if name != "hashing":
            raise StrictViolation(
                f"encoder {name!r} is not available yet: the dense runtime is built in Phase 2 (docs/spec/07)"
            )
        encoder = HashingEncoder(dim=int(self.cfg.get("model.dim", 4096)))
        if self.cfg.strict and not encoder.submission_capable and not self.harness_dry_run:
            raise StrictViolation(
                "the hashing stand-in encoder may not serve a strict run; it exists to validate the harness only"
            )
        return encoder

    @property
    def harness_dry_run(self) -> bool:
        """Explicit opt-out that lets the *official pipeline* be exercised on a fixture task (Phase 1).

        Without it the strict-mode interlock — which correctly refuses a stand-in encoder — would also make the
        official path impossible to test, leaving the single most consequential piece of code in the repository
        untested until the day it runs. The flag is written into the manifest and `verify-submission` fails any
        run that carries it, so it cannot quietly produce a submission.
        """
        return os.environ.get(DRY_RUN_ENV) == "1"

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        **kw: Any,
    ) -> np.ndarray:
        """Honest embeddings for mteb's own search wrapper.

        Mode B must use the **same model and the same preprocessing** as Mode A (docs/spec/03 §2.4) — otherwise the
        A-vs-B comparison measures the prep rather than the pipeline. Batch size never changes a vector (INV-3).
        """
        from mteb.types import PromptType  # noqa: PLC0415

        self.encode_calls += 1
        texts: list[str] = []
        for batch in inputs:
            texts.extend(list(batch["text"]))
        is_query = prompt_type == PromptType.query
        prepared = [self._prepare_query(t) if is_query else self._prepare_document(t) for t in texts]
        encoder = self.engine.encoder
        assert encoder is not None
        return np.asarray(
            encoder.encode(prepared, is_query=is_query, batch_size=int(kw.get("batch_size", 64))), dtype=np.float32
        )

    def _prepare_query(self, text: str) -> str:
        """Exactly the preparation Mode A applies, including the engine's own input rules.

        Going straight to `build_view` here would skip `q1` normalisation and the over-long-query rule that
        `AcisEngine.normalise_query` enforces, so Mode A and Mode B would encode subtly different text and the
        A-vs-B comparison (gate G-AB) would be measuring the preprocessing rather than the pipeline.
        """
        from acis.prep.truncate import head_tail  # noqa: PLC0415
        from acis.prep.views import build_view  # noqa: PLC0415

        normalised, _truncated = self.engine.normalise_query(text)
        prep = self.cfg.section("prep").get("query", {})
        view = build_view(normalised, str(prep.get("view", "V0")))
        return head_tail(
            view,
            max_tokens=int(prep.get("max_tokens", 1024)),
            head=int(prep.get("head", 768)),
            tail=int(prep.get("tail", 256)),
        ).text

    def _prepare_document(self, text: str) -> str:
        from acis.prep.normalize import d1  # noqa: PLC0415
        from acis.prep.truncate import head_tail  # noqa: PLC0415

        prep = self.cfg.section("prep").get("doc", {})
        return head_tail(
            d1(text),
            max_tokens=int(prep.get("max_tokens", 1024)),
            head=int(prep.get("head", 768)),
            tail=int(prep.get("tail", 256)),
        ).text

    # -- manifest ---------------------------------------------------------------------------------------------
    def run_dir(self) -> Path:
        return Path(os.environ.get(RUN_DIR_ENV) or (acis_root() / "runs" / "latest"))

    def manifest(self, task_metadata: Any = None, hf_split: str | None = None) -> dict[str, Any]:
        engine = self._engine
        return {
            "adapter_invocations": self.adapter_invocations,
            "encode_calls": self.encode_calls,
            "mode": "A" if isinstance(self, PrePostPipelineEncoder) else "B",
            "strict": bool(self.cfg.strict),
            "harness_dry_run": self.harness_dry_run,
            "agent_calls": 0,
            "fallbacks": sum(
                v for k, v in (engine.counters.snapshot().items() if engine else []) if k.startswith("degradation.")
            ),
            "degradations": list(engine.counters.degradations()) if engine else [],
            # How many held-out touches this run spends. A dev run spends none; an official A+B run spends two.
            "test_touch_count": self.test_touch_count,
            "config_hash": self.cfg.config_hash,
            "model_revision": revision_for(self.cfg),
            "model_fingerprint": engine.model_fingerprint if engine else "none",
            "snapshot_id": getattr(self._snapshot, "snapshot_id", None),
            "task": getattr(task_metadata, "name", None),
            "split": hf_split,
            "dataset_revision": (getattr(task_metadata, "dataset", {}) or {}).get("revision"),
            "started_ts": self._run_started,
            "written_ts": time.time(),
        }

    def write_run_manifest(self, task_metadata: Any = None, hf_split: str | None = None, *, suffix: str = "") -> Path:
        """Write the manifest. `suffix` keeps an A+B run's two manifests apart (`manifest.A.json`)."""
        name = MANIFEST_NAME if not suffix else MANIFEST_NAME.replace(".json", f".{suffix}.json")
        path = self.run_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.manifest(task_metadata, hf_split), indent=2, default=_json_default), encoding="utf-8"
        )
        return path


class PrePostPipelineEncoder(_EncoderSurface):
    """Mode A: the same class plus `index`/`search`, which makes it a `SearchProtocol` for mteb's dispatch."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if os.environ.get(MODE_ENV, "A").upper() == "B":
            return _EncoderSurface(*args, **kwargs)  # no index/search -> SearchEncoderWrapper path
        return super().__new__(cls)

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
        num_proc: int | None = None,
        **_: Any,
    ) -> None:
        ids, titles, texts = _corpus_columns(corpus)
        docs = [Snippet(handle=str(i), text=doc_text(t, x)) for i, t, x in zip(ids, titles, texts, strict=True)]
        source = f"mteb:{getattr(task_metadata, 'name', 'unknown')}:{hf_split}"
        self._snapshot = self.engine.build_snapshot(docs, source=source)

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        top_k: int = 1000,
        encode_kwargs: Mapping[str, Any] | None = None,
        top_ranked: Mapping[str, Sequence[str]] | None = None,
        num_proc: int | None = None,
        **_: Any,
    ) -> dict[str, dict[str, float]]:
        if self._snapshot is None:
            from acis.core.errors import IndexRequired  # noqa: PLC0415

            raise IndexRequired("search() was called before index()")
        self.adapter_invocations += 1

        qids = [str(i) for i in queries["id"]]
        qtexts = [str(t) for t in queries["text"]]
        ranked = self.engine.search_batch(
            self._snapshot, qids, qtexts, top_k=top_k, restrict_to=top_ranked, strict=bool(self.cfg.strict)
        )
        corpus_size = self.engine.snapshot_data(self._snapshot).size
        out: dict[str, dict[str, float]] = {}
        for qid, hits in ranked.items():
            # A reranking task restricts the candidate set, so the contract is min(top_k, |candidates|) there.
            available = len(top_ranked[qid]) if top_ranked and qid in top_ranked else corpus_size
            scores = rank_derived_scores(hits, top_k)
            assert_mode_a_contract(scores, top_k=top_k, corpus_size=available)
            out[qid] = scores
        self.write_run_manifest(task_metadata, hf_split)
        return out


def _corpus_columns(corpus: Any) -> tuple[list[str], list[str], list[str]]:
    """Read `id`/`title`/`text` from a datasets.Dataset or a plain mapping, without assuming a title column."""
    if hasattr(corpus, "column_names"):
        names = set(corpus.column_names)
        ids = [str(i) for i in corpus["id"]]
        titles = [str(t or "") for t in corpus["title"]] if "title" in names else [""] * len(ids)
        texts = [str(t) for t in corpus["text"]]
        return ids, titles, texts
    ids = [str(i) for i in corpus["id"]]
    titles = [str(t or "") for t in corpus.get("title", [""] * len(ids))]
    texts = [str(t) for t in corpus["text"]]
    return ids, titles, texts


__all__ = [
    "CONFIG_ENV",
    "DRY_RUN_ENV",
    "MANIFEST_NAME",
    "MODE_ENV",
    "RUN_DIR_ENV",
    "PrePostPipelineEncoder",
    "doc_text",
    "write_official_json",
]
