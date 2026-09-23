"""The dense encoder runtime (docs/spec/02 §3, D4, D15, INV-3).

This is the expensive step — the cold official pass is ~3M tokens through it — so the design is about keeping the
work honest and the cost visible rather than about clever tricks:

* **Offline and pinned.** `HF_HUB_OFFLINE=1`, safetensors only, never `trust_remote_code`. The weights that load
  are whatever is on disk, so the model's identity is *verified* (per-file SHA-256 from the card) rather than
  assumed, and the `model_fingerprint` that keys every cached vector is derived from the weights actually loaded.
* **Batch invariance (INV-3).** Batching is a cost optimisation and must never be observable in a result. Vectors
  are computed under `torch.inference_mode()`, in length-sorted token-budget batches, and restored to input order;
  the same text encodes to the same vector whatever it was batched with. The test for that runs the same inputs
  through wildly different batch budgets and demands bit-identical output.
* **The cache is part of the runtime, not bolted on.** A re-run, an incremental version bump (P1) and a warm
  official pass all lean on it, so a miss is the only thing that costs a forward pass.

The model itself is injected rather than constructed here (`Backend`), which is what makes the loop testable
without weights: the batching, ordering, caching, pooling and normalisation are all exercised against a stub, so
when real weights arrive the only untested thing is the model.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from acis.core.errors import InvalidInput, NotReady
from acis.core.hashing import hash_obj, sha256_file, sha256_text
from acis.core.numeric import REFERENCE_PROFILE, apply_threads, get_profile, resolve_threads
from acis.embed.base import l2_normalize
from acis.embed.batching import DEFAULT_TOKEN_BUDGET, plan_batches
from acis.embed.cache import VectorCache, vector_key
from acis.embed.pooling import pool
from acis.embed.registry import ModelCard, verify_pinned_files
from acis.obs.log import get_logger

log = get_logger("acis.embed")

OFFLINE_ENV = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


class Backend(Protocol):
    """The minimum a transformer needs to expose. Injected so the loop can be tested without weights."""

    def tokenize(self, texts: Sequence[str], *, max_length: int) -> Any: ...

    def forward(self, batch: Any) -> Any: ...

    @property
    def dim(self) -> int: ...

    @property
    def weights_digest(self) -> str: ...


def enforce_offline() -> None:
    """D15: no query-time network. Set before any model or tokenizer is constructed."""
    for name in OFFLINE_ENV:
        os.environ[name] = "1"


@dataclass(slots=True)
class TransformersBackend:
    """A `transformers` model loaded offline from a pinned directory. Constructed only when weights exist."""

    card: ModelCard
    model_dir: Path
    profile: str = REFERENCE_PROFILE
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)
    _digest: str = ""

    def __post_init__(self) -> None:
        enforce_offline()
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        problems = verify_pinned_files(self.card, self.model_dir)
        if problems and os.environ.get("ACIS_ALLOW_UNPINNED_MODEL") != "1":
            raise InvalidInput(
                "the weights on disk do not match the card's pins; refusing to encode with an unidentified model",
                model=self.card.name,
                problems=problems[:3],
            )
        dtype = getattr(torch, DTYPES.get(get_profile(self.profile).dtype, "float32"))
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        )
        self._tokenizer.padding_side = self.card.padding_side
        self._model = AutoModel.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, dtype=dtype
        )
        self._model.eval()
        self._digest = _weights_digest(self.model_dir)

    def tokenize(self, texts: Sequence[str], *, max_length: int) -> Any:
        return self._tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    def forward(self, batch: Any) -> Any:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            output = self._model(**batch)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise NotReady(f"model {self.card.name} returned no last_hidden_state")
        return pool(hidden, batch["attention_mask"], strategy=self.card.pooling, padding_side=self.card.padding_side)

    @property
    def dim(self) -> int:
        return int(self._model.config.hidden_size)

    @property
    def weights_digest(self) -> str:
        return self._digest


def _weights_digest(model_dir: Path) -> str:
    """Hash of the files that decide what the model computes — the basis of `model_fingerprint`."""
    directory = Path(model_dir)
    interesting = sorted(
        p for p in directory.glob("*") if p.suffix in (".safetensors", ".json", ".model", ".txt") and p.is_file()
    )
    return hash_obj({p.name: sha256_file(p) for p in interesting}) if interesting else sha256_text(str(directory))


@dataclass(slots=True)
class EncoderRuntime:
    """The `Encoder` the engine talks to: prep-aware, cached, batch-invariant."""

    card: ModelCard
    backend: Backend
    profile: str = REFERENCE_PROFILE
    cache: VectorCache | None = None
    prep_hash: str = ""
    token_budget: int = DEFAULT_TOKEN_BUDGET
    threads: int = 0
    forward_calls: int = 0
    rows_encoded: int = 0

    def __post_init__(self) -> None:
        self.threads = apply_threads(self.threads or resolve_threads("auto_physical"))

    # -- identity -------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.card.name

    @property
    def dim(self) -> int:
        return self.backend.dim

    @property
    def submission_capable(self) -> bool:
        """A real model may ship; the harness stand-in may not. Strict runs check this."""
        return True

    @property
    def fingerprint(self) -> str:
        """Weights + tokenizer + card + profile. Everything that changes a vector is in the cache key."""
        return hash_obj(
            {
                "weights": self.backend.weights_digest,
                "card": self.card.fingerprint,
                "profile": self.profile,
            }
        )

    # -- encoding -------------------------------------------------------------------------------------------
    def _render(self, text: str, *, is_query: bool, route: str) -> str:
        return self.card.format_query(text, route=route) if is_query else self.card.format_document(text)

    def _cache_key(self, rendered: str, *, is_query: bool, route: str) -> str:
        # A query's vector depends on the instruction it was rendered with, so the route is part of its identity.
        prompt = hash_obj({"route": route, "task": self.card.task_string(route)}) if is_query else ""
        return vector_key(
            model_fingerprint=self.fingerprint,
            numeric_profile=self.profile,
            prep_hash=self.prep_hash,
            text=rendered,
            prompt_hash=prompt,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool = False,
        batch_size: int = 64,
        route: str = "generic",
    ) -> np.ndarray:
        """Vectors for `texts`, in input order. `batch_size` is accepted for the mteb contract and ignored:

        batching is by token budget, which is what actually bounds the work (docs/spec/02 §3).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        rendered = [self._render(t, is_query=is_query, route=route) for t in texts]
        keys = [self._cache_key(r, is_query=is_query, route=route) for r in rendered]

        # Identical inputs share one vector by construction, so they must also share one forward pass. Without
        # this, a corpus with duplicate documents — APPS has 11 exact pairs — pays for each copy, and a P1 batch
        # that re-sends unchanged units pays for all of them.
        representatives: dict[str, int] = {}
        for index, key in enumerate(keys):
            representatives.setdefault(key, index)

        out: list[np.ndarray | None] = [None] * len(texts)
        todo = list(representatives.values())
        if self.cache is not None:
            found, missing = self.cache.get_many([keys[i] for i in todo])
            for position, vector in found.items():
                out[todo[position]] = vector
            todo = [todo[position] for position in missing]

        if todo:
            lengths = [self._token_length(rendered[i]) for i in todo]
            for batch in plan_batches(lengths, token_budget=self.token_budget):
                rows = [todo[i] for i in batch.indices]
                vectors = self._forward([rendered[i] for i in rows])
                for row, vector in zip(rows, vectors, strict=True):
                    out[row] = vector
                    if self.cache is not None:
                        self.cache.put(keys[row], vector)

        # Fill every row from its representative.
        for index, key in enumerate(keys):
            if out[index] is None:
                out[index] = out[representatives[key]]

        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise NotReady("the encoder produced no vector for some inputs", n_missing=len(missing))
        return np.stack([v for v in out if v is not None]).astype(np.float32, copy=False)

    def _token_length(self, text: str) -> int:
        """Cheap length proxy for batching. Exactness is not needed: it only orders and groups rows."""
        return min(self.card.max_tokens, max(1, len(text) // 3))

    def _forward(self, texts: Sequence[str]) -> np.ndarray:
        batch = self.backend.tokenize(texts, max_length=self.card.max_tokens)
        pooled = self.backend.forward(batch)
        self.forward_calls += 1
        self.rows_encoded += len(texts)
        array = np.asarray(_to_numpy(pooled), dtype=np.float32)
        return l2_normalize(array) if self.card.normalize else array

    # -- diagnostics ----------------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "model": self.card.name,
            "profile": self.profile,
            "threads": self.threads,
            "dim": self.dim,
            "forward_calls": self.forward_calls,
            "rows_encoded": self.rows_encoded,
            "cache": dict(self.cache.stats) if self.cache else None,
        }


def _to_numpy(tensor: Any) -> Any:
    """Accept a torch tensor or anything array-like, without importing torch when it is not involved."""
    if hasattr(tensor, "detach"):
        return tensor.detach().to("cpu").float().numpy()
    return np.asarray(tensor)


def load_runtime(
    card: ModelCard,
    model_dir: str | Path,
    *,
    profile: str = REFERENCE_PROFILE,
    cache: VectorCache | None = None,
    prep_hash: str = "",
) -> EncoderRuntime:
    """Build a runtime over real weights. Raises clearly when the weights are simply not here yet."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise NotReady(
            f"no weights for {card.name} at {directory}. They are fetched once, offline-capable, by the owner "
            "(G0.4 pins them); until then the bake-off cannot run.",
        )
    return EncoderRuntime(
        card=card,
        backend=TransformersBackend(card, directory, profile),
        profile=profile,
        cache=cache,
        prep_hash=prep_hash,
    )


__all__ = [
    "DTYPES",
    "OFFLINE_ENV",
    "Backend",
    "EncoderRuntime",
    "TransformersBackend",
    "enforce_offline",
    "load_runtime",
]
