"""`acis.embed` — the dense-channel contract plus the harness stand-in encoder (the real runtime is Phase 2)."""

from __future__ import annotations

from acis.embed.base import Encoder, exact_search, l2_normalize, top_k_from_scores
from acis.embed.batching import Batch, plan_batches, restore_order
from acis.embed.cache import VectorCache, vector_key
from acis.embed.hashing import HashingEncoder
from acis.embed.registry import ModelCard, available_cards, load_card

__all__ = [
    "Batch",
    "Encoder",
    "HashingEncoder",
    "ModelCard",
    "VectorCache",
    "available_cards",
    "exact_search",
    "l2_normalize",
    "load_card",
    "plan_batches",
    "restore_order",
    "top_k_from_scores",
    "vector_key",
]
