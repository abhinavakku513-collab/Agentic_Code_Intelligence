"""`acis.embed` — the dense-channel contract plus the harness stand-in encoder (the real runtime is Phase 2)."""

from __future__ import annotations

from acis.embed.base import Encoder, exact_search, l2_normalize, top_k_from_scores
from acis.embed.hashing import HashingEncoder

__all__ = ["Encoder", "HashingEncoder", "exact_search", "l2_normalize", "top_k_from_scores"]
