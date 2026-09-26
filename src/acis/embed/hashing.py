"""A deterministic, model-free encoder used to validate the harness — **never a submission** (Phase 1).

It exists so that the adapter, both modes and the dev task can be exercised end to end before the real encoder
runtime lands in Phase 2: the same object can be driven through mteb's `SearchEncoderWrapper` (Mode B) and through
our `SearchProtocol` path (Mode A), and the two rankings must agree (parity P1). That test is only meaningful if the
encoder is exactly reproducible, which a hashing encoder is and a neural one is not (to the last bit).

`submission_capable` is `False`, so a strict run refuses it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from acis.core.hashing import hash_obj
from acis.embed.base import l2_normalize

DEFAULT_DIM = 4096  # 256 collided badly: hash collisions, not ranking logic, made the stand-in look brittle
_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens plus identifier sub-tokens, so `maxSum` and `max_sum` share features."""
    out: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        out.append(token)
        if "_" in token:
            out.extend(p for p in token.split("_") if p)
    return out


def _bucket(token: str, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big") % dim


@dataclass(frozen=True, slots=True)
class HashingEncoder:
    """Feature-hashed bag of tokens with sublinear term frequency and L2 normalisation."""

    dim: int = DEFAULT_DIM
    query_prefix: str = ""
    doc_prefix: str = ""

    @property
    def name(self) -> str:
        return f"acis/hashing-{self.dim}"

    @property
    def fingerprint(self) -> str:
        return hash_obj({"encoder": "hashing", "dim": self.dim, "q": self.query_prefix, "d": self.doc_prefix})

    @property
    def submission_capable(self) -> bool:
        return False

    #: A bag of hashed tokens has no instruction, so no route can change its vectors.
    route_sensitive = False

    def encode(
        self, texts: Sequence[str], *, is_query: bool = False, batch_size: int = 64, route: str = "generic"
    ) -> np.ndarray:
        """Batch size changes nothing here — which is exactly what the batch-invariance test needs (INV-3).

        The route changes nothing either: a bag of hashed tokens has no instruction to vary. It is accepted so the
        stand-in satisfies the same contract as a real encoder, and ignored honestly rather than faked.
        """
        _ = route
        prefix = self.query_prefix if is_query else self.doc_prefix
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in _tokens(prefix + (text or "")):
                idx = _bucket(token, self.dim)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            for idx, count in counts.items():
                matrix[row, idx] = 1.0 + math.log(count)
        return l2_normalize(matrix)


__all__ = ["DEFAULT_DIM", "HashingEncoder"]
