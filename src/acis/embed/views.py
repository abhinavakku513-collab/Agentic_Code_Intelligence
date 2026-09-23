"""Query views in embedding space (docs/spec/02 §2, §5).

`acis.prep.views` renders V0 (the whole text) and V1 (statement + I/O spec, without examples or notes). **V2** is
the mean of their two embeddings, so it cannot live in `prep` — it is the only view that needs an encoder.

Which view a route uses is decided by gate G1, not here; this module only builds them, and it refuses an unknown
view rather than quietly serving V0, because a silent fallback in a sweep makes two rungs look identical when one
of them never ran.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from acis.embed.base import Encoder, l2_normalize
from acis.prep.views import build_view

VIEWS = ("V0", "V1", "V2")


def encode_views(
    encoder: Encoder,
    texts: Sequence[str],
    *,
    views: Sequence[str] = ("V0",),
    route: str = "generic",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Embed `texts` under each requested view. Rows stay in input order; V2 is derived, never re-encoded.

    V0 and V1 coincide for any query that carries no structure — most hands-on queries — so the V1 pass is skipped
    whenever the rendered texts are identical. That keeps V2 free for exactly the queries it cannot help.
    """
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; known: {VIEWS}")
    if not texts:
        return {v: np.zeros((0, encoder.dim), dtype=np.float32) for v in views}

    needed = set(views) | ({"V0", "V1"} if "V2" in views else set())
    rendered = {v: [build_view(t, v) for t in texts] for v in sorted(needed) if v != "V2"}

    embedded: dict[str, np.ndarray] = {}
    for view, prepared in rendered.items():
        if view == "V1" and prepared == rendered.get("V0"):
            embedded["V1"] = embedded["V0"]  # identical text, identical vector: do not pay twice
            continue
        embedded[view] = np.asarray(
            encoder.encode(prepared, is_query=True, batch_size=batch_size, route=route), dtype=np.float32
        )

    if "V2" in views:
        embedded["V2"] = l2_normalize(embedded["V0"] + embedded["V1"])
    return {v: embedded[v] for v in views}


__all__ = ["VIEWS", "encode_views"]
