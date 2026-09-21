"""Toy TEMPLATE-COUPLED engine: what a naive implementation written against one query template looks like.
It boosts stdin-scripts only when the query carries a benchmark-style '-----Input-----' marker. The robustness tests must catch it."""

from __future__ import annotations

import re

from toy_generic import CORPUS
from toy_generic import search as _generic

_MARK = re.compile(r"-{3,}\s*Input\s*-{3,}", re.I)


def search(query: str, top_k: int = 10) -> list[str]:
    ids = _generic(query, top_k=len(CORPUS))
    stdin_first = bool(_MARK.search(query))
    ids = sorted(ids, key=lambda d: ((("input(" in CORPUS[d]) != stdin_first), ids.index(d)))
    return ids[:top_k]
