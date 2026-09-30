"""Exact symbol retrieval — the precision channel (docs/spec/02 §6b "exact identifier channel", D6).

BM25 splits `sum_intervals` into `sum` and `intervals` and weighs them like any other words; a dense encoder sees a
blur of both. A query that *names* a symbol — `def sum_intervals(`, `heapq.heappush`, `UnionFind`, `lru_cache` —
means that exact token, and the units that define or call it are the strongest evidence the corpus has.

**Query side, by shape (INV-15).** A token is a symbol when its *form* says it is code, never because it is on a
list: dotted (`heapq.heappush`), snake_case (`sum_intervals`), camelCase / PascalCase with an inner capital
(`UnionFind`, `maxProfit`), followed by `(` (`solve(`), or inside backticks. A query that is one identifier-shaped
token is a symbol too. Plain English words are not — "shortest path" names no symbol.

**Corpus side.** Every identifier and dotted name a unit contains, from a regex over its text (parse-free, so a
unit that does not parse still has symbols). Scored by IDF, so a rare symbol shared by one unit outweighs a common
one shared by thousands. Deterministic, per snapshot, rebuilt with it (content-addressed), never keyed on query text.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
#: Identifiers and dotted names in code.
CODE_SYMBOL = re.compile(rf"(?<![A-Za-z0-9_.]){_IDENT}(?:\.{_IDENT})*")
_DOTTED = re.compile(rf"^{_IDENT}(?:\.{_IDENT})+$")
_SNAKE = re.compile(r"^_?[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+_?$")
_CAMEL = re.compile(r"^[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*$")
_BACKTICK = re.compile(r"`([^`\n]{1,64})`")
_CALL = re.compile(rf"(?<![A-Za-z0-9_.])({_IDENT}(?:\.{_IDENT})*)\(")  # `solve(`, not "array (sorted)"
MAX_QUERY_SYMBOLS = 24


def query_symbols(query: str) -> tuple[str, ...]:
    """Code-shaped tokens a query names, in order of appearance, without duplicates. Empty when none."""
    text = (query or "")[:20_000]
    found: list[str] = []

    def add(token: str) -> None:
        token = token.strip().removesuffix("()")
        if token and len(token) <= 64 and token not in found and CODE_SYMBOL.fullmatch(token):
            found.append(token)

    for inner in _BACKTICK.findall(text):
        for token in CODE_SYMBOL.findall(inner):
            add(token)
    for token in _CALL.findall(text):
        add(token)
    for token in CODE_SYMBOL.findall(text):
        if _looks_like_code(token):
            add(token)
    stripped = text.strip()
    if not found and CODE_SYMBOL.fullmatch(stripped.removesuffix("()")) and len(stripped) <= 64:
        add(stripped)
    return tuple(found[:MAX_QUERY_SYMBOLS])


def _looks_like_code(token: str) -> bool:
    bare = token.strip(".")
    return bool(_DOTTED.match(bare) or _SNAKE.match(bare) or _CAMEL.match(bare))


def unit_symbols(text: str) -> frozenset[str]:
    """Every identifier and dotted name in a unit, plus each dotted name's parts (`heapq.heappush` -> `heapq`)."""
    out: set[str] = set()
    for token in CODE_SYMBOL.findall(text or ""):
        out.add(token)
        if "." in token:
            out.update(token.split("."))
    return frozenset(out)


@dataclass(slots=True)
class SymbolIndex:
    """Inverted index symbol -> units, for one snapshot."""

    doc_ids: tuple[str, ...]
    postings: dict[str, list[int]] = field(default_factory=dict)
    symbols_of: list[frozenset[str]] = field(default_factory=list)

    @classmethod
    def build(cls, doc_ids: Sequence[str], texts: Sequence[str]) -> SymbolIndex:
        index = cls(doc_ids=tuple(doc_ids))
        for position, text in enumerate(texts):
            symbols = unit_symbols(text)
            index.symbols_of.append(symbols)
            for symbol in symbols:
                index.postings.setdefault(symbol, []).append(position)
        return index

    def idf(self, symbol: str) -> float:
        df = len(self.postings.get(symbol, ()))
        return math.log(1.0 + len(self.doc_ids) / df) if df else 0.0

    def search(self, symbols: Sequence[str], k: int) -> list[tuple[str, float]]:
        """Units containing any of `symbols`, scored by the summed IDF of the symbols they contain."""
        if not symbols or k <= 0:
            return []
        scores: dict[int, float] = {}
        for symbol in dict.fromkeys(symbols):
            weight = self.idf(symbol)
            for position in self.postings.get(symbol, ()):
                scores[position] = scores.get(position, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.doc_ids[position], score) for position, score in ranked]

    def hits(self, position: int, symbols: Sequence[str]) -> int:
        """How many of the query's symbols unit `position` contains."""
        own = self.symbols_of[position]
        return sum(1 for s in dict.fromkeys(symbols) if s in own)


__all__ = ["CODE_SYMBOL", "MAX_QUERY_SYMBOLS", "SymbolIndex", "query_symbols", "unit_symbols"]
