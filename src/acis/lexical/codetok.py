"""The code-aware tokeniser (docs/spec/02 §2, gate G2).

A problem statement is prose and a solution is Python, so the lexical channel is matching across a vocabulary gap
that plain whitespace tokenisation makes worse: `binary_search` and "binary search" share no token, `10^9+7` and
`1000000007` share no token, and `%` — the operator a modular-arithmetic problem is *about* — is thrown away as
punctuation.

This tokeniser closes those three gaps and nothing else:

* **identifier splitting** — `binary_search`, `binarySearch` and `BinarySearch2D` all yield `binary`, `search`
  (plus the joined form, so an exact identifier match still scores);
* **numeric folding** — `10^9+7`, `10**9 + 7` and `1e9+7` all fold to `1000000007`, reusing the same generic
  arithmetic the query side already uses (`prep.normalize`, INV-15: no list of interesting constants);
* **operator tokens** — `%` → `op_mod`, `//` → `op_floordiv`, `**` → `op_pow`, `<<` → `op_shl`, so an operator can
  be matched instead of deleted.

Algorithm-carrying builtins (`heapq`, `bisect`, `deque`, `gcd`, …) survive stopword removal because they are the
strongest lexical evidence a solution has; ordinary English stopwords still go.

It is **not** on by default. Gate G2 decides whether it ships, against the stock tokenisation that parity P2 pins
(`acis.lexical.tokenize`) — an untested lexical variant that merely looks cleverer is how a channel gets worse
without anyone noticing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from acis.prep.normalize import fold_numeric_literals

#: Operators that carry algorithmic meaning. Ordered longest-first so `//` is never read as two `/`.
OPERATORS: tuple[tuple[str, str], ...] = (
    ("**", "op_pow"),
    ("//", "op_floordiv"),
    ("<<", "op_shl"),
    (">>", "op_shr"),
    ("%", "op_mod"),
    ("^", "op_xor"),
    ("&", "op_and"),
    ("|", "op_or"),
)

#: Names that are evidence rather than noise: a solution importing `heapq` is telling us what it is.
KEEP = frozenset(
    {
        "heapq",
        "bisect",
        "deque",
        "defaultdict",
        "counter",
        "itertools",
        "permutations",
        "combinations",
        "gcd",
        "lcm",
        "factorial",
        "sqrt",
        "ceil",
        "floor",
        "mod",
        "modulo",
        "dp",
        "memo",
        "lru_cache",
        "cache",
        "sort",
        "sorted",
        "reverse",
        "queue",
        "stack",
        "graph",
        "tree",
        "node",
        "edge",
        "visit",
        "visited",
        "dfs",
        "bfs",
        "dijkstra",
        "prime",
        "sieve",
        "binary",
        "search",
        "matrix",
        "grid",
        "sum",
        "min",
        "max",
        "abs",
        "set",
        "dict",
        "list",
        "tuple",
        "str",
        "int",
        "float",
        "range",
        "input",
        "print",
    }
)

#: Python writes the exponent the prose does not: `10**9 + 7` against `10^9+7`. Rewriting `**` to `^` lets the
#: *same* generic fold serve both sides, instead of a second implementation that could drift from the first.
_PY_POWER = re.compile(r"(\d{1,18}+)\s*+\*\*\s*+(\d{1,18}+)")  # bounded: see `prep.normalize`
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
# The readable form — `(['"])(?:\\.|(?!\1).)*\1` — is **exponential**, and on exactly the text this corpus is
# made of. `\\.` and `(?!\1).` both match a backslash followed by anything, so every backslash doubles the number
# of paths the engine explores; an APPS statement is full of LaTeX (`\le`, `\sum_{i}`, `\cdot`) and apostrophes
# ("Bob's"), and one unmatched quote after eight LaTeX groups took **29 seconds** on 180 characters. The two
# branches here are disjoint — one cannot start with a backslash or a quote, the other must — and the possessive
# star stops the engine giving characters back. Same literals found, 0.00001 s.
_STRING = re.compile(r"""(['"])(?:[^'"\\]|\\.)*+\1""")
MIN_TOKEN_CHARS = 2


def split_identifier(name: str) -> list[str]:
    """`binary_search` / `binarySearch` / `HTTPResponse2` → their parts, plus the joined form when it differs.

    The joined form is kept on purpose: an exact identifier match is stronger evidence than its parts, and
    dropping it would trade a precise signal for a fuzzy one.
    """
    parts: list[str] = []
    for chunk in name.split("_"):
        parts.extend(m.group(0).lower() for m in _CAMEL.finditer(chunk))
    parts = [p for p in parts if p]
    joined = name.lower().replace("_", "")
    if len(parts) > 1 and joined and joined not in parts:
        parts.append(joined)
    return parts or ([joined] if joined else [])


def operator_tokens(text: str) -> list[str]:
    """One token per operator *occurrence*, so using `%` twice counts twice — term frequency is the point."""
    found: list[str] = []
    index = 0
    while index < len(text):
        for symbol, token in OPERATORS:
            if text.startswith(symbol, index):
                found.append(token)
                index += len(symbol)
                break
        else:
            index += 1
    return found


def code_tokens(text: str, *, keep_stopwords: bool = False, stopwords: frozenset[str] | None = None) -> list[str]:
    """Tokenise one document or query for the code-aware lexical channel."""
    if not text:
        return []
    # Operators are read from the *original* text: rewriting `**` for the numeric fold would otherwise erase the
    # `op_pow` evidence that a solution exponentiates at all.
    tokens: list[str] = operator_tokens(text)
    folded = fold_numeric_literals(_PY_POWER.sub(r"\1^\2", text))

    # String literals are content — an expected output of "YES" is a real signal — and they are tokenised as
    # ordinary text rather than kept whole, so "YES\n" and "YES" match.
    for match in _STRING.finditer(folded):
        tokens.extend(part.lower() for part in _TOKEN.findall(match.group(0)))

    for match in _TOKEN.finditer(folded):
        raw = match.group(0)
        if raw.isdigit():
            tokens.append(raw)
            continue
        tokens.extend(split_identifier(raw))

    if keep_stopwords or stopwords is None:
        return [t for t in tokens if len(t) >= MIN_TOKEN_CHARS or t.isdigit()]
    return [t for t in tokens if (len(t) >= MIN_TOKEN_CHARS or t.isdigit()) and (t not in stopwords or t in KEEP)]


def tokenize_code(texts: Sequence[str], *, stopwords: Sequence[str] | None = None, stemmer: Any | None = None) -> Any:
    """Tokenise a corpus for `bm25s`, in the `Tokenized` shape its index expects.

    The tokens are produced here and then handed to `bm25s` as space-joined text with its own filtering switched
    off. `bm25s.tokenize` takes strings, not pre-split lists — and its default word pattern keeps exactly what
    this module emits: `op_mod` stays one token (underscores are word characters), numbers survive, and nothing
    shorter than two characters is produced in the first place. So the round trip through a string is lossless,
    and it costs one join instead of a second implementation of `Tokenized`.
    """
    import bm25s  # noqa: PLC0415

    stops = frozenset(s.lower() for s in (stopwords or ()))
    split = [code_tokens(text, stopwords=stops or None) for text in texts]
    if stemmer is not None:
        split = [[str(token) for token in stemmer.stemWords(tokens)] if tokens else [] for tokens in split]
    return bm25s.tokenize([" ".join(tokens) for tokens in split], stopwords=None, stemmer=None, show_progress=False)


__all__ = [
    "KEEP",
    "MIN_TOKEN_CHARS",
    "OPERATORS",
    "code_tokens",
    "operator_tokens",
    "split_identifier",
    "tokenize_code",
]
