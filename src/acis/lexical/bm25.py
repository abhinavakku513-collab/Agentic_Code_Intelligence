"""Per-snapshot BM25 (D10, docs/spec/02 §2).

One index per snapshot, so term statistics come from exactly the documents a query is allowed to see (INV-2). The
defaults mirror `mteb/baseline-bm25s` (`k1=1.5, b=0.75, delta=0.5, method="lucene"`) so the B1 rung is a true parity
check; `configs/*.yaml` can override `k1`/`b` once gate G2 has something to say about them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from acis.core.errors import InvalidInput
from acis.lexical.tokenize import english_stopwords, load_stemmer, tokenize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_DELTA = 0.5
DEFAULT_METHOD = "lucene"


@dataclass
class Bm25Index:
    """An immutable BM25 index over one snapshot's documents."""

    doc_ids: tuple[str, ...]
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    delta: float = DEFAULT_DELTA
    method: str = DEFAULT_METHOD
    stemmer_language: str | None = "english"
    _retriever: Any = field(default=None, repr=False)
    _stopwords: list[str] = field(default_factory=list, repr=False)
    _stemmer: Any = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        delta: float = DEFAULT_DELTA,
        method: str = DEFAULT_METHOD,
        stemmer_language: str | None = "english",
    ) -> Bm25Index:
        import bm25s  # noqa: PLC0415

        if len(doc_ids) != len(texts):
            raise InvalidInput("doc_ids and texts must have the same length", n_ids=len(doc_ids), n_texts=len(texts))
        stopwords = english_stopwords()
        stemmer = load_stemmer(stemmer_language)
        encoded = tokenize(texts, stopwords=stopwords, stemmer=stemmer)
        retriever = bm25s.BM25(k1=k1, b=b, delta=delta, method=method)
        retriever.index(encoded, show_progress=False)
        return cls(
            doc_ids=tuple(str(d) for d in doc_ids),
            k1=k1,
            b=b,
            delta=delta,
            method=method,
            stemmer_language=stemmer_language,
            _retriever=retriever,
            _stopwords=stopwords,
            _stemmer=stemmer,
        )

    def retrieve(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Top-`k` `(doc_id, score)` per query, in bm25s' own order (highest score first)."""
        if self._retriever is None:
            raise InvalidInput("index has not been built")
        if not queries:
            return []
        k = max(1, min(int(k), self.size))
        encoded = tokenize(list(queries), stopwords=self._stopwords, stemmer=self._stemmer)
        indices, scores = self._retriever.retrieve(encoded, k=k, show_progress=False)
        out: list[list[tuple[str, float]]] = []
        for row_idx, row_scores in zip(indices, scores, strict=True):
            out.append([(self.doc_ids[int(i)], float(s)) for i, s in zip(row_idx, row_scores, strict=True)])
        return out

    def search_one(self, query: str, k: int) -> list[tuple[str, float]]:
        """One query — identical to that query's row inside any batch (INV-3)."""
        return self.retrieve([query], k)[0] if self.size else []


__all__ = ["DEFAULT_B", "DEFAULT_DELTA", "DEFAULT_K1", "DEFAULT_METHOD", "Bm25Index"]
