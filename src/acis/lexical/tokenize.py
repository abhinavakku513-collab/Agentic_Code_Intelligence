"""Lexical tokenisation (docs/spec/02 §2).

Phase 1 ships the **stock** tokenisation only: the exact configuration `mteb/baseline-bm25s` uses, so our BM25 rung
(B1) can be compared to mteb's baseline rank-for-rank (parity P2). The code-aware tokeniser of spec 02 §2 is a
Phase 4 (hybrid) deliverable and is deliberately absent here — an untested lexical variant in the harness phase would
make the parity result meaningless.

Matched configuration, read from `mteb/models/model_implementations/bm25.py`:
`corpus_text = "\\n".join([title, text])`, bm25s' own whitespace tokeniser, bm25s' English stopword list, and a
Snowball stemmer when PyStemmer is installed (it is optional; `stemmer_language=""` pins the stemmer-free variant).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

STOPWORDS_KEY = "en"


def corpus_text(title: str, text: str) -> str:
    """The document string `mteb/baseline-bm25s` indexes."""
    return "\n".join([title or "", text or ""])


def english_stopwords() -> list[str]:
    from bm25s.tokenization import STOPWORDS_EN  # noqa: PLC0415

    return list(STOPWORDS_EN)


def load_stemmer(language: str | None = "english") -> Any | None:
    """PyStemmer is optional. Returns `None` when it is absent or when `language` is falsy."""
    if not language:
        return None
    try:
        import Stemmer  # noqa: PLC0415
    except ImportError:
        return None
    return Stemmer.Stemmer(language)


def stemmer_available() -> bool:
    return load_stemmer("english") is not None


def tokenize(texts: Sequence[str], *, stopwords: list[str] | None = None, stemmer: Any | None = None) -> Any:
    """bm25s tokenisation with an explicit stopword list and optional stemmer (returns bm25s' `Tokenized`)."""
    import bm25s  # noqa: PLC0415

    return bm25s.tokenize(list(texts), stopwords=stopwords, stemmer=stemmer, show_progress=False)


__all__ = ["STOPWORDS_KEY", "corpus_text", "english_stopwords", "load_stemmer", "stemmer_available", "tokenize"]
