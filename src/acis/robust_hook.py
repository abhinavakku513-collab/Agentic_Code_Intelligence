"""The plug-in point of `tests/robustness` (docs/spec/10 §8).

    search(query: str, top_k: int = 10) -> list[str]

Ranked document ids over a **fixed** fixture corpus of at least 200 documents. The suite it serves checks
label-free properties an arbitrary query must satisfy: determinism, format noise as a strict no-op, graceful
degradation under mild and moderate perturbation, and no crash on hostile input (empty, 16k+, NUL, RTL, emoji,
injection strings). Those are exactly the properties INV-15 is about, and they need no relevance labels.

Corpus resolution, in order: `$ACIS_ROBUSTNESS_CORPUS` · the committed fixture · the first documents of the local
dataset. The corpus is loaded once per process, so a ranking depends on the query alone.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from acis.core.config import freeze_config
from acis.core.errors import InvalidInput
from acis.core.paths import acis_root
from acis.core.types import Snippet

CORPUS_ENV = "ACIS_ROBUSTNESS_CORPUS"
FIXTURE_PATH = ("tests", "fixtures", "robust_corpus.jsonl")
MIN_DOCS = 200
FALLBACK_DOCS = 256


def fixture_path() -> Path:
    return acis_root().joinpath(*FIXTURE_PATH)


def _read_jsonl(path: Path) -> list[Snippet]:
    docs: list[Snippet] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(Snippet(handle=str(row["id"]), text=str(row["text"])))
    return docs


@lru_cache(maxsize=1)
def load_corpus() -> tuple[Snippet, ...]:
    """The fixed fixture corpus. Raises only if no source at all is available."""
    override = os.environ.get(CORPUS_ENV)
    if override:
        return tuple(_read_jsonl(Path(override)))
    if fixture_path().is_file():
        return tuple(_read_jsonl(fixture_path()))

    from acis.appsdata import apps  # noqa: PLC0415

    if not apps.is_available():
        raise InvalidInput(
            "no robustness fixture corpus: run `uv run python scripts/make_robust_fixture.py` or set " + CORPUS_ENV
        )
    return tuple(apps.load_corpus()[:FALLBACK_DOCS])


@lru_cache(maxsize=1)
def _engine_and_snapshot() -> tuple[object, object]:
    from acis.embed.hashing import HashingEncoder  # noqa: PLC0415
    from acis.engine import AcisEngine  # noqa: PLC0415
    from acis.engine.core import DEFAULT_CONFIG  # noqa: PLC0415

    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snapshot = engine.build_snapshot(list(load_corpus()), source="robustness-fixture")
    return engine, snapshot


def search(query: str, top_k: int = 10) -> list[str]:
    """Ranked document ids for one query. Raises `InvalidInput` for an empty or whitespace-only query."""
    engine, snapshot = _engine_and_snapshot()
    ranked = engine.search_batch(snapshot, ["robustness"], [query], top_k=max(1, int(top_k)))  # type: ignore[attr-defined]
    return [doc_id for doc_id, _ in ranked["robustness"]]


def corpus_ids() -> Sequence[str]:
    return [d.handle for d in load_corpus()]


def corpus_size() -> int:
    return len(load_corpus())


__all__ = [
    "CORPUS_ENV",
    "FIXTURE_PATH",
    "MIN_DOCS",
    "corpus_ids",
    "corpus_size",
    "fixture_path",
    "load_corpus",
    "search",
]
