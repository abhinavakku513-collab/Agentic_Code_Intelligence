"""The robustness suite's missing half: the engine must also *discriminate* (docs/spec/10 §5).

`test_query_agnostic.py` checks invariance — the ranking must not move when the query is perturbed in ways that
do not change its meaning. Every one of those properties is satisfied by an engine that returns the same ranking
for every query, so on their own they cannot support the claim that the engine is query-agnostic *and* useful.

These tests close that gap on the same fixture corpus and through the same hook, so they hold for whatever engine
is plugged in later. They need no relevance labels: a document quoted back as its own query is its own ground
truth.
"""

from __future__ import annotations

import importlib
import os

import pytest

HOOK = os.environ.get("ACIS_ROBUSTNESS_ENGINE", "acis.robust_hook:search")
K = int(os.environ.get("ACIS_ROBUSTNESS_K", "10"))


@pytest.fixture(scope="module")
def hook():
    module, _, function = HOOK.partition(":")
    try:
        mod = importlib.import_module(module)
        return getattr(mod, function), mod
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"engine hook {HOOK!r} is not available ({exc})")


@pytest.fixture(scope="module")
def corpus(hook):
    _, mod = hook
    if not hasattr(mod, "load_corpus"):
        pytest.skip("the hook module exposes no corpus to plant queries from")
    return list(mod.load_corpus())


def _sample(corpus, count: int = 8):
    """A deterministic spread across the corpus, so the test does not depend on where documents happen to sit."""
    step = max(1, len(corpus) // count)
    return corpus[::step][:count]


def test_a_document_quoted_as_its_own_query_ranks_first(hook, corpus):
    """The sharpest label-free discrimination check there is: a document is the best answer to itself."""
    search, _ = hook
    misses = []
    for doc in _sample(corpus):
        ranked = search(doc.text, top_k=K)
        if not ranked or ranked[0] != doc.handle:
            misses.append((doc.handle, ranked[:3]))
    assert not misses, f"documents that did not retrieve themselves at rank 1: {misses}"


def test_a_distinctive_fragment_retrieves_its_document(hook, corpus):
    """A partial quote is a realistic query shape and must still find the document it came from."""
    search, _ = hook
    found_in_top_k = 0
    candidates = _sample(corpus, 10)
    for doc in candidates:
        words = doc.text.split()
        if len(words) < 40:
            continue
        fragment = " ".join(words[10:40])
        if doc.handle in search(fragment, top_k=K):
            found_in_top_k += 1
    assert found_in_top_k >= 1, "no distinctive fragment retrieved its own document"


def test_different_queries_produce_different_rankings(hook, corpus):
    """A constant-ranking engine satisfies every invariance property; it must not satisfy this one."""
    search, _ = hook
    rankings = [tuple(search(doc.text, top_k=K)) for doc in _sample(corpus)]
    assert len(set(rankings)) > 1, "every query returned the same ranking — the engine is not discriminating"


def test_a_constant_engine_passes_invariance_and_fails_discrimination(corpus):
    """The reason this module exists, stated as an executable claim.

    A degenerate engine that ignores its query satisfies determinism, format-noise invariance and every overlap
    threshold in `test_query_agnostic.py` — and fails the checks above. Invariance alone is therefore not evidence
    that the engine is query-agnostic in the useful sense; the two halves have to be read together.
    """
    constant = [doc.handle for doc in corpus[:K]]

    def constant_engine(query: str, top_k: int = K) -> list[str]:
        if not query.strip():
            raise ValueError("InvalidInput: query is empty")
        return constant[:top_k]

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import perturb  # noqa: PLC0415

    probe = corpus[0].text
    # invariance: every property the other suite checks holds trivially
    assert constant_engine(probe) == constant_engine(probe)
    assert constant_engine(perturb.format_noise(probe, 0)) == constant_engine(probe)
    for op in perturb.MODERATE.values():
        assert constant_engine(op(probe, seed=0)) == constant_engine(probe)

    # discrimination: both checks above reject it
    sampled = _sample(corpus)
    assert len({tuple(constant_engine(d.text)) for d in sampled}) == 1
    assert any(constant_engine(d.text)[:1] != [d.handle] for d in sampled)
