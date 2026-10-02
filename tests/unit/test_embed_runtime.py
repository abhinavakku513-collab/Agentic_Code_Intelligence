"""The encoder runtime, tested against real tensors and a stub model — no weights required.

The point of injecting the backend is that everything *around* the model can be wrong in ways that do not crash:
a pooling strategy that includes padding, a batching path that reorders rows, a cache key that ignores the
instruction. Each of those produces plausible vectors and a worse score, and the bake-off would blame the model.
So they are pinned here, and when real weights arrive the only untested thing is the model itself.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from acis.core.errors import NotReady  # noqa: E402
from acis.embed import runtime  # noqa: E402
from acis.embed.cache import VectorCache  # noqa: E402
from acis.embed.pooling import pool  # noqa: E402
from acis.embed.registry import ModelCard  # noqa: E402


def make_card(**overrides) -> ModelCard:
    base = dict(
        key="stub",
        name="acis/stub",
        base_commit="deadbeef",
        licence="apache-2.0",
        pooling="mean",
        normalize=True,
        padding_side="right",
        query_template="Instruct: {task}\nQuery: {query}",
        document_template="{text}",
        tasks={"T1": "statement task", "T3": "generic task"},
        max_tokens=128,
    )
    base.update(overrides)
    return ModelCard(**base)  # type: ignore[arg-type]


class StubBackend:
    """A deterministic, model-free 'transformer': each token contributes a fixed vector.

    It is linear in the tokens present, which makes the *right* pooled answer computable by hand — so a pooling
    or masking mistake shows up as a wrong number rather than as a plausible one.
    """

    def __init__(self, dim: int = 8, *, padding_side: str = "right") -> None:
        self._dim = dim
        self.padding_side = padding_side
        self.batches: list[list[str]] = []

    def tokenize(self, texts, *, max_length):
        self.batches.append(list(texts))
        token_lists = [[(abs(hash(w)) % 97) + 1 for w in t.split()][:max_length] or [1] for t in texts]
        width = max(len(t) for t in token_lists)
        ids, mask = [], []
        for tokens in token_lists:
            pad = [0] * (width - len(tokens))
            ids.append(pad + tokens if self.padding_side == "left" else tokens + pad)
            mask.append(
                [0] * len(pad) + [1] * len(tokens)
                if self.padding_side == "left"
                else [1] * len(tokens) + [0] * len(pad)
            )
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def forward(self, batch):
        ids, mask = batch["input_ids"], batch["attention_mask"]
        hidden = torch.zeros((*ids.shape, self._dim), dtype=torch.float32)
        for d in range(self._dim):
            hidden[:, :, d] = (ids % (d + 3)).float()
        return pool(hidden, mask, strategy="mean", padding_side=self.padding_side)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def weights_digest(self) -> str:
        return "stub-weights-v1"


def make_runtime(**kwargs) -> runtime.EncoderRuntime:
    card = kwargs.pop("card", make_card())
    backend = kwargs.pop("backend", StubBackend(padding_side=card.padding_side))
    return runtime.EncoderRuntime(card=card, backend=backend, **kwargs)


# -- pooling against real tensors ---------------------------------------------------------------------------------
def test_mean_pooling_ignores_padding():
    """Otherwise a short document's vector depends on the longest document in its batch — and INV-3 breaks."""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    np.testing.assert_allclose(pool(hidden, mask, strategy="mean").numpy(), [[2.0, 2.0]])


def test_last_token_pooling_follows_the_padding_side():
    """With left padding the last real token is index -1; with right padding it is mask.sum() - 1."""
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])

    right_mask = torch.tensor([[1, 1, 0]])
    assert pool(hidden, right_mask, strategy="last_token", padding_side="right").item() == 2.0

    left_mask = torch.tensor([[0, 1, 1]])
    assert pool(hidden, left_mask, strategy="last_token", padding_side="left").item() == 3.0


def test_cls_pooling_finds_the_real_first_token_under_left_padding():
    hidden = torch.tensor([[[9.0], [1.0], [2.0]]])
    assert pool(hidden, torch.tensor([[0, 1, 1]]), strategy="cls", padding_side="left").item() == 1.0
    assert pool(hidden, torch.tensor([[1, 1, 1]]), strategy="cls", padding_side="right").item() == 9.0


def test_max_pooling_never_selects_a_pad():
    hidden = torch.tensor([[[1.0], [5.0], [100.0]]])
    assert pool(hidden, torch.tensor([[1, 1, 0]]), strategy="max").item() == 5.0


def test_an_unknown_strategy_is_refused():
    hidden, mask = torch.zeros((1, 2, 2)), torch.ones((1, 2))
    with pytest.raises(ValueError, match="unknown pooling strategy"):
        pool(hidden, mask, strategy="magic")
    with pytest.raises(ValueError, match="padding_side"):
        pool(hidden, mask, strategy="mean", padding_side="middle")


# -- batch invariance (INV-3) ---------------------------------------------------------------------------------------
def test_batching_is_never_observable_in_a_vector():
    """Batching is a cost optimisation. If it changes a result, every number after it is suspect."""
    texts = [f"document number {i} with {'long ' * (i % 7)} content" for i in range(40)]
    tiny = make_runtime(token_budget=16).encode(texts)
    huge = make_runtime(token_budget=1_000_000).encode(texts)
    np.testing.assert_array_equal(tiny, huge)


def test_a_text_encodes_the_same_alone_as_in_a_crowd():
    texts = [f"row {i}" for i in range(12)]
    together = make_runtime().encode(texts)
    for i, text in enumerate(texts):
        np.testing.assert_array_equal(make_runtime().encode([text])[0], together[i])


def test_input_order_is_preserved_through_length_sorting():
    """Length sorting reorders the work; row i out must still be row i in."""
    texts = ["tiny", "a much longer document " * 20, "mid sized text here", "x"]
    vectors = make_runtime(token_budget=32).encode(texts)
    for i, text in enumerate(texts):
        np.testing.assert_array_equal(vectors[i], make_runtime().encode([text])[0])


def test_vectors_are_normalised_when_the_card_says_so():
    vectors = make_runtime().encode(["alpha beta", "gamma"])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    unnormalised = make_runtime(card=make_card(normalize=False)).encode(["alpha beta"])
    assert not np.isclose(np.linalg.norm(unnormalised[0]), 1.0)


def test_empty_input_returns_an_empty_matrix_of_the_right_width():
    vectors = make_runtime().encode([])
    assert vectors.shape == (0, 8)


# -- the cache is part of the runtime --------------------------------------------------------------------------------
def test_a_second_pass_costs_no_forward_calls(tmp_path):
    store = VectorCache.open(tmp_path)
    first = make_runtime(cache=store)
    texts = ["alpha", "beta", "gamma"]
    cold = first.encode(texts)
    assert first.forward_calls > 0

    second = make_runtime(cache=store)
    warm = second.encode(texts)
    assert second.forward_calls == 0
    np.testing.assert_array_equal(cold, warm)


def test_duplicate_texts_are_embedded_once(tmp_path):
    """The corpus has 11 exact-duplicate documents; they should cost one forward pass, not eleven.

    Deduplication has to happen *within* a call as well as across calls: the cache alone only helps the second
    time, and a P1 batch that re-sends unchanged units would pay for every copy on the first.
    """
    store = VectorCache.open(tmp_path)
    engine = make_runtime(cache=store)
    vectors = engine.encode(["same"] * 5)
    assert engine.rows_encoded == 1
    for row in vectors[1:]:
        np.testing.assert_array_equal(row, vectors[0])


def test_deduplication_does_not_disturb_order_or_content(tmp_path):
    """Sharing a vector between identical rows must not let a row pick up its neighbour's."""
    engine = make_runtime(cache=VectorCache.open(tmp_path))
    texts = ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]
    vectors = engine.encode(texts)

    assert engine.rows_encoded == 3  # three distinct texts
    plain = make_runtime()
    for i, text in enumerate(texts):
        np.testing.assert_array_equal(vectors[i], plain.encode([text])[0])


def test_a_query_and_a_document_of_the_same_text_do_not_share_a_vector(tmp_path):
    """The instruction is part of a query's identity; sharing the entry would serve the wrong vector."""
    store = VectorCache.open(tmp_path)
    engine = make_runtime(cache=store)
    as_doc = engine.encode(["binary search"], is_query=False)
    as_query = engine.encode(["binary search"], is_query=True, route="statement_like")
    assert not np.allclose(as_doc[0], as_query[0])


def test_the_two_routes_produce_different_query_vectors(tmp_path):
    """INV-15: only `statement_like` gets the APPS-tuned instruction, so the vectors must differ."""
    engine = make_runtime(cache=VectorCache.open(tmp_path))
    statement = engine.encode(["find code"], is_query=True, route="statement_like")
    generic = engine.encode(["find code"], is_query=True, route="generic")
    assert not np.allclose(statement[0], generic[0])


def test_the_fingerprint_moves_with_weights_card_and_profile(tmp_path):
    """Anything that changes a vector must change the key that caches it."""
    base = make_runtime()
    assert base.fingerprint == make_runtime().fingerprint

    other_profile = make_runtime(profile="cpu-bf16")
    assert other_profile.fingerprint != base.fingerprint

    other_card = make_runtime(card=make_card(pooling="last_token"))
    assert other_card.fingerprint != base.fingerprint

    class OtherWeights(StubBackend):
        @property
        def weights_digest(self) -> str:
            return "stub-weights-v2"

    assert make_runtime(backend=OtherWeights()).fingerprint != base.fingerprint


def test_a_different_fingerprint_does_not_reuse_cached_vectors(tmp_path):
    """The failure this prevents: a cache quietly serving vectors from a different model."""
    store = VectorCache.open(tmp_path)
    make_runtime(cache=store).encode(["alpha"])

    class OtherWeights(StubBackend):
        @property
        def weights_digest(self) -> str:
            return "stub-weights-v2"

    other = make_runtime(backend=OtherWeights(), cache=store)
    other.encode(["alpha"])
    assert other.forward_calls == 1  # it recomputed rather than reusing


# -- loading ----------------------------------------------------------------------------------------------------------
def test_loading_without_weights_says_so_plainly(tmp_path):
    with pytest.raises(NotReady, match="no weights"):
        runtime.load_runtime(make_card(), tmp_path / "not-here")


def test_offline_is_enforced_before_anything_loads(monkeypatch):
    """D15: no query-time network, set before a tokenizer or model can reach for one."""
    for name in runtime.OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    runtime.enforce_offline()
    import os

    assert all(os.environ[name] == "1" for name in runtime.OFFLINE_ENV)


def test_the_runtime_reports_what_it_cost(tmp_path):
    engine = make_runtime(cache=VectorCache.open(tmp_path))
    engine.encode(["a", "b", "c"])
    stats = engine.stats()
    assert stats["rows_encoded"] == 3 and stats["forward_calls"] >= 1
    assert stats["model"] == "acis/stub" and stats["dim"] == 8
    assert stats["cache"]["writes"] == 3


def test_a_real_runtime_may_ship_unlike_the_stand_in():
    """The strict-mode interlock distinguishes a real encoder from the harness stand-in."""
    from acis.embed.hashing import HashingEncoder

    assert make_runtime().submission_capable
    assert not HashingEncoder().submission_capable


def test_a_cold_run_encodes_a_repeated_query_once_and_never_memoises_documents():
    """With no persistent cache (a cold official run) routing and the dense channel encode the same query text:
    one forward pass must serve both. Documents are never memoised (each is encoded once per run anyway)."""
    rt = make_runtime(cache=None)
    first = rt.encode(["find the longest path"], is_query=True)
    calls = rt.forward_calls
    again = rt.encode(["find the longest path"], is_query=True)
    assert rt.forward_calls == calls  # served from this process's memo
    np.testing.assert_array_equal(first, again)
    rt.encode(["def f(): pass"], is_query=False)
    rt.encode(["def f(): pass"], is_query=False)
    assert rt.forward_calls == calls + 2
    # A fresh runtime (a new process) starts empty: the run stays cold.
    assert make_runtime(cache=None)._session_queries == {}
