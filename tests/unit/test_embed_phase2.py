"""The Phase 2 encoder infrastructure that needs no model weights: batching, the vector cache, model cards.

These three carry most of the risk in the dense engine and none of the cost. Batching decides whether the cold
pass takes two hours or eight; the cache key decides whether a number is about the model we think it is; the card
decides whether a candidate is used the way its authors intended. All are pure enough to test exactly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from acis.core.errors import InvalidInput
from acis.embed import batching, cache, registry


# -- batching ----------------------------------------------------------------------------------------------------
def test_every_input_appears_exactly_once():
    """The contract `encode()` depends on: row i out for row i in, whatever the batching did."""
    lengths = [7, 3, 900, 12, 12, 1, 450]
    batches = batching.plan_batches(lengths, token_budget=1000, max_rows=4)
    seen = [i for b in batches for i in b.indices]
    assert sorted(seen) == list(range(len(lengths)))


def test_no_batch_exceeds_the_padded_token_budget():
    """The budget is about *padded* tokens, because that is what the model computes."""
    rng = np.random.default_rng(0)
    lengths = [int(x) for x in rng.integers(1, 600, size=300)]
    for b in batching.plan_batches(lengths, token_budget=4096, max_rows=64):
        assert b.padded_tokens <= 4096 or b.size == 1  # a single over-budget row gets its own batch


def test_a_row_longer_than_the_budget_gets_its_own_batch():
    """Truncation is `acis.prep`'s decision and has already happened; batching must not silently drop a row."""
    batches = batching.plan_batches([10, 50_000, 10], token_budget=1024, max_rows=32)
    big = [b for b in batches if 1 in b.indices]
    assert len(big) == 1 and big[0].indices == (1,)


def test_length_sorting_cuts_padding_waste():
    """The whole reason for sorting: a batch of similar lengths barely pads."""
    rng = np.random.default_rng(1)
    lengths = [int(x) for x in rng.integers(1, 800, size=400)]
    sorted_waste = batching.padding_waste(batching.plan_batches(lengths, token_budget=8192))
    unsorted_waste = batching.padding_waste(batching.plan_batches(lengths, token_budget=8192, sort_by_length=False))
    assert sorted_waste < unsorted_waste
    assert sorted_waste < 0.15


def test_batching_is_deterministic():
    lengths = [5, 5, 5, 900, 12]
    first = batching.plan_batches(lengths, token_budget=1000)
    second = batching.plan_batches(lengths, token_budget=1000)
    assert [b.indices for b in first] == [b.indices for b in second]


def test_restore_order_refuses_gaps_and_duplicates():
    assert batching.restore_order([(1, "b"), (0, "a")], 2) == ["a", "b"]
    with pytest.raises(ValueError, match="no result"):
        batching.restore_order([(0, "a")], 2)
    with pytest.raises(ValueError, match="two results"):
        batching.restore_order([(0, "a"), (0, "b")], 2)
    with pytest.raises(IndexError):
        batching.restore_order([(5, "a")], 2)


def test_empty_input_plans_nothing():
    assert batching.plan_batches([]) == []


# -- the vector cache ----------------------------------------------------------------------------------------------
def test_the_key_covers_every_component_that_changes_the_vector():
    """Leaving one out is how a cache serves a vector from a different model, profile or preprocessing."""
    base = dict(model_fingerprint="m1", numeric_profile="cpu-fp32", prep_hash="p1", text="hello")
    key = cache.vector_key(**base)
    assert cache.vector_key(**{**base, "model_fingerprint": "m2"}) != key
    assert cache.vector_key(**{**base, "numeric_profile": "cpu-bf16"}) != key
    assert cache.vector_key(**{**base, "prep_hash": "p2"}) != key
    assert cache.vector_key(**{**base, "text": "hello!"}) != key
    assert cache.vector_key(**base, prompt_hash="instruct-a") != key
    assert cache.vector_key(**base) == key  # and is stable


def test_identical_text_shares_one_entry(tmp_path):
    """Content addressing: 11 duplicate documents in the corpus should cost one embedding, not 11."""
    store = cache.VectorCache.open(tmp_path)
    args = dict(model_fingerprint="m", numeric_profile="cpu-fp32", prep_hash="p")
    assert cache.vector_key(**args, text="same") == cache.vector_key(**args, text="same")
    store.put(cache.vector_key(**args, text="same"), np.arange(4, dtype=np.float32))
    assert store.get(cache.vector_key(**args, text="same")) is not None


def test_round_trip_is_exact(tmp_path):
    store = cache.VectorCache.open(tmp_path, dim=4)
    vector = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    store.put("abcd1234", vector)
    fetched = store.get("abcd1234")
    assert fetched is not None
    np.testing.assert_array_equal(fetched, vector)


def test_a_miss_is_a_miss_not_an_exception(tmp_path):
    store = cache.VectorCache.open(tmp_path)
    assert store.get("0" * 64) is None
    assert store.stats["misses"] == 1


def test_a_corrupt_entry_is_a_miss_rather_than_a_crash(tmp_path):
    """The worst case for a damaged cache entry is recomputing it, never failing the run."""
    store = cache.VectorCache.open(tmp_path)
    store.put("deadbeef", np.ones(3, dtype=np.float32))
    store._mem.clear()
    store.path_for("deadbeef").write_bytes(b"not a numpy file")
    assert store.get("deadbeef") is None


def test_a_wrong_dimension_is_refused(tmp_path):
    store = cache.VectorCache.open(tmp_path, dim=4)
    with pytest.raises(ValueError, match="dimension"):
        store.put("k", np.ones(8, dtype=np.float32))


def test_writes_are_atomic(tmp_path):
    """A crashed run must leave no half-written vector behind for the next run to trust."""
    store = cache.VectorCache.open(tmp_path)
    store.put("cafebabe", np.ones(4, dtype=np.float32))
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


def test_get_many_reports_what_is_missing(tmp_path):
    store = cache.VectorCache.open(tmp_path)
    keys = ["a" * 8, "b" * 8, "c" * 8]
    store.put(keys[1], np.ones(2, dtype=np.float32))
    found, missing = store.get_many(keys)
    assert set(found) == {1} and missing == [0, 2]


def test_the_audit_catches_a_cache_that_disagrees_with_the_model(tmp_path):
    """docs/spec/02 §3: recompute 1 % of hits and require cosine ≥ 0.9999."""
    store = cache.VectorCache.open(tmp_path)
    good, bad = "1" * 8, "2" * 8
    store.put(good, np.array([1.0, 0.0], dtype=np.float32))
    store.put(bad, np.array([1.0, 0.0], dtype=np.float32))

    honest = cache.audit_sample(store, [good], lambda _k: np.array([1.0, 0.0], np.float32), fraction=1.0)
    assert honest["failures"] == [] and honest["worst_cosine"] == pytest.approx(1.0)

    drifted = cache.audit_sample(store, [bad], lambda _k: np.array([0.0, 1.0], np.float32), fraction=1.0)
    assert drifted["failures"]


# -- model cards -------------------------------------------------------------------------------------------------
def test_the_seed_card_loads_and_describes_itself():
    card = registry.load_card("qwen3-embedding-0.6b")
    assert card.name == "Qwen/Qwen3-Embedding-0.6B"
    assert card.pooling == "last_token" and card.padding_side == "left" and card.normalize
    assert card.is_pinned  # G0.4 pinned it: `acis fetch --models … --pin` wrote the commit and the file hashes
    assert registry.licence_is_permissive(card)
    assert registry.describe(card)["card_fingerprint"]


def test_every_shipped_card_is_pinned_permissive_and_free_of_remote_code():
    """D4, checked against what is actually on disk rather than against one example."""
    for key in registry.available_cards():
        card = registry.load_card(key)
        registry.assert_shippable(card)
        assert card.file_sha256, f"{key} pins no file checksums"


def test_an_unpinned_card_may_not_ship():
    """The rule, tested on a card constructed unpinned — the shipped ones are all pinned now."""
    card = replace(registry.load_card("qwen3-embedding-0.6b"), base_commit=None)
    with pytest.raises(InvalidInput, match="not pinned"):
        registry.assert_shippable(card)


def test_instructions_are_per_route(tmp_path):
    """INV-15: only `statement_like` uses the task string G1 tuned on APPS statements."""
    card = registry.load_card("qwen3-embedding-0.6b")
    statement = card.format_query("find me code", route="statement_like")
    generic = card.format_query("find me code", route="generic")
    assert statement != generic
    assert card.tasks["T1"] in statement and card.tasks["T3"] in generic
    # An unknown route must take the generic path, not the APPS-tuned one (INV-15).
    assert card.format_query("find me code", route="anything-unknown") == generic


def test_a_model_without_an_instruction_format_gets_the_bare_text():
    """Inventing a prefix for a model not trained with one is a silent accuracy change."""
    card = registry.ModelCard(
        key="plain",
        name="x/plain",
        base_commit=None,
        licence="mit",
        pooling="mean",
        normalize=True,
        padding_side="right",
        query_template="",
        document_template="{text}",
        tasks={},
        max_tokens=512,
    )
    assert card.format_query("hello", route="statement_like") == "hello"
    assert card.format_document("body") == "body"


def test_a_card_requesting_remote_code_is_refused(tmp_path, monkeypatch):
    import yaml

    from acis.core import paths

    monkeypatch.setattr(paths, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(registry, "repo_path", lambda *p: tmp_path.joinpath(*p))
    directory = tmp_path / "configs" / "models"
    directory.mkdir(parents=True)
    (directory / "hostile.yaml").write_text(
        yaml.safe_dump({"name": "x/y", "pooling": "mean", "normalize": True, "trust_remote_code": True}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidInput, match="trust_remote_code"):
        registry.load_card("hostile")


def test_pinned_files_are_verified_against_disk(tmp_path):
    """`HF_HUB_OFFLINE=1` means whatever is on disk loads; the identity has to be checked, not assumed."""
    from acis.core.hashing import sha256_file

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    card = registry.ModelCard(
        key="pinned",
        name="x/y",
        base_commit="abc123",
        licence="apache-2.0",
        pooling="mean",
        normalize=True,
        padding_side="right",
        query_template="",
        document_template="{text}",
        tasks={},
        max_tokens=512,
        file_sha256={"model.safetensors": sha256_file(weights)},
    )
    assert registry.verify_pinned_files(card, tmp_path) == []
    registry.assert_shippable(card)

    weights.write_bytes(b"different weights")
    assert registry.verify_pinned_files(card, tmp_path) == ["checksum mismatch: model.safetensors"]


def test_a_card_pinning_nothing_says_so():
    card = replace(registry.load_card("qwen3-embedding-0.6b"), file_sha256={})
    assert registry.verify_pinned_files(card, ".") == ["the card pins no file checksums (G0.4 records them)"]


def test_a_pinned_card_catches_weights_that_do_not_match(tmp_path):
    """The point of the pins: a file that is not what was pinned must not be encoded with."""
    card = registry.load_card("qwen3-embedding-0.6b")
    (tmp_path / "model.safetensors").write_bytes(b"not the pinned weights")
    problems = registry.verify_pinned_files(card, tmp_path)
    assert problems and any("model.safetensors" in p for p in problems)
