"""Wiring the real encoder into the engine and the adapter (docs/spec/02 §2–§4, D4, INV-2, INV-15).

Phase 1 hard-wired the model-free stand-in everywhere, which was honest but meant nothing above the encoder had
ever chosen a model. These tests pin the three joints where a real encoder plugs in:

* **the factory** — one place turns configuration into an encoder, so Mode A, Mode B and the bake-off cannot end
  up with different models or different caches while reporting one number;
* **the route** — the instruction a query is encoded with is chosen per route (INV-15), so the route has to reach
  the encoder *and* the query-vector cache key. Losing it is invisible: every vector still looks fine;
* **V2** — the mean of the V0 and V1 embeddings, one of the things G1 chooses between.

None of it needs weights: the stand-in and a recording fake exercise every path.
"""

from __future__ import annotations

import numpy as np
import pytest

from acis.core.config import freeze_config
from acis.core.errors import NotReady
from acis.core.types import SearchRequest, Snippet
from acis.embed import factory
from acis.embed.hashing import HashingEncoder
from acis.embed.views import encode_views
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG

DEV_MODEL = {"encoder": "hashing", "dim": 64, "config": "configs/models/qwen3-embedding-0.6b.yaml"}


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Every test gets its own ACIS_HOME: caches and weights must never be looked for inside the repository."""
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    return tmp_path


def config(**model) -> object:
    merged = {**DEFAULT_CONFIG, "model": {**DEV_MODEL, **model}}
    return freeze_config(merged, source_path="<test>")


class RecordingEncoder:
    """A stand-in that remembers how it was called. Vectors depend on the route, as a real instruction would."""

    name = "recording"
    dim = 8
    fingerprint = "recording-v1"
    submission_capable = True

    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []

    def encode(self, texts, *, is_query=False, batch_size=64, route="generic"):
        self.calls.append((is_query, route))
        seed = abs(hash(route)) % 97 if is_query else 0
        rows = [[float(((abs(hash(t)) + seed * (d + 1)) % 1000) / 1000.0) for d in range(self.dim)] for t in texts]
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


DOCS = [Snippet(handle=f"d{i}", text=f"def solve_{i}():\n    return {i}\n") for i in range(6)]


# -- the factory ----------------------------------------------------------------------------------------------
def test_the_dev_configuration_builds_the_stand_in_and_says_it_cannot_ship():
    encoder = factory.build_encoder(config())
    assert isinstance(encoder, HashingEncoder)
    assert encoder.dim == 64 and not encoder.submission_capable


def test_a_real_model_without_weights_names_the_missing_step_rather_than_failing_obscurely():
    """The weights are fetched once, by the owner (G0.4). Until then this must be the error, not an import crash."""
    with pytest.raises(NotReady) as excinfo:
        factory.build_encoder(config(encoder="qwen3-embedding-0.6b"))
    message = str(excinfo.value)
    assert "qwen3-embedding-0.6b" in message and "weights" in message.lower()


def test_an_unknown_model_name_is_refused_before_anything_is_loaded():
    with pytest.raises(NotReady, match="no model card"):
        factory.build_encoder(config(encoder="a-model-we-never-carded"))


def test_weights_and_vectors_live_outside_the_repository(_home):
    """A cache inside the tree would carry gigabytes into every worktree and trip the seal scan (CLAUDE.md §7)."""
    from acis.core.paths import acis_root

    for path in (factory.model_dir("qwen3-embedding-0.6b"), factory.vector_cache_dir("qwen3-embedding-0.6b")):
        assert acis_root() not in path.parents and path.is_relative_to(_home / "home")


def test_the_prep_settings_are_part_of_the_cache_identity():
    """Vectors truncated at 1024 tokens are not the same vectors as those truncated at 512 (D3)."""
    long_prep = {**DEFAULT_CONFIG["prep"], "doc": {"version": "d1", "max_tokens": 512, "head": 384, "tail": 128}}
    a = factory.prep_hash(config())
    b = factory.prep_hash(freeze_config({**DEFAULT_CONFIG, "model": DEV_MODEL, "prep": long_prep}))
    assert a != b


def test_the_numeric_profile_follows_the_configuration():
    assert factory.profile_of(config()) == "cpu-fp32"
    assert factory.profile_of(config(numeric_profile="cpu-bf16")) == "cpu-bf16"


# -- the route reaches the encoder (INV-15) --------------------------------------------------------------------
def test_the_engine_encodes_a_query_with_its_own_route():
    """If the route stops at the engine, every query silently gets the generic instruction and G1 measures noise."""
    encoder = RecordingEncoder()
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=encoder)
    engine.build_snapshot(DOCS, source="test")
    engine.search(SearchRequest(query="how do I sort a list", top_k=3))

    query_calls = [route for is_query, route in encoder.calls if is_query]
    assert query_calls == [engine.route("how do I sort a list")]


def test_documents_are_never_encoded_as_a_route():
    """A document has no instruction: it is not a query, and giving it one is a silent accuracy loss."""
    encoder = RecordingEncoder()
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=encoder)
    engine.build_snapshot(DOCS, source="test")
    assert encoder.calls and all(not is_query for is_query, _ in encoder.calls)


def test_two_routes_do_not_share_one_cached_query_vector():
    """The route chooses the instruction, so it belongs in the cache key exactly as the model and profile do."""
    encoder = RecordingEncoder()
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=encoder)
    snapshot = engine.build_snapshot(DOCS, source="test")

    generic = engine._query_vector(snapshot.snapshot_id, "sort a list", route="generic")
    statement = engine._query_vector(snapshot.snapshot_id, "sort a list", route="statement_like")
    assert not np.allclose(generic, statement)
    assert engine.counters.snapshot().get("cache.qemb.hit", 0) == 0  # two routes, two misses, no false hit


def test_the_same_route_still_hits_the_cache():
    encoder = RecordingEncoder()
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=encoder)
    snapshot = engine.build_snapshot(DOCS, source="test")
    engine._query_vector(snapshot.snapshot_id, "sort a list", route="generic")
    engine._query_vector(snapshot.snapshot_id, "sort a list", route="generic")
    assert engine.counters.snapshot().get("cache.qemb.hit", 0) == 1


# -- V2 -------------------------------------------------------------------------------------------------------
def test_v2_is_the_normalised_mean_of_the_two_view_vectors():
    encoder = HashingEncoder(dim=32)
    text = "Solve it.\n-----Input-----\nn\n-----Examples-----\n1\n"
    vectors = encode_views(encoder, [text], views=("V0", "V1", "V2"))
    expected = vectors["V0"][0] + vectors["V1"][0]
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(vectors["V2"][0], expected, atol=1e-6)


def test_v2_costs_nothing_extra_when_a_query_cannot_be_segmented():
    """No markers ⇒ V1 is V0 ⇒ V2 is V0. A query with no structure must not pay for a second encode (INV-15)."""
    encoder = HashingEncoder(dim=32)
    vectors = encode_views(encoder, ["what is a heap"], views=("V0", "V2"))
    np.testing.assert_allclose(vectors["V2"][0], vectors["V0"][0], atol=1e-6)


def test_an_unknown_view_is_refused_rather_than_silently_becoming_v0():
    with pytest.raises(ValueError, match="unknown view"):
        encode_views(HashingEncoder(dim=8), ["x"], views=("V9",))


# -- the adapter uses the same factory -------------------------------------------------------------------------
def test_mode_b_encodes_through_the_configured_encoder_not_a_hard_coded_one():
    """Mode A and Mode B must share one model, or G-AB compares two systems instead of two surfaces."""
    from acis.mteb_adapter import PrePostPipelineEncoder

    model = PrePostPipelineEncoder(config_path="configs/dev.yaml")
    assert model.engine.encoder is not None
    assert model.engine.encoder.fingerprint == factory.build_encoder(model.cfg).fingerprint
