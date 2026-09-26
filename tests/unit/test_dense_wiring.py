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

from pathlib import Path

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
    a = factory.prep_hash(config(), side="doc")
    b = factory.prep_hash(freeze_config({**DEFAULT_CONFIG, "model": DEV_MODEL, "prep": long_prep}), side="doc")
    assert a != b


def test_each_side_of_the_cache_key_moves_only_with_its_own_prep():
    """Sweeping the *query* truncation must not invalidate 8,765 cached *document* vectors (G1 would pay a full
    corpus re-encode per cell), and a document setting must not invalidate cached queries."""
    base = config()
    short_query = {**DEFAULT_CONFIG["prep"], "query": {**DEFAULT_CONFIG["prep"]["query"], "max_tokens": 256}}
    short_doc = {**DEFAULT_CONFIG["prep"], "doc": {**DEFAULT_CONFIG["prep"]["doc"], "max_tokens": 256}}
    q_changed = freeze_config({**DEFAULT_CONFIG, "model": DEV_MODEL, "prep": short_query})
    d_changed = freeze_config({**DEFAULT_CONFIG, "model": DEV_MODEL, "prep": short_doc})

    assert factory.prep_hash(q_changed, side="doc") == factory.prep_hash(base, side="doc")
    assert factory.prep_hash(q_changed, side="query") != factory.prep_hash(base, side="query")
    assert factory.prep_hash(d_changed, side="query") == factory.prep_hash(base, side="query")
    assert factory.prep_hash(d_changed, side="doc") != factory.prep_hash(base, side="doc")


def test_the_runtime_keys_queries_and_documents_on_their_own_prep():
    from acis.embed.runtime import EncoderRuntime

    class Backend:
        dim = 4
        weights_digest = "w"

    card = __import__("acis.embed.registry", fromlist=["load_card"]).load_card("gte-modernbert-base")
    rt = EncoderRuntime(card=card, backend=Backend(), prep_hash="DOC", query_prep_hash="QUERY", threads=1)
    other = EncoderRuntime(card=card, backend=Backend(), prep_hash="DOC", query_prep_hash="QUERY2", threads=1)
    assert rt._cache_key("t", is_query=False, route="generic") == other._cache_key("t", is_query=False, route="generic")
    assert rt._cache_key("t", is_query=True, route="generic") != other._cache_key("t", is_query=True, route="generic")


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


# -- which task string a route uses (the G1 dimension) ---------------------------------------------------------
def test_a_route_uses_the_conservative_default_until_g1_decides():
    """Only `statement_like` gets the APPS-tuned instruction; every other route, known or not, gets generic."""
    from acis.embed.registry import load_card

    card = load_card("qwen3-embedding-0.6b")
    assert card.task_key_for("statement_like") == "T1"
    assert card.task_key_for("generic") == "T3"
    assert card.task_key_for("a-route-we-have-never-seen") == "T3"


def test_g1_can_point_one_route_at_another_task_without_touching_the_other():
    from acis.embed.registry import load_card

    card = load_card("qwen3-embedding-0.6b").with_route_task("statement_like", "T2")
    assert card.task_key_for("statement_like") == "T2"
    assert card.task_key_for("generic") == "T3"  # untouched
    assert "competitive programming" in card.format_query("find code", route="statement_like")


def test_a_task_the_card_does_not_define_is_refused():
    """The sweep chooses among the card's task strings; it never invents one (D4)."""
    from acis.core.errors import InvalidInput
    from acis.embed.registry import load_card

    with pytest.raises(InvalidInput, match="no task"):
        load_card("qwen3-embedding-0.6b").with_route_task("generic", "T9")


def test_changing_a_routes_task_changes_the_cards_fingerprint():
    """Two sweep cells must not serve each other's cached query vectors."""
    from acis.embed.registry import load_card

    card = load_card("qwen3-embedding-0.6b")
    assert card.with_route_task("generic", "T1").fingerprint != card.fingerprint


# -- the adapter uses the same factory -------------------------------------------------------------------------
def test_mode_b_encodes_through_the_configured_encoder_not_a_hard_coded_one(tmp_path):
    """Mode A and Mode B must share one model, or G-AB compares two systems instead of two surfaces.

    Asserted against the *configured* encoder rather than the dev default: the default is the G-M winner now, and
    a unit test that needed 300 MB of weights on disk would be testing the filesystem.
    """
    import yaml

    from acis.mteb_adapter import PrePostPipelineEncoder

    config = yaml.safe_load(Path("configs/dev.yaml").read_text(encoding="utf-8"))
    config["model"] = {**config["model"], "encoder": "hashing", "dim": 64}
    path = tmp_path / "standin.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    model = PrePostPipelineEncoder(config_path=str(path))
    assert model.engine.encoder is not None
    assert model.engine.encoder.fingerprint == factory.build_encoder(model.cfg).fingerprint


# -- routing v1.1 (spec 10 §4) ---------------------------------------------------------------------------------
def test_a_query_unlike_the_training_set_takes_the_generic_path():
    """R-Q2: the learned ranker is trained on problem statements, so it must not run on everything."""
    import numpy as np

    from acis.engine.routing import QueryBank, decide

    bank = QueryBank.from_vectors(np.eye(4, dtype=np.float32), k=2)
    statement_like = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    stranger = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    assert decide(statement_like, 1.0, bank, tau=0.4).route == "statement_like"
    assert decide(stranger, 1.0, bank, tau=0.6).route == "generic"


def test_a_statement_like_query_with_no_evidence_still_takes_the_generic_path():
    """Both conditions are required: resembling the training set is not enough to have anything to rank on."""
    import numpy as np

    from acis.engine.routing import QueryBank, decide

    bank = QueryBank.from_vectors(np.eye(3, dtype=np.float32), k=1)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert decide(vector, 0.0, bank, tau=0.4, rho=0.35).route == "generic"
    assert decide(vector, 0.8, bank, tau=0.4, rho=0.35).route == "statement_like"


def test_routing_without_a_bank_routes_down_never_up():
    """A missing bank, a failed encode, an empty vector: every routing failure is the generic path."""
    from acis.engine.routing import decide

    assert decide(None, 1.0, None).route == "generic"


def test_the_route_is_counted_but_is_not_a_degradation():
    """A query served by the generic path was served as designed; a manifest must not report a fallback."""
    from acis.core.config import freeze_config
    from acis.core.types import SearchRequest
    from acis.engine.core import DEFAULT_CONFIG

    config = freeze_config({**DEFAULT_CONFIG, "run": {**DEFAULT_CONFIG["run"], "channel": "hybrid"}})
    engine = AcisEngine.from_config(config, encoder=HashingEncoder(dim=128))
    engine.build_snapshot(DOCS, source="route")
    response = engine.search(SearchRequest(query="anything at all", top_k=3))
    assert "ltr_off_route" not in " ".join(response.degradations)


# -- the route is computed only where it can change something ----------------------------------------------------
class InsensitiveEncoder(RecordingEncoder):
    """A model with no instruction format: every route renders the same input, so the route cannot matter."""

    route_sensitive = False


def _engine_with_bank(encoder, channel="dense"):
    from acis.engine.routing import QueryBank

    cfg = freeze_config({**DEFAULT_CONFIG, "run": {**DEFAULT_CONFIG["run"], "channel": channel}})
    engine = AcisEngine.from_config(cfg, encoder=encoder)
    engine._bank = QueryBank.from_vectors(np.eye(encoder.dim, dtype=np.float32), k=2)
    engine.build_snapshot(DOCS, source="route-cost")
    encoder.calls.clear()
    return engine


def test_a_route_insensitive_dense_search_encodes_each_query_once():
    """Routing embeds the query itself; when the route cannot change the vector or the pipeline, that second
    full-length encode is pure cost — on the official run, one extra forward pass per query."""
    engine = _engine_with_bank(InsensitiveEncoder())
    snap = engine.build_snapshot(DOCS, source="route-cost")
    engine.encoder.calls.clear()
    engine.search_batch(snap, ["q1", "q2"], ["sort a list", "reverse a string"], top_k=3)
    assert sum(1 for is_query, _ in engine.encoder.calls if is_query) == 2


def test_the_hybrid_channel_still_routes(monkeypatch):
    """The route picks the ranker or the generic path: there it changes the answer and must be computed."""
    engine = _engine_with_bank(InsensitiveEncoder(), channel="hybrid")
    snap = engine.build_snapshot(DOCS, source="route-cost")
    seen = []
    monkeypatch.setattr(engine, "route", lambda q, **kw: seen.append(q) or "generic")
    engine.search_batch(snap, ["q1"], ["sort a list"], top_k=3)
    assert seen


def test_a_route_sensitive_encoder_still_routes(monkeypatch):
    """An encoder that does not declare itself insensitive is assumed sensitive: the default is the safe one."""
    engine = _engine_with_bank(RecordingEncoder())
    snap = engine.build_snapshot(DOCS, source="route-cost")
    seen = []
    monkeypatch.setattr(engine, "route", lambda q, **kw: seen.append(q) or "generic")
    engine.search_batch(snap, ["q1"], ["sort a list"], top_k=3)
    assert seen


def test_the_runtime_is_route_sensitive_only_when_its_input_differs_by_route():
    from acis.embed.registry import load_card
    from acis.embed.runtime import EncoderRuntime

    class Backend:
        dim = 4
        weights_digest = "w"

    plain = load_card("gte-modernbert-base")  # no instruction format
    assert EncoderRuntime(card=plain, backend=Backend(), threads=1).route_sensitive is False
    instructed = replace_card(plain, query_template="{task}: {query}", tasks={"T1": "statement", "T3": "generic"})
    assert EncoderRuntime(card=instructed, backend=Backend(), threads=1).route_sensitive is True
    same = replace_card(plain, query_template="{task}: {query}", tasks={"T1": "same", "T3": "same"})
    assert EncoderRuntime(card=same, backend=Backend(), threads=1).route_sensitive is False


def replace_card(card, **fields):
    from dataclasses import replace

    return replace(card, **fields)


def test_the_stand_in_declares_itself_route_insensitive():
    assert HashingEncoder(dim=16).route_sensitive is False
