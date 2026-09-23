"""Version isolation over random histories (docs/spec/04 §5, INV-2).

The example-based P1 tests check histories somebody wrote down. This one generates them: random units, random
edits, random numbers of versions, and then the one property that must hold for every history — **a query pinned
to a version returns that version's content and nothing else**. A leak is the failure that would make P1 worthless
while every demo still looked right, and it is exactly the kind of thing a hand-written example misses.

`kill -9` recovery, rollback and incremental-equals-scratch are covered elsewhere; this file is only about what a
search may see.
"""

from __future__ import annotations

import string
import threading

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from acis.core.config import freeze_config
from acis.core.types import SearchRequest, SourceSpec
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG

SLOW = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

identifiers = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6)
bodies = st.text(alphabet=string.ascii_letters + " \n()_=", min_size=1, max_size=60)
#: A query is text a person typed. An empty one is `InvalidInput` by contract (spec 02 §6b) and is tested there;
#: here it would only re-test that, so the strategy asks for queries that have something in them.
queries = bodies.filter(lambda text: text.strip())


@st.composite
def histories(draw):
    """A random history: a first version, then edits that add, change or remove units."""
    keys = draw(st.lists(identifiers, min_size=1, max_size=5, unique=True))
    current = {f"{k}.py": draw(bodies) for k in keys}
    versions = {"v1": dict(current)}
    for i in range(2, draw(st.integers(min_value=2, max_value=4)) + 1):
        action = draw(st.sampled_from(["add", "change", "remove"]))
        if action == "add":
            current[f"new{i}.py"] = draw(bodies)
        elif action == "change" and current:
            current[draw(st.sampled_from(sorted(current)))] = draw(bodies)
        elif current and len(current) > 1:
            current.pop(draw(st.sampled_from(sorted(current))))
        versions[f"v{i}"] = dict(current)
    return versions


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    return tmp_path


def build(versions):
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=128))
    engine.ingest(SourceSpec(kind="memory", location="prop", options={"versions": versions}), repo_id="r")
    return engine


@given(versions=histories(), query=queries)
@SLOW
def test_a_pinned_version_only_ever_returns_its_own_units(home, versions, query):
    engine = build(versions)
    for label, units in versions.items():
        response = engine.search(SearchRequest(query=query, repo_id="r", version=label, top_k=50))
        returned = {hit.unit.key for hit in response.results}
        assert returned <= set(units), f"{label} leaked {returned - set(units)}"


@given(versions=histories(), query=queries)
@SLOW
def test_every_returned_body_is_the_body_that_version_holds(home, versions, query):
    """A key can exist in two versions with different content; each must serve its own (INV-1 + INV-2)."""
    engine = build(versions)
    for label, units in versions.items():
        response = engine.search(SearchRequest(query=query, repo_id="r", version=label, top_k=50))
        for hit in response.results:
            assert hit.source.strip() == units[hit.unit.key].strip()


@given(versions=histories())
@SLOW
def test_a_snapshot_is_reused_whenever_the_tree_repeats(home, versions):
    """Identical content is the same snapshot, so a history that returns to an earlier state costs nothing."""
    engine = build(versions)
    rows = {v["label"]: v["snapshot_id"] for v in engine.versions("r")}
    for a, units_a in versions.items():
        for b, units_b in versions.items():
            assert (rows[a] == rows[b]) == (units_a == units_b)


def test_a_search_running_during_an_activation_never_sees_a_torn_result(home):
    """Activation is a ref rename, so a reader sees one version or the other — never a mixture (spec 04 §5)."""
    v1 = {f"u{i}.py": f"def f{i}():\n    return {i}\n" for i in range(12)}
    v2 = {f"u{i}.py": f"def f{i}():\n    return {i * 10}\n" for i in range(12)}
    engine = build({"v1": v1})

    seen: list[frozenset[str]] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                response = engine.search(SearchRequest(query="def return", repo_id="r", top_k=20))
                seen.append(frozenset(h.source.strip() for h in response.results))
            except BaseException as exc:  # noqa: BLE001 — a reader must never see an error either
                errors.append(exc)
                return

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(3):
            engine.update_version("r", SourceSpec(kind="memory", location="p", options={"versions": {"v2": v2}}))
            engine.rollback("r")
    finally:
        stop.set()
        thread.join(timeout=30)

    assert not errors, errors[:1]
    whole_v1 = frozenset(t.strip() for t in v1.values())
    whole_v2 = frozenset(t.strip() for t in v2.values())
    for observation in seen:
        assert observation <= whole_v1 or observation <= whole_v2, "a search saw two versions at once"
