"""The Bonus end to end: searching every version and answering with lineages (docs/spec/04 §6, D12).

The claim being tested is narrow and measurable. A flat search over a history returns the same unit many times
over; a grouped one returns each unit once, at its best revision, with the history attached. Everything here is
built on Apps-Evolve, which knows the right answer by construction, so "it grouped correctly" is a measurement
rather than an impression.
"""

from __future__ import annotations

import pytest

from acis.core.config import freeze_config
from acis.core.errors import InvalidInput
from acis.core.types import EvolveRequest, SourceSpec
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG
from acis.eval import appsevolve as ae

SEEDS = [
    ("sort.py", "def bubble_sort(xs):\n    for i in range(len(xs)):\n        xs.sort()\n    return xs\n"),
    (
        "graph.py",
        "import heapq\ndef dijkstra(g, s):\n    pq = [(0, s)]\n    while pq:\n        heapq.heappop(pq)\n    return {}\n",
    ),
    ("cache.py", "from functools import lru_cache\n\n@lru_cache\ndef fib(n):\n    return fib(n-1) + fib(n-2)\n"),
]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))


@pytest.fixture
def repo():
    """A five-version history of in-place edits: three units, each one lineage, by construction."""
    evolution = ae.generate(SEEDS, n_versions=5, seed=11, edits_per_version=1, operators=list(ae.EDIT_OPERATORS))
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=256))
    engine.ingest(SourceSpec(kind="memory", location="evolve", options=evolution.as_source_options()), repo_id="hist")
    return engine, evolution


def test_the_history_becomes_one_lineage_per_seed(repo):
    engine, evolution = repo
    index = engine.lineage_index("hist")
    assert index.size == len(SEEDS)
    assert all(len(lin.members) == len(evolution.labels) for lin in index.lineages)


def test_a_grouped_answer_returns_each_unit_once(repo):
    engine, _ = repo
    response = engine.retrieve_evolution(
        EvolveRequest(query="heap priority queue shortest path", repo_id="hist", top_k=5)
    )
    ids = [g["lineage_id"] for g in response.groups]
    assert len(ids) == len(set(ids)), "a grouped answer repeated a lineage"
    assert len(ids) <= len(SEEDS)


def test_the_flat_list_is_the_one_full_of_duplicates(repo):
    """The comparison the Bonus exists to win: the same unit, five times, crowding out the others."""
    engine, _ = repo
    response = engine.retrieve_evolution(EvolveRequest(query="heap dijkstra", repo_id="hist", top_k=10, flat=True))
    rate = next(d for d in response.degradations if d.startswith("flat_duplicate_rate="))
    assert float(rate.split("=")[1]) > 0.0
    assert response.flat_results  # the flat list is still available when asked for


def test_each_group_names_its_best_revision_and_its_span(repo):
    engine, evolution = repo
    response = engine.retrieve_evolution(EvolveRequest(query="sort a list", repo_id="hist", top_k=3))
    group = response.groups[0]
    assert group["span"][0] == evolution.labels[0] and group["span"][1] == evolution.labels[-1]
    assert group["best"]["version"] in evolution.labels
    assert group["n_revisions"] == len(evolution.labels)


def test_a_group_carries_a_timeline_of_what_changed(repo):
    engine, _ = repo
    response = engine.retrieve_evolution(EvolveRequest(query="fibonacci memo cache", repo_id="hist", top_k=3))
    timeline = response.groups[0]["timeline"]
    assert timeline and timeline[0]["version"] == "v1"
    assert any(step["changed"] for step in timeline[1:]), "an edited unit must show a change point"


def test_distinct_units_are_never_merged_into_one_group(repo):
    """The failure that would be invisible: a wrong merge hides a different implementation."""
    engine, _ = repo
    response = engine.retrieve_evolution(EvolveRequest(query="anything at all", repo_id="hist", top_k=10))
    keys = {g["best"]["key"] for g in response.groups}
    assert len(keys) == len(response.groups)


def test_evolution_retrieval_needs_a_repository(repo):
    engine, _ = repo
    with pytest.raises(InvalidInput, match="repository"):
        engine.retrieve_evolution(EvolveRequest(query="x"))


def test_the_lineage_index_is_rebuilt_when_a_version_is_added(repo):
    engine, _ = repo
    before = engine.lineage_index("hist")
    engine.update_version(
        "hist",
        SourceSpec(
            kind="memory",
            location="evolve",
            options={"versions": {"v6": {"new.py": "def brand_new():\n    return 1\n"}}},
        ),
    )
    after = engine.lineage_index("hist")
    assert after.size == before.size + 1  # the new unit is its own lineage, not a merge into an existing one
