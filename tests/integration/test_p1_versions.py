"""P1: retrieval across versions (docs/spec/04 §3–§5, D11, INV-2, INV-9).

The official requirement is that a corpus with versions can be indexed, re-indexed as it changes "in a reasonable
amount of time", and searched per version. The failure that matters is not slowness — it is a **leak**: a query
pinned to v2 returning content that only exists in v3. That is what most of this file is about.

The rest is the promise that makes the speed honest: an incremental build must produce exactly what a build from
scratch would, or "we only embed what changed" is a shortcut rather than an optimisation.
"""

from __future__ import annotations

import pytest

from acis.core.config import freeze_config
from acis.core.errors import IndexRequired, NotFound, VersionConflict
from acis.core.types import SearchRequest, SourceSpec
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG

V1 = {
    "sort.py": "def bubble_sort(xs):\n    return sorted(xs)\n",
    "search.py": "def binary_search(xs, t):\n    return xs.index(t)\n",
}
V2 = {
    "sort.py": "def bubble_sort(xs):\n    return sorted(xs)\n",  # unchanged
    "search.py": "def binary_search(xs, t):\n    lo, hi = 0, len(xs)\n    return lo\n",  # changed
    "graph.py": "def dijkstra(g, s):\n    import heapq\n    return {}\n",  # added
}
V3 = {
    "sort.py": "def bubble_sort(xs):\n    return sorted(xs)\n",
    "graph.py": "def dijkstra(g, s):\n    import heapq\n    return {}\n",
    "cache.py": "def memoize(f):\n    return f\n",  # added; search.py removed
}


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))


def engine():
    return AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=256))


def memory_source(versions):
    return SourceSpec(kind="memory", location="p1-test", options={"versions": versions})


@pytest.fixture
def repo():
    eng = engine()
    eng.ingest(memory_source({"v1": V1, "v2": V2, "v3": V3}), repo_id="demo")
    return eng


def keys(response):
    return [hit.unit.key for hit in response.results]


def search(eng, version, query="binary search", top_k=10):
    return eng.search(SearchRequest(query=query, repo_id="demo", version=version, top_k=top_k))


# -- ingest and index ------------------------------------------------------------------------------------------
def test_ingesting_a_history_builds_every_version(repo):
    versions = repo.versions("demo")
    assert [v["label"] for v in versions] == ["v1", "v2", "v3"]
    assert all(v["snapshot_id"] for v in versions)


def test_the_newest_version_is_the_one_that_is_active(repo):
    assert sorted(keys(search(repo, "latest", "anything", top_k=10))) == sorted(V3)


def test_a_repository_can_be_ingested_from_a_jsonl_file(tmp_path):
    import json

    path = tmp_path / "corpus.jsonl"
    rows = [{"id": k, "version": v, "text": t} for v, units in (("v1", V1), ("v2", V2)) for k, t in units.items()]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    eng = engine()
    handle = eng.ingest(SourceSpec(kind="jsonl", location=str(path)), repo_id="fromfile")
    assert handle.state == "done"
    assert [v["label"] for v in eng.versions("fromfile")] == ["v1", "v2"]


# -- isolation (the leak that matters) ------------------------------------------------------------------------------
def test_a_query_pinned_to_a_version_never_returns_another_versions_content(repo):
    """INV-2. `graph.py` does not exist in v1 and `search.py` does not exist in v3."""
    assert "graph.py" not in keys(search(repo, "v1", "dijkstra heap graph"))
    assert "cache.py" not in keys(search(repo, "v2", "memoize cache"))
    assert "search.py" not in keys(search(repo, "v3", "binary search"))


def test_every_version_returns_exactly_its_own_units(repo):
    for label, expected in (("v1", V1), ("v2", V2), ("v3", V3)):
        assert sorted(keys(search(repo, label, "anything", top_k=50))) == sorted(expected)


def test_the_text_returned_is_the_text_of_that_version(repo):
    """The same key holds different content in v1 and v2; each version must return its own."""
    v1 = next(h for h in search(repo, "v1", "binary search").results if h.unit.key == "search.py")
    v2 = next(h for h in search(repo, "v2", "binary search").results if h.unit.key == "search.py")
    assert "xs.index(t)" in v1.source and "lo, hi" in v2.source


def test_a_response_says_which_snapshot_answered(repo):
    response = search(repo, "v2")
    assert response.snapshot.version == "v2"
    assert response.snapshot.id == repo.versions("demo")[1]["snapshot_id"]


def test_an_unknown_version_is_not_found(repo):
    with pytest.raises(NotFound):
        search(repo, "v99")


def test_searching_a_repository_that_was_never_indexed_asks_for_an_index():
    eng = engine()
    eng.register("brand-new")
    with pytest.raises(IndexRequired):
        eng.search(SearchRequest(query="anything", repo_id="brand-new"))


# -- incremental builds --------------------------------------------------------------------------------------------
def test_unchanged_content_is_not_re_embedded(repo):
    """The whole P1 cost story: `sort.py` is identical in all three versions and is embedded once."""
    reports = repo.build_reports("demo")
    assert reports[0].units_new == 2  # v1: both units are new
    assert reports[1].units_new == 2  # v2: search.py changed, graph.py added; sort.py reused
    assert reports[2].units_new == 1  # v3: only cache.py is new
    assert reports[2].units_reused == 2


def test_an_incremental_build_equals_a_build_from_scratch(tmp_path, monkeypatch):
    """A standing differential test (spec 04 §5): same rankings, same units, same snapshot id."""
    incremental = engine()
    incremental.ingest(memory_source({"v1": V1, "v2": V2, "v3": V3}), repo_id="demo")
    incremental_ranking = keys(search(incremental, "v3", "dijkstra heap", top_k=10))
    incremental_id = incremental.versions("demo")[-1]["snapshot_id"]

    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "scratch-home"))
    scratch = engine()
    scratch.ingest(memory_source({"v3": V3}), repo_id="demo")
    scratch_ranking = [h.unit.key for h in scratch.search(SearchRequest(query="dijkstra heap", repo_id="demo")).results]

    assert incremental_ranking == scratch_ranking
    assert incremental_id == scratch.versions("demo")[-1]["snapshot_id"]


def test_re_ingesting_identical_content_produces_the_same_snapshot(repo):
    before = repo.versions("demo")[-1]["snapshot_id"]
    repo.ingest(memory_source({"v3": V3}), repo_id="demo")
    assert repo.versions("demo")[-1]["snapshot_id"] == before


def test_update_version_adds_one_version_to_an_existing_repository(repo):
    v4 = {**V3, "extra.py": "def extra():\n    return 4\n"}
    report = repo.update_version("demo", memory_source({"v4": v4}))
    assert report.units_new == 1 and report.units_total == 4
    assert sorted(keys(search(repo, "v4", "anything", top_k=50))) == sorted(v4)
    assert sorted(keys(search(repo, "v1", "anything", top_k=50))) == sorted(V1)  # history is untouched


def test_a_stale_writer_loses_rather_than_overwriting(repo):
    with pytest.raises(VersionConflict):
        repo.update_version("demo", memory_source({"v9": V1}), expected_active="s_not_the_active_one")


# -- rollback ---------------------------------------------------------------------------------------------------------
def test_rollback_changes_what_latest_answers(repo):
    assert sorted(keys(search(repo, "latest", "anything", top_k=50))) == sorted(V3)
    repo.rollback("demo")
    assert sorted(keys(search(repo, "latest", "anything", top_k=50))) == sorted(V2)


def test_rollback_does_not_destroy_the_version_it_rolled_back_from(repo):
    repo.rollback("demo")
    assert sorted(keys(search(repo, "v3", "anything", top_k=50))) == sorted(V3)


# -- comparison ---------------------------------------------------------------------------------------------------
def test_compare_versions_reports_what_changed(repo):
    comparison = repo.compare_versions("demo", "v1", "v2")
    assert set(comparison.added) == {"graph.py"}
    assert set(comparison.changed) == {"search.py"}
    assert comparison.removed == ()


def test_compare_versions_reports_removals(repo):
    comparison = repo.compare_versions("demo", "v2", "v3")
    assert set(comparison.removed) == {"search.py"}
    assert set(comparison.added) == {"cache.py"}


def test_comparing_a_version_with_itself_finds_nothing(repo):
    comparison = repo.compare_versions("demo", "v2", "v2")
    assert not comparison.added and not comparison.removed and not comparison.changed


def test_compare_versions_can_show_what_a_query_would_return_on_each_side(repo):
    comparison = repo.compare_versions("demo", "v1", "v3", query="dijkstra heap graph")
    assert "a" in comparison.query_effect and "b" in comparison.query_effect
    assert "graph.py" in [d for d, _ in comparison.query_effect["b"]]
    assert "graph.py" not in [d for d, _ in comparison.query_effect["a"]]


# -- durability across processes ---------------------------------------------------------------------------------
def test_a_new_engine_serves_what_a_previous_one_built(repo):
    """The store is on disk, so a restart keeps every version searchable without rebuilding anything."""
    fresh = engine()
    assert sorted(keys(search(fresh, "v2", "anything", top_k=50))) == sorted(V2)
    assert [v["label"] for v in fresh.versions("demo")] == ["v1", "v2", "v3"]
