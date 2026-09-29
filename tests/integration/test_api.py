"""The HTTP surface (docs/spec/06 §2, `.claude/rules/security.md`).

The API is where input arrives from someone who has read none of the rules, so the tests are mostly about what
it refuses: an over-long query, an unknown field, a top_k beyond the engine's contract. Plus the two structural
guarantees — a typed error becomes its status code rather than a traceback, and **no endpoint can grade
anything**, because scoring reads labels and nothing that reads labels is reachable over a socket.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from acis.core.config import freeze_config  # noqa: E402
from acis.core.types import Snippet, SourceSpec  # noqa: E402
from acis.embed.hashing import HashingEncoder  # noqa: E402
from acis.engine import AcisEngine  # noqa: E402
from acis.engine.core import DEFAULT_CONFIG  # noqa: E402

CORPUS = [
    Snippet(handle="d1", text="def binary_search(xs, t):\n    lo = 0\n    return lo\n"),
    Snippet(handle="d2", text="import heapq\ndef dijkstra(g, s):\n    return {}\n"),
    Snippet(handle="d3", text="def is_palindrome(s):\n    return s == s[::-1]\n"),
]
VERSIONS = {
    "v1": {"a.py": "def solve():\n    return 1\n"},
    "v2": {"a.py": "def solve():\n    return 2\n", "b.py": "def helper():\n    return 0\n"},
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    from acis.api.app import create_app

    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=256))
    engine.build_snapshot(CORPUS, source="api-test")
    engine.ingest(SourceSpec(kind="memory", location="api", options={"versions": VERSIONS}), repo_id="demo")
    return TestClient(create_app(engine))


# -- search ----------------------------------------------------------------------------------------------
def test_search_returns_ranked_hits_with_their_evidence(client):
    body = client.post("/v1/search", json={"query": "binary search sorted array", "top_k": 3}).json()
    assert len(body["results"]) == 3
    assert [h["rank"] for h in body["results"]] == [1, 2, 3]
    assert all(h["source"] and h["unit"]["body_hash"] for h in body["results"])
    assert body["timings_ms"]["total"] >= 0


def test_a_versioned_search_is_answered_from_that_version_alone(client):
    v1 = client.post("/v1/search", json={"query": "solve", "repo_id": "demo", "version": "v1", "top_k": 5}).json()
    assert {h["unit"]["key"] for h in v1["results"]} == {"a.py"}
    v2 = client.post("/v1/search", json={"query": "solve", "repo_id": "demo", "version": "v2", "top_k": 5}).json()
    assert {h["unit"]["key"] for h in v2["results"]} == {"a.py", "b.py"}


def test_every_retrieval_channel_is_reachable(client):
    for mode in ("auto", "dense", "lexical", "hybrid"):
        response = client.post("/v1/search", json={"query": "heap priority queue", "mode": mode, "top_k": 2})
        assert response.status_code == 200, (mode, response.text[:200])
        assert response.json()["results"]


# -- validation ------------------------------------------------------------------------------------------
def test_an_empty_query_is_rejected_before_it_reaches_the_engine(client):
    assert client.post("/v1/search", json={"query": ""}).status_code == 422


def test_an_over_long_query_is_rejected(client):
    assert client.post("/v1/search", json={"query": "x" * 20_000}).status_code == 422


def test_a_top_k_beyond_the_engines_contract_is_rejected(client):
    assert client.post("/v1/search", json={"query": "x", "top_k": 100_000}).status_code == 422


def test_an_unknown_field_is_rejected_rather_than_ignored(client):
    assert client.post("/v1/search", json={"query": "x", "rerank_with_llm": True}).status_code == 422


def test_a_typed_error_becomes_its_status_code_not_a_traceback(client):
    response = client.post("/v1/search", json={"query": "x", "repo_id": "demo", "version": "v99"})
    assert response.status_code == 404
    assert response.json()["code"] and "v99" in response.json()["message"]


# -- the Bonus -------------------------------------------------------------------------------------------
def test_evolve_groups_revisions_into_lineages(client):
    body = client.post("/v1/evolve", json={"query": "solve", "repo_id": "demo", "top_k": 5}).json()
    assert body["groups"]
    assert {g["lineage_id"] for g in body["groups"]}
    assert all("timeline" in g and "span" in g for g in body["groups"])


def test_evolve_needs_a_repository(client):
    assert client.post("/v1/evolve", json={"query": "x", "repo_id": ""}).status_code == 422


# -- repositories ----------------------------------------------------------------------------------------
def test_versions_are_listed_with_the_active_one_marked(client):
    body = client.get("/v1/repos/demo/versions").json()
    assert [v["label"] for v in body["versions"]] == ["v1", "v2"]
    assert sum(v["active"] for v in body["versions"]) == 1


def test_a_diff_reports_unit_level_changes(client):
    body = client.get("/v1/repos/demo/diff", params={"a": "v1", "b": "v2"}).json()
    assert body["added"] == ["b.py"] and body["changed"] == ["a.py"]


def test_rollback_moves_the_active_version(client):
    before = client.get("/v1/repos/demo/versions").json()
    client.post("/v1/repos/demo/rollback")
    after = client.get("/v1/repos/demo/versions").json()
    assert before["active"] != after["active"]


# -- operations ------------------------------------------------------------------------------------------
def test_health_reports_the_encoder_and_whether_it_may_ship(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["submission_capable"] is False  # the stand-in, and the API says so rather than implying otherwise


def test_metrics_are_prometheus_text_including_the_engines_own_counters(client):
    client.post("/v1/search", json={"query": "heap", "top_k": 2})
    text = client.get("/metrics").text
    assert "acis_searches_total 1" in text
    assert "# TYPE" in text


def test_there_is_no_evaluation_endpoint(client):
    """docs/spec/06 §1: scoring reads labels, so nothing that reads labels is reachable over a socket."""
    paths = {route.path for route in client.app.routes}
    assert not any("eval" in path or "qrels" in path or "score" in path for path in paths)
    for path in ("/v1/eval", "/v1/evaluate", "/eval"):
        assert client.post(path, json={}).status_code in (404, 405)


def test_the_ui_is_served_and_references_no_remote_asset(client):
    """D15: the demo machine has no network, so a CDN reference would be a blank page on the day. And every asset
    the page references must load *from where the page is served*: they were mounted under /ui while the page is
    served at /, so the browser's `app.css`/`app.js` requests 404'd and the page had no style and no behaviour."""
    import re

    page = client.get("/").text
    assert "<title>ACIS" in page
    assets = re.findall(r'(?:href|src)="([^"]+)"', page)
    assert {"app.css", "app.js", "favicon.svg"} <= set(assets)
    for asset in assets:
        assert not asset.startswith(("http:", "https:", "//")), asset
        assert client.get("/" + asset).status_code == 200, asset
    assert client.get("/favicon.ico").status_code == 200
    for asset in ("app.js", "app.css"):
        body = client.get("/" + asset).text
        assert "http://" not in body and "https://" not in body and "//cdn" not in body, asset
    assert client.get("/../pyproject.toml").status_code == 404
    assert client.get("/healthz").json()["status"] == "ok"  # the asset route never shadows the API


# -- what `acis serve` starts with ------------------------------------------------------------------------------
def test_the_served_app_can_search_the_p0_corpus_from_its_first_request(tmp_path, monkeypatch):
    """The page's default corpus is the APPS one (`repo_id="-"`). Served without a snapshot, a judge's first
    free-text query answered "no snapshot has been built yet" — the fixture above hid it by building one."""
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    from acis.api import app as api_app
    from acis.appsdata import apps

    monkeypatch.setattr(apps, "is_available", lambda: True)
    monkeypatch.setattr(apps, "load_corpus", lambda: list(CORPUS))
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=256))
    assert api_app.preload_p0(engine) is not None
    client = TestClient(api_app.create_app(engine))

    assert client.get("/readyz").json()["p0_corpus"] is True
    response = client.post("/v1/search", json={"query": "shortest path with a heap", "repo_id": "-", "top_k": 2})
    assert response.status_code == 200 and len(response.json()["results"]) == 2


def test_without_the_dataset_the_service_still_starts_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    from acis.api import app as api_app
    from acis.appsdata import apps

    monkeypatch.setattr(apps, "is_available", lambda: False)
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=256))
    assert api_app.preload_p0(engine) is None
    assert TestClient(api_app.create_app(engine)).get("/readyz").json()["p0_corpus"] is False


def test_the_channel_the_user_picks_is_the_channel_that_ranks(client):
    """The page's channel toggle was ignored: every request ranked with the configured channel."""
    query = {"query": "binary search lo hi", "repo_id": "-", "top_k": 3}
    lexical = client.post("/v1/search", json={**query, "mode": "lexical"}).json()["results"]
    dense = client.post("/v1/search", json={**query, "mode": "dense"}).json()["results"]
    assert [h["score"] for h in lexical] != [h["score"] for h in dense]


def test_a_demo_commit_becomes_a_searchable_version_and_says_what_it_cost(client):
    """P1 on the page: a change arrives, is embedded, is searchable — and the page shows how long that took."""
    before = [v["label"] for v in client.get("/v1/repos/demo/versions").json()["versions"]]
    out = client.post("/v1/repos/demo/commit", json={"edits": 1, "seed": 3}).json()
    assert out["version"] not in before and out["units_total"] >= 1
    assert out["units_new"] + out["units_reused"] == out["units_total"]
    assert out["seconds_searchable"] >= out["seconds_build"] > 0
    after = [v["label"] for v in client.get("/v1/repos/demo/versions").json()["versions"]]
    assert out["version"] in after
    hits = client.post("/v1/search", json={"query": "solve", "repo_id": "demo", "version": out["version"], "top_k": 3})
    assert hits.status_code == 200 and hits.json()["results"]


def test_a_demo_commit_is_bounded(client):
    assert client.post("/v1/repos/demo/commit", json={"edits": 5000}).status_code == 422


# -- the corpus a search answers from is the corpus that was asked for (bug report 4a) ---------------------------
def test_a_search_is_answered_from_the_selected_repository_and_version(client):
    """Every result, and the snapshot it names, must belong to the repository and version the request selected.

    Reported from the page: a repository picked on the Versions tab, results and a snapshot id from another corpus.
    The response now says which repository answered, and this pins it for every repository and every version.
    """
    client.post("/v1/repos/other/ingest", json={"kind": "memory", "location": "x"})  # absent source: ignored below
    engine = client.app.state.engine if hasattr(client.app.state, "engine") else None
    _ = engine
    repos = client.get("/v1/repos").json()["repos"]
    for repo in [r["repo_id"] for r in repos]:
        versions = client.get(f"/v1/repos/{repo}/versions").json()
        own = {v["snapshot_id"]: v["label"] for v in versions["versions"]}
        for selector in ["latest", *own.values()]:
            body = {"query": "solve", "repo_id": repo, "version": selector, "top_k": 5}
            out = client.post("/v1/search", json=body).json()
            assert out["snapshot"]["repo_id"] == repo, (repo, selector)
            assert out["snapshot"]["id"] in own, (repo, selector)
            if selector != "latest":
                assert own[out["snapshot"]["id"]] == selector or out["snapshot"]["version"] == selector
            assert all(h["unit"]["repo_id"] == repo for h in out["results"])
    p0 = client.post("/v1/search", json={"query": "solve", "repo_id": "-", "top_k": 3}).json()
    assert p0["snapshot"]["repo_id"] == "-" and p0["snapshot"]["n_units"] == len(CORPUS)


def test_two_repositories_with_identical_content_stay_separate(tmp_path, monkeypatch):
    """Snapshot ids are content-addressed, so two repositories holding the same content share one. Each must still
    list its own versions with their real sizes, and answer only from itself."""
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    from acis.api.app import create_app

    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=256))
    for repo in ("alpha", "beta"):
        engine.ingest(SourceSpec(kind="memory", location=repo, options={"versions": VERSIONS}), repo_id=repo)
    client = TestClient(create_app(engine))
    for repo in ("alpha", "beta"):
        listed = client.get(f"/v1/repos/{repo}/versions").json()["versions"]
        assert [v["n_units"] for v in listed] == [1, 2], (repo, listed)
        out = client.post("/v1/search", json={"query": "helper", "repo_id": repo, "top_k": 2}).json()
        assert out["snapshot"]["repo_id"] == repo
