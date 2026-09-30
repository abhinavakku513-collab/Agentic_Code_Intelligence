"""The P0 evaluation the page shows is the ledger's number, exactly — no drift, no second computation (INV-14).

The page's panel is HTML rendered once by the server from the ledger row and inserted verbatim by `app.js`. These
tests prove each link of that chain:

* every rendered metric carries the run id and the exact float of the ledger row it came from, and its text is
  that float formatted once (points, two decimals) — checked against the ledger **file** read independently here;
* the same holds for the real ledger whenever a pipeline evaluation has been recorded in it;
* an altered per-query artifact is reported as altered and never served;
* `app.js` inserts the panel's HTML unchanged and does no arithmetic on it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from acis.api import benchmarks  # noqa: E402
from acis.api.app import STATIC_DIR, create_app  # noqa: E402
from acis.core.config import freeze_config  # noqa: E402
from acis.core.types import Snippet  # noqa: E402
from acis.embed.hashing import HashingEncoder  # noqa: E402
from acis.engine import AcisEngine  # noqa: E402
from acis.engine.core import DEFAULT_CONFIG  # noqa: E402
from acis.eval import ledger  # noqa: E402

CELL = re.compile(
    r'<td class="num" data-metric="([a-z_0-9]+)" data-system="([a-z]+)" data-run="([^"]+)" '
    r'data-value="([^"]+)">([^<]+)</td>'
)


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder(dim=64))
    engine.build_snapshot([Snippet(handle="d1", text="def f():\n    return 1\n")], source="t")
    return TestClient(create_app(engine))


def _record(tmp_path: Path, monkeypatch, *, full: float, dense: float) -> dict[str, str]:
    """Append a real (hash-chained) full + dense pair and the artifact it pins, in an isolated ledger and root."""
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    monkeypatch.setattr(benchmarks, "acis_root", lambda: tmp_path)
    artifact = tmp_path / "runs" / "eval" / "p0-test" / "per_query.jsonl"
    artifact.parent.mkdir(parents=True)
    body = (
        json.dumps(
            {
                "query_id": "q1",
                "gold": ["d1"],
                "rank_full": 1,
                "rank_dense": 2,
                "route": {"route": "generic"},
                "ordered_by": "fusion.applied",
                "top10_full": ["d1"],
                "query_excerpt": "x",
                "query_chars": 1,
                "fold": 0,
            }
        )
        + "\n"
    )
    artifact.write_text(body, "utf-8")
    import hashlib

    sha = hashlib.sha256(body.encode()).hexdigest()
    ids = {}
    for system, value in (("dense", dense), ("full", full)):
        row = (
            ledger.LedgerRowBuilder(kind="dev")
            .with_metrics({"ndcg_at_10": value, "mrr_at_10": value / 1.07, "recall_at_100": 0.93})
            .with_fields(
                rung=f"p0-pipeline:{system}",
                system=system,
                split="train",
                n_queries=5000,
                artifact="runs/eval/p0-test/per_query.jsonl",
                artifact_sha256=sha,
            )
            .build()
        )
        ids[system] = ledger.append(row).run_id
    return ids


def _assert_panel_is_the_ledger(html: str, ledger_file: Path) -> int:
    rows = {json.loads(line)["run_id"]: json.loads(line) for line in ledger_file.read_text("utf-8").splitlines()}
    cells = CELL.findall(html)
    assert cells, "the panel rendered no metric cell"
    for metric, _system, run_id, data_value, text in cells:
        recorded = rows[run_id]["metrics"][metric]  # read from the file, independently of the module under test
        assert float(data_value) == recorded, (metric, run_id)  # the exact float, not a re-computation
        assert text == f"{100.0 * recorded:.2f}", (metric, run_id, text)
    return len(cells)


def test_every_number_on_the_panel_is_its_ledger_value(tmp_path, monkeypatch):
    ids = _record(tmp_path, monkeypatch, full=0.7285631234567891, dense=0.7103459876543219)
    client = _client(tmp_path, monkeypatch)
    html = client.get("/v1/benchmarks/p0/panel").text
    assert _assert_panel_is_the_ledger(html, tmp_path / "ledger.jsonl") == 6  # 3 metrics x 2 systems
    assert ids["full"] in html and ids["dense"] in html
    assert "72.86" in html and "71.03" in html
    api = client.get("/v1/benchmarks/p0").json()
    assert api["systems"]["full"]["metrics"]["ndcg_at_10"] == 0.7285631234567891
    assert api["ledger_chain_intact"] is True and api["artifact"]["sha256_matches"] is True


def test_what_is_not_measured_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    html = _client(tmp_path, monkeypatch).get("/v1/benchmarks/p0/panel").text
    assert "Not measured" in html and "data-value" not in html


def test_an_altered_artifact_is_reported_and_never_served(tmp_path, monkeypatch):
    _record(tmp_path, monkeypatch, full=0.7, dense=0.6)
    (tmp_path / "runs" / "eval" / "p0-test" / "per_query.jsonl").write_text("{}\n", "utf-8")
    client = _client(tmp_path, monkeypatch)
    assert "MISSING OR ALTERED" in client.get("/v1/benchmarks/p0/panel").text
    assert client.get("/v1/benchmarks/p0/queries").status_code == 404


def test_the_real_ledger_panel_matches_the_real_ledger(tmp_path, monkeypatch):
    """Against the repository's own ledger: whatever pipeline evaluation it holds is shown exactly."""
    real = ledger.ledger_path()
    if not real.is_file() or '"p0-pipeline:full"' not in real.read_text("utf-8"):
        pytest.skip("no pipeline evaluation recorded in the ledger yet")
    html = _client(tmp_path, monkeypatch).get("/v1/benchmarks/p0/panel").text
    assert _assert_panel_is_the_ledger(html, real) >= 2


def test_the_page_inserts_the_panel_verbatim_and_computes_nothing():
    source = (STATIC_DIR / "app.js").read_text("utf-8")
    assert 'const html = await apiText("/v1/benchmarks/p0/panel");' in source
    assert '$("eval-panel").innerHTML = html;' in source
    # The panel's element is only ever given that HTML, an error, or a placeholder — never a number built here.
    writes = re.findall(r'\$\("eval-panel"\)\.innerHTML = ([^;]+);', source)
    assert set(writes) <= {"html", '`<div class="error">${escapeHtml(err.message)}</div>`'}, writes
    assert "ndcg" not in source.lower() and "mrr" not in source.lower()


# -- adversarial inputs to the read-only benchmark endpoints (.claude/rules/security.md) --------------------------
def test_benchmark_endpoints_refuse_or_bound_hostile_input(tmp_path, monkeypatch):
    _record(tmp_path, monkeypatch, full=0.7, dense=0.6)
    client = _client(tmp_path, monkeypatch)
    assert client.get("/v1/benchmarks/p0/queries?only=__import__").status_code == 400
    page = client.get("/v1/benchmarks/p0/queries?limit=100000&offset=-5").json()
    assert page["offset"] == 0 and len(page["records"]) <= 200
    for qid in ("../../etc/passwd", "q1%00", "' OR 1=1 --", "x" * 500):
        assert client.get(f"/v1/benchmarks/p0/queries/{qid}").status_code in (404, 422)
    assert client.get("/v1/benchmarks/p0/queries/q1").json()["query_id"] == "q1"


def test_an_artifact_path_outside_runs_eval_is_never_read(tmp_path, monkeypatch):
    """The artifact path comes from a ledger row; a row pointing elsewhere must not turn into a file read."""
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    monkeypatch.setattr(benchmarks, "acis_root", lambda: tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve\n", "utf-8")
    row = (
        ledger.LedgerRowBuilder(kind="dev")
        .with_metrics({"ndcg_at_10": 0.5})
        .with_fields(
            rung="p0-pipeline:full",
            system="full",
            split="train",
            artifact="runs/eval/../../secret.txt",
            artifact_sha256="0" * 64,
        )
        .build()
    )
    ledger.append(row)
    client = _client(tmp_path, monkeypatch)
    assert client.get("/v1/benchmarks/p0").json()["artifact"]["present"] is False
    assert client.get("/v1/benchmarks/p0/queries").status_code == 404
    assert "do not serve" not in client.get("/v1/benchmarks/p0/panel").text


# -- the search tab's KPI strip and the architecture tab's bake-off table: same rule as the panel -----------------
TILE = re.compile(r'data-metric="([a-z_0-9]+)" data-run="([^"]+)" data-value="([^"]+)"><div class="v">([0-9.]+)</div>')


def test_every_headline_number_is_its_ledger_value(tmp_path, monkeypatch):
    ids = _record(tmp_path, monkeypatch, full=0.7285631234567891, dense=0.7103459876543219)
    html = _client(tmp_path, monkeypatch).get("/v1/benchmarks/p0/headline").text
    rows = {json.loads(x)["run_id"]: json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text("utf-8").splitlines()}
    tiles = TILE.findall(html)
    assert {m for m, *_ in tiles} == {"ndcg_at_10", "mrr_at_10", "recall_at_100"}  # hit@10 was not recorded here
    for metric, run_id, value, text in tiles:
        assert run_id == ids["full"]
        assert float(value) == rows[run_id]["metrics"][metric]
        assert text == f"{100.0 * rows[run_id]['metrics'][metric]:.2f}"
    assert "5,000 queries" in html and ids["full"] in html


def test_an_unrecorded_headline_says_not_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    html = _client(tmp_path, monkeypatch).get("/v1/benchmarks/p0/headline").text
    assert "Not measured" in html and "data-value" not in html


def test_the_page_inserts_headline_and_bakeoff_verbatim():
    source = (STATIC_DIR / "app.js").read_text("utf-8")
    assert '$("kpis").innerHTML = await apiText("/v1/benchmarks/p0/headline");' in source
    assert '$("sys-bakeoff").innerHTML = sys.bakeoff_html;' in source


def test_the_bakeoff_table_is_its_ledger_rows():
    from acis.api.system import render_bakeoff

    rows = [
        {
            "model": "a",
            "name": "org/a",
            "params": 149_000_000,
            "ndcg_at_10": 0.7103456,
            "mrr_at_10": 0.6745,
            "n_queries": 5000,
            "projected_cold_pass_hours": 2.58,
            "run_id": "gate-aaa",
        },
        {
            "model": "b",
            "name": "org/b",
            "params": 47_000_000,
            "ndcg_at_10": 0.5394,
            "mrr_at_10": 0.5087,
            "n_queries": 5000,
            "projected_cold_pass_hours": None,
            "run_id": "gate-bbb",
        },
    ]
    html = render_bakeoff(rows, selected="a")
    assert "71.03" in html and "53.94" in html and "gate-aaa" in html and "gate-bbb" in html
    assert 'data-value="0.7103456"' in html and html.count("selected</span>") == 1
