"""The page's official TEST block is the release-candidate ledger row, verbatim — or an explicit "not recorded"."""

from __future__ import annotations

from acis.api import benchmarks


def test_no_release_candidate_row_means_no_number(monkeypatch):
    monkeypatch.setattr(benchmarks, "_rows", lambda: [{"kind": "dev", "metrics": {"ndcg_at_10": 0.9}}])
    result = benchmarks.official_result()
    assert result["available"] is False
    html = benchmarks.render_official(result)
    assert "Not recorded yet" in html and "90.00" not in html


def test_the_block_shows_exactly_the_latest_release_candidate_row(monkeypatch):
    rows = [
        {"kind": "rc", "run_id": "rc-old", "metrics": {"ndcg_at_10": 0.5, "mrr_at_10": 0.4}},
        {
            "kind": "rc",
            "run_id": "rc-new",
            "rc": "RC1",
            "primary_mode": "A",
            "metrics": {"ndcg_at_10": 0.78283, "mrr_at_10": 0.7439315752861569, "recall_at_100": 0.98327},
            "evaluation_time": 27849.1,
            "cold": True,
            "model_revision": "abc+def",
            "ts": 0,
        },
    ]
    monkeypatch.setattr(benchmarks, "_rows", lambda: rows)
    result = benchmarks.official_result()
    assert result["run_id"] == "rc-new" and result["split"] == "test"
    html = benchmarks.render_official(result)
    assert 'data-value="0.78283"' in html and "78.28" in html and "74.39" in html and "rc-new" in html
    assert "TEST" in html and "50.00" not in html
