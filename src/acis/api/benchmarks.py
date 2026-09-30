"""The P0 evaluation the page shows — read from the ledger, never computed here (INV-14, spec 06 §1).

The page's "P0 evaluation" panel has exactly one source of truth: the ledger row that `scripts/bench/
eval_pipeline.py` appended, and the per-query artifact that row pins by SHA-256. This module *reads* them and
returns them verbatim:

* no metric is computed, rounded, averaged or estimated here — a value is the ledger's float, unchanged;
* the artifact is re-hashed on every read, and a mismatch is reported rather than served as if it were fine;
* the ledger's hash chain is verified on every read, so a row edited after the fact is visible as such;
* nothing reads a label file. The per-query records name the dev (TRAIN-split) gold document because the
  evaluation run wrote it into its artifact; TEST labels are never involved (D19), and an artifact whose split is
  not `train` is refused.

What the page calls "similarity" is a cosine for one live query; what it calls NDCG@10 comes only from here. The
two are never mixed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from acis.core.errors import NotFound
from acis.core.paths import acis_root

#: Ledger rungs written by `scripts/bench/eval_pipeline.py`.
PIPELINE_RUNGS = ("p0-pipeline:full", "p0-pipeline:dense")
METRICS_SHOWN = (
    "ndcg_at_10",
    "mrr_at_10",
    "recall_at_10",
    "recall_at_100",
    "ndcg_at_100",
    "mrr_at_100",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "hit_rate_at_10",
)
PROVENANCE = (
    "encoder",
    "encoder_commit",
    "model_fingerprint",
    "numeric_profile",
    "config",
    "config_hash",
    "prep_hash_doc",
    "prep_hash_query",
    "dataset",
    "split",
    "decision_set",
    "n_queries",
    "n_docs",
    "channel",
    "ranker",
    "ranker_sha256",
    "ranker_protocol",
    "route_bank_sha256",
    "route_protocol",
    "route_tau",
    "route_rho",
    "generic_alpha",
    "threads",
    "blas_threads",
    "artifact",
    "artifact_sha256",
    "counts",
    "seconds",
    "system_description",
)


def _rows() -> list[Mapping[str, Any]]:
    from acis.eval import ledger  # noqa: PLC0415

    return [row.payload for row in ledger.read_rows()]


def latest_pipeline_runs() -> dict[str, Any]:
    """The most recent full + dense pair (same artifact), with the chain and artifact checked."""
    from acis.eval import ledger  # noqa: PLC0415

    rows = [r for r in _rows() if r.get("rung") in PIPELINE_RUNGS]
    if not rows:
        return {"available": False, "reason": "no P0 pipeline evaluation has been recorded in the ledger yet"}
    latest_full = next((r for r in reversed(rows) if r.get("rung") == "p0-pipeline:full"), None)
    if latest_full is None:
        return {"available": False, "reason": "no full-pipeline row in the ledger"}
    artifact_sha = latest_full.get("artifact_sha256")
    pair = [r for r in rows if r.get("artifact_sha256") == artifact_sha]
    chain_problems = ledger.verify_chain()
    systems = {}
    for row in pair:
        system = str(row.get("system"))
        systems[system] = {
            "run_id": row.get("run_id"),
            "rung": row.get("rung"),
            "ts": row.get("ts"),
            "metrics": {k: row["metrics"][k] for k in METRICS_SHOWN if k in row.get("metrics", {})},
            "provenance": {k: row.get(k) for k in PROVENANCE if k in row},
            "git_sha": row.get("environment", {}).get("git_sha"),
            "dirty_tree": row.get("environment", {}).get("dirty"),
            "versions": row.get("environment", {}).get("versions", {}),
            "bootstrap_vs_dense": row.get("bootstrap_vs_dense"),
        }
    artifact = _verify_artifact(str(latest_full.get("artifact", "")), str(artifact_sha or ""))
    return {
        "available": True,
        "split": latest_full.get("split"),
        "systems": systems,
        "artifact": artifact,
        "ledger_chain_intact": not chain_problems,
        "ledger_chain_problems": chain_problems[:5],
        "note": (
            "Every value is the ledger row's own field, unmodified. NDCG@10 and MRR@10 are benchmark metrics over "
            "the dev (TRAIN) split; they are not the cosine similarity shown next to a live search result."
        ),
    }


def _verify_artifact(relative: str, expected: str) -> dict[str, Any]:
    path = (acis_root() / relative).resolve()
    inside = acis_root().resolve() / "runs" / "eval"
    if not relative or inside not in path.parents or not path.is_file():
        return {"path": relative, "present": False, "sha256_matches": False, "ledger_sha256": expected}
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": relative,
        "present": True,
        "sha256_matches": actual == expected,
        "sha256": actual,
        "ledger_sha256": expected,
    }


def _records(relative: str, expected: str) -> tuple[dict[str, Any], ...]:
    """The artifact's records — only after its bytes hash to what the ledger row pins, on **every** request."""
    status = _verify_artifact(relative, expected)
    if not status["present"] or not status["sha256_matches"]:
        raise NotFound("the per-query artifact is missing or does not match the hash its ledger row pins")
    return _parse(str((acis_root() / relative).resolve()), str(status["sha256"]))


@lru_cache(maxsize=4)
def _parse(path: str, actual_sha256: str) -> tuple[dict[str, Any], ...]:
    """Keyed by the bytes' own hash, so a changed file can never be answered from the cache."""
    _ = actual_sha256
    return tuple(json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line)


def per_query(*, only: str = "all", offset: int = 0, limit: int = 50) -> dict[str, Any]:
    """A page of per-query records: which rank the gold document reached, under which route and stage."""
    runs = latest_pipeline_runs()
    if not runs.get("available"):
        raise NotFound(str(runs.get("reason", "no evaluation recorded")))
    if runs.get("split") != "train":
        raise NotFound("only dev (TRAIN-split) evaluations are served")
    records = _records(runs["artifact"]["path"], runs["artifact"]["ledger_sha256"])
    selected = [r for r in records if _matches(r, only)]
    page = selected[offset : offset + limit]
    return {
        "total": len(selected),
        "offset": offset,
        "records": [
            {k: r.get(k) for k in ("query_id", "fold", "route", "ordered_by", "rank_full", "rank_dense", "query_chars")}
            | {"query_head": str(r.get("query_excerpt", ""))[:140]}
            for r in page
        ],
    }


def query_detail(query_id: str) -> dict[str, Any]:
    runs = latest_pipeline_runs()
    if not runs.get("available"):
        raise NotFound(str(runs.get("reason", "no evaluation recorded")))
    records = _records(runs["artifact"]["path"], runs["artifact"]["ledger_sha256"])
    for record in records:
        if record.get("query_id") == query_id:
            return dict(record)
    raise NotFound(f"query {query_id!r} is not in the recorded evaluation")


def _matches(record: Mapping[str, Any], only: str) -> bool:
    full, dense = record.get("rank_full"), record.get("rank_dense")
    if only == "missed":
        return full is None or full > 10
    if only == "improved":
        return (full or 1 << 30) < (dense or 1 << 30)
    if only == "worsened":
        return (full or 1 << 30) > (dense or 1 << 30)
    if only == "generic":
        return str(record.get("route", {}).get("route")) == "generic"
    return True


# -- the panel, rendered once, server-side, from the ledger row ---------------------------------------------------
LABELS = {
    "ndcg_at_10": "NDCG@10",
    "mrr_at_10": "MRR@10",
    "recall_at_10": "Recall@10",
    "recall_at_100": "Recall@100",
    "hit_rate_at_1": "Hit@1",
    "hit_rate_at_5": "Hit@5",
}


def format_points(value: float) -> str:
    """The one place a ledger metric becomes text: points with two decimals (0.72855 -> "72.86")."""
    return f"{100.0 * float(value):.2f}"


def _esc(value: Any) -> str:
    import html  # noqa: PLC0415

    return html.escape(str(value), quote=True)


def render_panel(runs: Mapping[str, Any]) -> str:
    """HTML for the P0 evaluation panel. Every number carries its ledger run id and its exact ledger value.

    The page inserts this verbatim; it does no arithmetic of its own on a metric. `data-value` is the ledger
    float's `repr`, so a test can prove the rendered text is exactly the ledger's number formatted once.
    """
    if not runs.get("available"):
        return (
            '<div class="eval-empty"><b>Not measured.</b> '
            f"{_esc(runs.get('reason', 'no evaluation recorded'))}. Run "
            "<code>uv run python scripts/bench/eval_pipeline.py</code> to record one.</div>"
        )
    systems = runs["systems"]
    order = [s for s in ("full", "dense") if s in systems]
    names = {"full": "Served pipeline (hybrid)", "dense": "Dense only (frozen encoder)"}
    head = "".join(f"<th>{_esc(names.get(s, s))}</th>" for s in order)
    body = []
    for key, label in LABELS.items():
        cells = []
        for s in order:
            value = systems[s]["metrics"].get(key)
            if value is None:
                cells.append('<td class="na">not measured</td>')
                continue
            cells.append(
                f'<td class="num" data-metric="{key}" data-system="{s}" data-run="{_esc(systems[s]["run_id"])}" '
                f'data-value="{float(value)!r}">{format_points(value)}</td>'
            )
        body.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    full = systems.get("full", {})
    prov = full.get("provenance", {})
    boot = full.get("bootstrap_vs_dense") or {}
    delta = boot.get("ndcg_at_10", {}) if isinstance(boot, Mapping) else {}
    delta_html = ""
    if delta:
        verdict = "passes" if delta.get("delta", 0) >= 0.5 and delta.get("ci_low", 0) > 0 else "does not pass"
        delta_html = (
            f'<p class="delta">Served pipeline vs dense, NDCG@10: <b data-bootstrap="delta" '
            f'data-value="{float(delta["delta"])!r}">{float(delta["delta"]):+.2f}</b> pt, 95 % CI '
            f"[{float(delta['ci_low']):+.2f}, {float(delta['ci_high']):+.2f}] (paired bootstrap, "
            f"{int(delta.get('resamples', 0)):,} resamples) — {verdict} the gate rule (Δ ≥ +0.5 and CI > 0).</p>"
        )
    counts = prov.get("counts") or {}
    facts = [
        ("Queries evaluated", f"{int(prov.get('n_queries', 0)):,}"),
        ("Corpus", f"{int(prov.get('n_docs', 0)):,} documents"),
        ("Split", f"{_esc(runs.get('split'))} (dev) — {_esc(prov.get('decision_set'))}"),
        ("Dataset", _esc(prov.get("dataset"))),
        ("Encoder", f"{_esc(prov.get('encoder'))} @ <code>{_esc(str(prov.get('encoder_commit'))[:12])}</code>"),
        ("Ranker", f"<code>{_esc(prov.get('ranker'))}</code> — {_esc(prov.get('ranker_protocol'))}"),
        ("Code", ""),  # placeholder, filled below (kept in the primary list)
    ]
    technical = [
        ("Numeric profile", _esc(prov.get("numeric_profile"))),
        ("Config", f"<code>{_esc(prov.get('config'))}</code> · <code>{_esc(str(prov.get('config_hash'))[:12])}</code>"),
        ("Routing", _esc(prov.get("route_protocol"))),
        (
            "Generic-route fusion α",
            _esc(prov.get("generic_alpha") if prov.get("generic_alpha") is not None else "not tuned"),
        ),
        (
            "Routes",
            _esc(", ".join(f"{k.split('.', 1)[1]} {v:,}" for k, v in sorted(counts.items()) if k.startswith("route."))),
        ),
        (
            "Ordered by",
            _esc(
                ", ".join(
                    f"{k.split('.', 1)[1]} {v:,}" for k, v in sorted(counts.items()) if k.startswith("ordered_by.")
                )
            ),
        ),
        ("Ranking time", f"{float(prov.get('seconds', 0.0)):.0f} s for both systems"),
    ]
    code = f"<code>{_esc(str(full.get('git_sha'))[:12])}</code>" + (
        " (uncommitted changes)" if full.get("dirty_tree") else ""
    )
    facts = [(k, code) if k == "Code" else (k, v) for k, v in facts]
    facts += [
        ("Recorded", _esc(_iso(full.get("ts")))),
        ("Ledger rows", " · ".join(f"<code>{_esc(systems[s]['run_id'])}</code>" for s in order)),
    ]
    art = runs.get("artifact", {})
    integrity = (
        f'<span class="badge {"good" if runs.get("ledger_chain_intact") else "warn"}">ledger chain '
        f"{'intact' if runs.get('ledger_chain_intact') else 'BROKEN'}</span> "
        f'<span class="badge {"good" if art.get("sha256_matches") else "warn"}">per-query artifact '
        f"{'matches its SHA-256' if art.get('sha256_matches') else 'MISSING OR ALTERED'}</span>"
    )
    return (
        f'<div class="eval-panel" data-artifact="{_esc(art.get("path"))}">'
        f'<div class="eval-integrity">{integrity}</div>'
        f'<table class="eval-table"><thead><tr><th>Metric (points)</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>{delta_html}"
        f'<dl class="facts">{"".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts)}</dl>'
        '<details class="tech-panel"><summary>Technical provenance</summary>'
        f'<dl class="facts tech-facts">{"".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in technical)}</dl></details>'
        f'<p class="muted">{_esc(runs.get("note", ""))}</p></div>'
    )


def _iso(ts: Any) -> str:
    import datetime as _dt  # noqa: PLC0415

    try:
        return _dt.datetime.fromtimestamp(float(ts), tz=_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return "unknown"


# -- the headline strip on the search tab, rendered once, server-side, from the same ledger row ---------------------
HEADLINE = (
    ("ndcg_at_10", "NDCG@10"),
    ("mrr_at_10", "MRR@10"),
    ("hit_rate_at_10", "Hit@10"),
    ("recall_at_100", "Recall@100"),
)


def render_headline(runs: Mapping[str, Any]) -> str:
    """The served pipeline's recorded accuracy as KPI tiles. Same rule as the panel: every value is the ledger
    row's float formatted once by `format_points`, carrying its run id and exact value in `data-*` attributes."""
    if not runs.get("available") or "full" not in runs.get("systems", {}):
        return '<div class="kpi-foot"><b>Not measured.</b> No P0 pipeline evaluation is recorded in the ledger.</div>'
    full = runs["systems"]["full"]
    tiles = []
    for key, label in HEADLINE:
        value = full["metrics"].get(key)
        if value is None:
            continue
        tiles.append(
            f'<div class="kpi" data-metric="{key}" data-run="{_esc(full["run_id"])}" data-value="{float(value)!r}">'
            f'<div class="v">{format_points(value)}</div><div class="k">{label}</div></div>'
        )
    boot = (full.get("bootstrap_vs_dense") or {}).get("ndcg_at_10") or {}
    if boot:
        tiles.append(
            f'<div class="kpi lift" data-bootstrap="delta" data-value="{float(boot["delta"])!r}" '
            f'title="paired bootstrap, 95 % CI [{float(boot["ci_low"]):+.2f}, {float(boot["ci_high"]):+.2f}]">'
            f'<div class="v">{float(boot["delta"]):+.2f}<small>pt</small></div>'
            '<div class="k">NDCG@10 vs dense only</div></div>'
        )
    verified = runs.get("ledger_chain_intact") and runs.get("artifact", {}).get("sha256_matches")
    n = int(full.get("provenance", {}).get("n_queries") or 0)
    tiles.append(
        f'<div class="kpi-foot">APPS dev split · {n:,} queries · out of fold · '
        f"ledger <code>{_esc(full['run_id'])}</code>"
        f"{' · chain and artifact verified' if verified else ' · <b>integrity check failed</b>'}</div>"
    )
    return "".join(tiles)


__all__ = ["format_points", "latest_pipeline_runs", "per_query", "query_detail", "render_headline", "render_panel"]
