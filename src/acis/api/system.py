"""What this server actually runs — the page's architecture view, read from the live engine (never a brochure).

Every field comes from an object the engine is using right now (the loaded encoder runtime, the loaded ranker, the
frozen config, the P0 snapshot's indexes), from a model card on disk, from the ledger, or from the progress record
an embedding job writes. A component that is not running is reported as not running, with the reason; nothing is
described as serving because it was designed to.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from acis.core.paths import acis_root

#: The second dense encoder the architecture reserves a channel for (spec 08 §3 L10).
SECOND_ENCODER = "qwen3-embedding-0.6b"
#: An embedding job that has not written progress for this long is reported as paused, not as running.
STALE_PROGRESS_S = 15 * 60


def _card(key: str) -> dict[str, Any]:
    from acis.embed.factory import model_dir  # noqa: PLC0415
    from acis.embed.registry import available_cards, load_card  # noqa: PLC0415

    if key not in available_cards():
        return {"key": key, "card": False}
    card = load_card(key)
    directory = model_dir(key)
    weights = sorted(directory.glob("*.safetensors")) if directory.is_dir() else []
    return {
        "key": key,
        "card": True,
        "name": card.name,
        "licence": card.licence,
        "commit": card.base_commit,
        "max_tokens": card.max_tokens,
        "pooling": card.pooling,
        "instruction_aware": bool(card.tasks),
        "weights_present": bool(weights),
        "size_mb": round(sum(p.stat().st_size for p in weights) / 2**20, 1) if weights else None,
    }


def _progress(key: str) -> dict[str, Any] | None:
    path = acis_root() / "runs" / "embed" / f"{key}.progress.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    record: dict[str, Any] = loaded
    record["running"] = (time.time() - float(record.get("updated", 0))) < STALE_PROGRESS_S
    return record


def _encoder_entry(runtime: Any, *, role: str, key: str) -> dict[str, Any]:
    card = getattr(runtime, "card", None)
    return {
        **_card(key),
        "role": role,
        "state": "serving",
        "name": getattr(runtime, "name", key),
        "params": getattr(runtime, "n_parameters", None),
        "size_mb": getattr(runtime, "size_mb", None),
        "dim": getattr(runtime, "dim", None),
        "max_tokens": getattr(card, "max_tokens", None),
        "profile": getattr(runtime, "profile", None),
        "fingerprint": str(getattr(runtime, "fingerprint", ""))[:12],
    }


def _second_encoder(engine: Any, p0: Any) -> dict[str, Any]:
    key = engine.aux_encoder_name or SECOND_ENCODER
    serving = bool(engine.aux_encoder_name) and p0 is not None and p0.aux_vectors is not None
    if serving:
        return _encoder_entry(engine.aux_encoder, role="second dense channel", key=key)
    entry = {**_card(key), "role": "second dense channel", "params": None, "dim": None}
    progress = _progress(key)
    if progress:  # kept in the API for operators; the page does not display it
        entry["progress"] = {
            k: progress.get(k)
            for k in ("stage", "corpus_done", "corpus_total", "queries_done", "queries_total", "updated", "running")
        }
    if entry.get("weights_present"):
        # Not serving: said by `state` never being "serving", and by the page leaving the channel out of the running
        # stack and dimming it after a search. What it *is*: a pinned model one config key away.
        entry["state"] = "optional"
        entry["reason"] = "Pinned (commit and SHA-256 per file); enabled with model.aux_encoder"
    else:
        entry["state"] = "not installed"
        entry["reason"] = "no weights on this host"
    return entry


def _gate_gm() -> dict[str, Any]:
    """The recorded G-M decision (configs/gates/G-M.yaml): the rule and why each other candidate was rejected."""
    import yaml  # noqa: PLC0415

    from acis.core.paths import repo_path  # noqa: PLC0415

    path = repo_path("configs", "gates", "G-M.yaml")
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text("utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _bakeoff() -> list[dict[str, Any]]:
    """The encoder bake-off (gate G-M), one entry per ledger row, newest row per model."""
    from acis.eval import ledger  # noqa: PLC0415

    rows: dict[str, Mapping[str, Any]] = {}
    for row in ledger.read_rows():
        payload = row.payload
        if str(payload.get("rung", "")).startswith("bakeoff:"):
            rows[str(payload.get("model"))] = payload
    out = []
    for model, r in rows.items():
        card = r.get("scorecard") or {}
        metrics = r.get("metrics") or {}
        out.append(
            {
                "model": model,
                "name": r.get("model_name"),
                "params": r.get("params"),
                "ndcg_at_10": metrics.get("ndcg_at_10"),
                "mrr_at_10": metrics.get("mrr_at_10"),
                "n_queries": r.get("n_queries"),
                "model_mb": card.get("model_mb"),
                "projected_cold_pass_hours": card.get("projected_cold_pass_hours"),
                "run_id": r.get("run_id"),
            }
        )
    return sorted(out, key=lambda x: -(x["ndcg_at_10"] or 0.0))


def render_bakeoff(rows: list[dict[str, Any]], *, selected: str, gate: Mapping[str, Any] | None = None) -> str:
    """The bake-off as an HTML table, every accuracy value formatted once from its ledger float (the page inserts it
    verbatim and computes nothing, like the evaluation panel)."""
    from acis.api.benchmarks import _esc, format_points  # noqa: PLC0415

    if not rows:
        return '<div class="muted">No bake-off rows in the ledger.</div>'
    best = max(float(r["ndcg_at_10"] or 0.0) for r in rows) or 1.0
    body = []
    for r in rows:
        name = str(r.get("name") or r["model"]).split("/")[-1]
        chosen = r["model"] == selected
        badge = ' <span class="badge good">selected</span>' if chosen else ""
        params = f"{round(int(r['params']) / 1e6)}M" if r.get("params") else "-"
        width = 100.0 * float(r["ndcg_at_10"] or 0.0) / best
        hours = r.get("projected_cold_pass_hours")
        rejected = ((gate or {}).get("evidence") or {}).get("rejected") or {}
        if chosen:
            slo = (gate or {}).get("slo_cold_pass_hours")
            why = "most accurate" + (
                f"; projected cold pass {float(hours):.1f} h, inside the {float(slo):g} h budget"
                if hours is not None and slo
                else ""
            )
        else:
            why = str(rejected.get(r["model"], "not selected"))
        body.append(
            f'<tr class="{"chosen" if chosen else ""}"><td>{_esc(name)}{badge}</td><td>{params}</td>'
            f'<td data-metric="ndcg_at_10" data-run="{_esc(r["run_id"])}" data-value="{float(r["ndcg_at_10"])!r}">'
            f"{format_points(r['ndcg_at_10'])}</td>"
            f'<td class="barcell"><div class="hbar"><span style="width:{width:.1f}%"></span></div></td>'
            f'<td data-metric="mrr_at_10" data-value="{float(r["mrr_at_10"])!r}">{format_points(r["mrr_at_10"])}</td>'
            f"<td>{f'{float(hours):.1f} h' if hours is not None else '-'}</td>"
            f'<td class="why">{_esc(why)}</td>'
            f"<td><code>{_esc(r['run_id'])}</code></td></tr>"
        )
    n = rows[0].get("n_queries") or 0
    return (
        '<div class="table-wrap"><table class="bake"><thead><tr><th>encoder</th><th>params</th><th>NDCG@10</th>'
        '<th class="barcell"></th><th>MRR@10</th><th>cold pass (projected)</th><th>decision</th><th>ledger</th>'
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        f'<p class="muted" style="margin:8px 0 0">Dense retrieval alone, each encoder on all {int(n):,} dev queries, '
        f"on the CPU. {_esc(_rule_text(gate or {}))}</p>"
    )


def _rule_text(gate: Mapping[str, Any]) -> str:
    """The selection rule in words, from the gate record's own numbers."""
    if not gate:
        return "Rule: the smallest permissively licensed, CPU-capable model within tolerance of the best."
    return (
        f"Rule (gate G-M): among permissively licensed, CPU-capable models of at most "
        f"{(gate.get('envelope') or {}).get('max_params_b', 1.0):g}B parameters, the smallest within "
        f"{float(gate.get('size_tolerance_pts', 1.0)):g} NDCG@10 pt of the best; the tolerance widens to "
        f"{float(gate.get('slow_tolerance_pts', 3.0)):g} pt when the best model's cold pass exceeds "
        f"{float(gate.get('slow_pass_hours', 2.0)):g} h. Decision: {gate.get('decision', '-')}."
    )


def _p0_snapshot(engine: Any) -> Any:
    return next(
        (d for d in getattr(engine, "_snapshots", {}).values() if getattr(d.snapshot, "repo_id", "-") == "-"),
        None,
    )


def describe(engine: Any) -> dict[str, Any]:
    config = engine.config
    retrieve = config.section("retrieve")
    lexical = config.section("lexical")
    p0 = _p0_snapshot(engine)
    ranker = engine.ranker
    encoders = []
    if engine.encoder is not None:
        encoders.append(
            _encoder_entry(engine.encoder, role="primary dense channel", key=str(config.get("model.encoder")))
        )
    encoders.append(_second_encoder(engine, p0))

    from acis.rank.candidates import GROUPS  # noqa: PLC0415

    ranker_info: dict[str, Any] = {"loaded": ranker is not None}
    if ranker is not None:
        own = set(ranker.feature_names)
        path = Path(str(config.get("rank.model", "")))
        ranker_info.update(
            {
                "kind": "LightGBM LambdaRank",
                "path": str(path),
                "features": len(ranker.feature_names),
                "groups": {g: list(names) for g, names in GROUPS.items() if set(names) <= own},
                "rounds": ranker.rounds,
                "trained_on_queries": ranker.meta.get("n_groups"),
                "group_dropout": ranker.meta.get("dropout"),
                "abstain_below": ranker.rho,
            }
        )
    symbols = getattr(p0, "symbols", None)
    bakeoff = _bakeoff()
    return {
        "device": {
            "device": "cpu",
            "numeric_profile": config.numeric_profile,
            "threads": engine.threads,
            "blas_threads": engine.blas_threads,
            "gpu_at_query_time": False,
            "network_at_query_time": False,
        },
        "encoders": encoders,
        "lexical": {
            "bm25": {
                "method": lexical.get("method"),
                "k1": lexical.get("k1"),
                "b": lexical.get("b"),
                "tokenizer": lexical.get("tokenizer"),
                "stemmer": lexical.get("stemmer"),
            },
            "symbols": {
                "distinct_symbols": len(symbols.postings) if symbols is not None else None,
                "defined_names": len(symbols.defined) if symbols is not None else None,
            },
        },
        "candidates": {
            "dense_k": retrieve.get("dense_k"),
            "lexical_k": retrieve.get("lexical_k"),
            "symbol_k": retrieve.get("symbol_k"),
            "aux_k": retrieve.get("aux_k", 0),
            "union_cap": retrieve.get("union_cap"),
            "search": "exact (one matrix product over every unit; no approximate index)",
        },
        "ranker": ranker_info,
        "confidence": {"calibrated": engine.calibration is not None},
        "corpus": {
            "loaded": p0 is not None,
            "units": p0.size if p0 is not None else 0,
            "snapshot": p0.snapshot.snapshot_id if p0 is not None else None,
        },
        "bakeoff": bakeoff,
        "bakeoff_html": render_bakeoff(bakeoff, selected=str(config.get("model.encoder")), gate=_gate_gm()),
        "config_hash": str(config.config_hash)[:12],
    }


__all__ = ["SECOND_ENCODER", "describe", "render_bakeoff"]
