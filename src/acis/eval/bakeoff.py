"""The encoder bake-off and gate G-M (D4, docs/spec/02 §3, docs/spec/03 §7, `configs/gates/G-M.yaml`).

The rule this implements is deliberately not "pick the best score":

    baseline  = argmax NDCG@10 over the candidates that could actually ship, measured on all 5,000 dev queries
    tolerance = 1.0 pt, or 3.0 pt when the baseline's projected cold pass exceeds 2 h
    eligible  = candidates within `tolerance` of the baseline, inside the D4 envelope, under the cold-pass SLO
    winner    = fewest parameters among the eligible; ties broken by higher NDCG@10

Resource use is scored by the organisers (FAQ), so a slightly weaker model that is much smaller and much faster is
often the better submission. The tolerance widening is the interesting half: when the best candidate is so slow
that its cold pass would dominate the run, the rule deliberately accepts a larger accuracy sacrifice.

A **non-permissively licensed** candidate is measured and reported — it tells us what the permissive-only rule
costs us — but can never be selected (D4, spec 02 §3). That is why the baseline is the best *selectable*
candidate rather than the best measured one: otherwise measuring a strong reference model would push every
shippable candidate out of tolerance and change the answer, which is exactly what a reference measurement must
never do. The gap between the two is reported as `reference_gap_pts`, since that gap is the thing the reference
measurement exists to produce.

The decision is a pure function of the measured table, so it is tested exhaustively without a single model. The
measuring is the part that needs weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from acis.core.errors import InvalidInput
from acis.core.paths import repo_path
from acis.embed.registry import ModelCard, available_cards, licence_is_permissive, load_card
from acis.embed.scorecard import Scorecard

GATE_CONFIG = ("configs", "gates", "G-M.yaml")
FULL_DEV_POOL = 5000


@dataclass(frozen=True, slots=True)
class GateRule:
    """The frozen G-M thresholds, read from `configs/gates/G-M.yaml` rather than hard-coded."""

    size_tolerance_pts: float = 1.0
    slow_pass_hours: float = 2.0
    slow_tolerance_pts: float = 3.0
    slo_cold_pass_hours: float = 4.0
    max_params_b: float = 1.0
    require_permissive: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None) -> GateRule:
        target = Path(path) if path else repo_path(*GATE_CONFIG)
        if not target.is_file():
            return cls()
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        envelope = raw.get("envelope") or {}
        return cls(
            size_tolerance_pts=float(raw.get("size_tolerance_pts", 1.0)),
            slow_pass_hours=float(raw.get("slow_pass_hours", 2.0)),
            slow_tolerance_pts=float(raw.get("slow_tolerance_pts", 3.0)),
            slo_cold_pass_hours=float(raw.get("slo_cold_pass_hours", 4.0)),
            max_params_b=float(envelope.get("max_params_b", 1.0)),
            require_permissive=bool(envelope.get("permissive_licence", True)),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One measured encoder: what it scored, what it cost, and whether it is allowed to ship."""

    key: str
    name: str
    ndcg_at_10: float
    params: int | None
    scorecard: Scorecard | None = None
    permissive: bool = True
    pinned: bool = True
    reference_only: bool = False
    n_queries: int = FULL_DEV_POOL
    ledger_run_id: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def params_b(self) -> float | None:
        return None if self.params is None else self.params / 1e9

    @property
    def projected_hours(self) -> float | None:
        return self.scorecard.projected_cold_pass_hours if self.scorecard else None


@dataclass(frozen=True, slots=True)
class GateDecision:
    """What G-M decided, and — as importantly — why each candidate was or was not eligible."""

    winner: str | None
    #: The best candidate *measured*, licence and eligibility aside. Reported because the table has to say so.
    best: str | None
    tolerance_pts: float
    tolerance_widened: bool
    eligible: tuple[str, ...]
    rejected: Mapping[str, str]
    default_applied: bool
    rule: GateRule
    notes: str = ""
    #: The best *selectable* candidate — the one the tolerance is measured from. Usually the same as `best`.
    baseline: str | None = None
    #: NDCG@10 points between `best` and `baseline` — exactly the number reference models are measured to learn:
    #: what the permissive-only, inside-the-envelope rule costs us (D4, spec 02 §3).
    reference_gap_pts: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "best": self.best,
            "baseline": self.baseline,
            "tolerance_pts": self.tolerance_pts,
            "tolerance_widened": self.tolerance_widened,
            "eligible": list(self.eligible),
            "rejected": dict(self.rejected),
            "default_applied": self.default_applied,
            "rule": asdict(self.rule),
            "notes": self.notes,
            "reference_gap_pts": self.reference_gap_pts,
        }


def decide(
    candidates: Sequence[Candidate],
    *,
    rule: GateRule | None = None,
    frozen_default: str = "qwen3-embedding-0.6b",
    decision_set_size: int = FULL_DEV_POOL,
) -> GateDecision:
    """Apply the D4 rule to a measured table. Pure: same table in, same decision out."""
    rule = rule or GateRule.load()
    if not candidates:
        raise InvalidInput("G-M needs at least one measured candidate")

    # Selectability first, **then** the tolerance baseline.
    #
    # A reference-only or non-permissive model is measured to quantify what the permissive-only rule costs us
    # (spec 02 §3). If it also set the bar, measuring a strong one would push every shippable candidate out of
    # tolerance and force the frozen default — so adding a reference measurement would change which shippable
    # model we pick, which is exactly backwards. The baseline is therefore the best *selectable* candidate, and
    # the reference gap is reported separately.
    rejected: dict[str, str] = {}
    selectable: list[Candidate] = []
    for c in candidates:
        reason = _unselectable_reason(c, rule=rule, decision_set_size=decision_set_size)
        if reason:
            rejected[c.key] = reason
        else:
            selectable.append(c)

    measured = [c for c in candidates if c.n_queries >= decision_set_size]
    best_overall = max(measured, key=lambda c: (c.ndcg_at_10, -(c.params or 0))) if measured else None

    if not selectable:
        return GateDecision(
            winner=frozen_default,
            best=best_overall.key if best_overall else None,
            tolerance_pts=rule.size_tolerance_pts,
            tolerance_widened=False,
            eligible=(),
            rejected=rejected,
            default_applied=True,
            rule=rule,
            notes="no candidate was both measured on the full decision set and inside the envelope",
        )

    baseline = max(selectable, key=lambda c: (c.ndcg_at_10, -(c.params or 0)))
    widened = baseline.projected_hours is not None and baseline.projected_hours > rule.slow_pass_hours
    tolerance = rule.slow_tolerance_pts if widened else rule.size_tolerance_pts

    eligible: list[Candidate] = []
    for c in selectable:
        gap = baseline.ndcg_at_10 - c.ndcg_at_10
        if gap > tolerance:
            rejected[c.key] = f"{gap:.2f} pt behind the best selectable candidate, tolerance is {tolerance} pt"
        else:
            eligible.append(c)

    # Smallest wins; a tie on size is broken by accuracy (spec 02 §3). Unknown size sorts last: an unmeasured
    # parameter count is not a small one.
    winner = min(eligible, key=lambda c: (c.params if c.params is not None else 1 << 62, -c.ndcg_at_10))
    reference_gap = round(best_overall.ndcg_at_10 - baseline.ndcg_at_10, 4) if best_overall else 0.0
    return GateDecision(
        winner=winner.key,
        best=best_overall.key if best_overall else baseline.key,
        baseline=baseline.key,
        reference_gap_pts=max(0.0, reference_gap),
        tolerance_pts=tolerance,
        tolerance_widened=widened,
        eligible=tuple(c.key for c in eligible),
        rejected=rejected,
        default_applied=winner.key == frozen_default,
        rule=rule,
        notes=(
            f"baseline={baseline.key} at {baseline.ndcg_at_10:.4f}; winner within {tolerance} pt with "
            f"{winner.params if winner.params is not None else 'unknown'} parameters"
            + (
                f"; the best measured candidate is {reference_gap:.2f} pt ahead but cannot ship"
                if reference_gap > 0
                else ""
            )
        ),
    )


def _unselectable_reason(c: Candidate, *, rule: GateRule, decision_set_size: int) -> str:
    """Why a candidate can never be selected, independent of how anything else scored.

    Kept separate from the tolerance comparison on purpose: these are properties of the candidate alone, so they
    can be evaluated before a baseline exists — which is what stops a reference model from setting the bar.
    """
    if c.n_queries < decision_set_size:
        return f"measured on {c.n_queries} queries, not the full decision set ({decision_set_size})"
    if c.reference_only:
        return "measured as reference only"
    if rule.require_permissive and not c.permissive:
        return "non-permissive licence: measured for reference, never shipped (D4)"
    if not c.pinned:
        return "not pinned to a commit (G0.4 pins it)"
    if c.params_b is not None and c.params_b > rule.max_params_b:
        return f"{c.params_b:.2f}B parameters exceeds the {rule.max_params_b}B envelope"
    if c.projected_hours is not None and c.projected_hours > rule.slo_cold_pass_hours:
        return f"projected cold pass {c.projected_hours} h exceeds the {rule.slo_cold_pass_hours} h SLO"
    return ""


# -- running the bake-off ------------------------------------------------------------------------------------------
def candidate_from_run(
    card: ModelCard,
    metrics: Mapping[str, float],
    scorecard: Scorecard | None,
    *,
    n_queries: int,
    ledger_run_id: str = "",
    reference_only: bool = False,
    params: int | None = None,
) -> Candidate:
    """Assemble a `Candidate` from one measured run, taking licence and pinning from the card."""
    return Candidate(
        key=card.key,
        name=card.name,
        ndcg_at_10=float(metrics.get("ndcg_at_10", 0.0)) * 100.0,  # the rule is written in points
        params=params,
        scorecard=scorecard,
        permissive=licence_is_permissive(card),
        pinned=card.is_pinned,
        reference_only=reference_only,
        n_queries=n_queries,
        ledger_run_id=ledger_run_id,
        metrics=dict(metrics),
    )


#: Documents plus queries in the official pass (D1). The cold-pass projection extrapolates a measured rate to it.
OFFICIAL_ROWS = 8_765 + 3_765
#: Rows the scorecard times. Large enough to amortise a batch, small enough that measuring is not the bake-off.
SCORECARD_SAMPLE = 128


def measure_card(
    key: str,
    *,
    query_ids: Sequence[str] | None = None,
    limit: int = 0,
    top_k: int = 1000,
    config_path: str = "configs/dev.yaml",
    reference_only: bool = False,
    record_row: bool = True,
    sample: int = SCORECARD_SAMPLE,
) -> Candidate:
    """Measure one candidate end to end: accuracy on the dev queries, cost on the declared host.

    Both halves run through the *shipping* path — the factory, the engine, the same preprocessing — so what is
    measured is the system, not a notebook approximation of it. The cost half deliberately runs with the vector
    cache switched off: a cold pass whose vectors are already on disk is a warm pass wearing its name (D17).
    """
    from acis.appsdata import apps  # noqa: PLC0415
    from acis.core.config import load_frozen_config  # noqa: PLC0415
    from acis.core.numeric import resolve_threads  # noqa: PLC0415
    from acis.embed.factory import build_encoder, profile_of  # noqa: PLC0415
    from acis.embed.scorecard import measure  # noqa: PLC0415
    from acis.engine import AcisEngine  # noqa: PLC0415
    from acis.eval import ladder  # noqa: PLC0415
    from acis.eval.dev_task import dev_qrels  # noqa: PLC0415
    from acis.eval.metrics import K_VALUES, score_run  # noqa: PLC0415
    from acis.rank.compose import rank_derived_scores  # noqa: PLC0415

    config = load_frozen_config(config_path).with_overrides(**{"model.encoder": key})
    encoder = build_encoder(config)  # NotReady here means the weights are not fetched yet (G0.4)
    card = load_card(key) if key in available_cards() else None

    ids = list(query_ids) if query_ids is not None else list(apps.dev_query_ids())
    if limit > 0:
        ids = ids[:limit]
    corpus = apps.load_corpus()
    queries = apps.load_queries()

    # -- cost, on an uncached encoder so "cold" means cold --------------------------------------------------
    cold_encoder = build_encoder(config, cache=False)
    scorecard = measure(
        lambda texts: cold_encoder.encode(texts, is_query=False),
        [d.text for d in corpus[:sample]],
        model=getattr(encoder, "name", key),
        numeric_profile=profile_of(config),
        threads=resolve_threads(config.get("run.threads", "auto_physical")),
        params=getattr(encoder, "n_parameters", None),
        model_mb=getattr(encoder, "size_mb", None),
        total_rows=OFFICIAL_ROWS,
        notes=f"scorecard sample n={min(sample, len(corpus))}; projection to the official {OFFICIAL_ROWS} rows",
    )

    # -- accuracy, through the engine the submission would use ----------------------------------------------
    engine = AcisEngine.from_config(config, encoder=encoder)
    snapshot = engine.build_snapshot(corpus, source=f"bakeoff:{key}")
    started = time.perf_counter()
    ranked = engine.search_batch(snapshot, ids, [queries[q] for q in ids], top_k=top_k)
    seconds = time.perf_counter() - started
    run = {qid: rank_derived_scores(hits, top_k) for qid, hits in ranked.items()}
    metrics = score_run(dev_qrels(ids), run, K_VALUES)

    run_id = ""
    if record_row:
        result = ladder.LadderResult(
            rung=f"bakeoff:{key}",
            run=run,
            metrics=metrics,
            seconds=seconds,
            n_queries=len(ids),
            n_docs=len(corpus),
            notes="G-M bake-off; dev split only",
            extra={"scorecard": scorecard.as_row()},
        )
        kind = "gate" if len(ids) >= FULL_DEV_POOL else "dev"
        run_id = ladder.record(result, kind=kind, extra={"model": key, "scorecard": scorecard.as_row()})

    # A stand-in encoder is measured like anything else and can never be selected: it says so itself.
    cannot_ship = reference_only or not getattr(encoder, "submission_capable", False)
    if card is not None:
        return candidate_from_run(
            card,
            metrics,
            scorecard,
            n_queries=len(ids),
            ledger_run_id=run_id,
            reference_only=cannot_ship,
            params=scorecard.params,
        )
    return Candidate(
        key=key,
        name=getattr(encoder, "name", key),
        ndcg_at_10=float(metrics.get("ndcg_at_10", 0.0)) * 100.0,
        params=scorecard.params,
        scorecard=scorecard,
        permissive=True,
        pinned=False,
        reference_only=cannot_ship,
        n_queries=len(ids),
        ledger_run_id=run_id,
        metrics=dict(metrics),
    )


def run_gate_m(
    keys: Sequence[str],
    *,
    limit: int = 0,
    rule: GateRule | None = None,
    reference_only: Sequence[str] = (),
    record_row: bool = True,
) -> tuple[list[Candidate], GateDecision]:
    """Measure every named candidate and apply the rule. One command, so G-M is a run rather than a project."""
    candidates = [
        measure_card(key, limit=limit, reference_only=key in set(reference_only), record_row=record_row) for key in keys
    ]
    return candidates, decide(candidates, rule=rule)


def record_decision(decision: GateDecision, *, path: str | Path | None = None) -> Path:
    """Write `configs/gates/G-M.yaml`. A decided gate is never edited — it is superseded via an ADR."""
    target = Path(path) if path else repo_path(*GATE_CONFIG)
    existing = yaml.safe_load(target.read_text(encoding="utf-8")) if target.is_file() else {}
    if existing and existing.get("status") == "decided":
        raise InvalidInput(
            "G-M is already decided; supersede it with an ADR rather than editing it",
            current=existing.get("decision"),
        )
    payload = {
        **(existing or {}),
        "status": "decided",
        "decision": decision.winner,
        "evidence": decision.as_dict(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def render_table(candidates: Sequence[Candidate], decision: GateDecision) -> str:
    """The B2 table: every candidate, its score, its cost, and its fate."""
    lines = [
        "| candidate | NDCG@10 (pt) | params | projected cold pass (h) | licence | outcome |",
        "|---|---|---|---|---|---|",
    ]
    for c in sorted(candidates, key=lambda x: -x.ndcg_at_10):
        if c.key == decision.winner:
            outcome = "**selected**"
        elif c.key in decision.eligible:
            outcome = "eligible"
        else:
            outcome = decision.rejected.get(c.key, "–")
        lines.append(
            f"| {c.name} | {c.ndcg_at_10:.2f} | {c.params or '–'} | "
            f"{c.projected_hours if c.projected_hours is not None else '–'} | "
            f"{'permissive' if c.permissive else 'non-permissive'} | {outcome} |"
        )
    return "\n".join(lines)


def run_bakeoff(
    cards: Sequence[ModelCard],
    evaluate: Callable[[ModelCard], tuple[Mapping[str, float], Scorecard | None, int]],
    *,
    rule: GateRule | None = None,
    reference_only: Sequence[str] = (),
) -> tuple[list[Candidate], GateDecision]:
    """Measure every candidate, then decide.

    `evaluate` is injected: it is the part that needs weights, and keeping it outside means the rule and the table
    are testable today and the measuring pass can be swapped for the real one without touching either.
    """
    candidates: list[Candidate] = []
    for card in cards:
        metrics, scorecard, n_queries = evaluate(card)
        candidates.append(
            candidate_from_run(
                card, metrics, scorecard, n_queries=n_queries, reference_only=card.key in set(reference_only)
            )
        )
    return candidates, decide(candidates, rule=rule)


__all__ = [
    "FULL_DEV_POOL",
    "GATE_CONFIG",
    "Candidate",
    "GateDecision",
    "GateRule",
    "candidate_from_run",
    "decide",
    "record_decision",
    "render_table",
    "run_bakeoff",
]
