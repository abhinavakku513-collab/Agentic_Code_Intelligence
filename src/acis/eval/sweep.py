"""The query-representation sweep and gate G1 (docs/spec/02 §2 and §5, docs/spec/03 §7, INV-15).

G1 chooses three things **per route**: which task string a query is encoded with, which view it is built from, and
how many tokens survive truncation. It is deliberately not an argmax over the grid:

    best     = the highest-scoring cell
    pool     = every cell the paired bootstrap cannot separate from `best` (Δ ≥ 0.5 pt **and** CI lower > 0)
    winner   = the cheapest cell in the pool — ties in cost broken by accuracy, and then by the incumbent
    guard    = a cell costlier than the default is adopted only if it beats *the default* by the same rule

The guard is the part that earns its place. Running time is scored (FAQ), V2 costs two encodes per query, and a
plain argmax will spend that forever for a tenth of a point of noise. The last tie-break keeps the incumbent for
the same reason: changing the shipped configuration has a cost that no measurement shows.

Per route, because a hands-on question and a full problem statement do not want the same instruction (INV-15,
spec 10 §4). A decision for one route says nothing about the other, and the file holds both.

Everything above is a pure function of the measured table, so it is tested exhaustively without a model; the
measuring pass below needs weights and runs unchanged the moment they exist.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from acis.core.errors import InvalidInput
from acis.core.paths import repo_path
from acis.eval.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED, PRACTICAL_THRESHOLD_PTS, paired_bootstrap

GATE_CONFIG = ("configs", "gates", "G1.yaml")
TASKS = ("T1", "T2", "T3")
VIEWS = ("V0", "V1", "V2")
LENGTHS = (256, 512, 1024, 2048)
#: V1 is the same cap over shorter text, so it is never dearer than V0 at the same cap; V2 pays for two encodes.
VIEW_COST_RANK = {"V1": 0, "V0": 1, "V2": 2}


@dataclass(frozen=True, slots=True)
class Cell:
    """One point of the grid: a task string, a view and a truncation length."""

    task: str
    view: str
    max_tokens: int

    @property
    def key(self) -> str:
        return f"{self.task}/{self.view}/{self.max_tokens}"

    @property
    def encoded_tokens(self) -> int:
        """Tokens the encoder actually processes per query — V2 embeds two views (spec 02 §2)."""
        return self.max_tokens * (2 if self.view == "V2" else 1)

    @property
    def cost_rank(self) -> tuple[int, int]:
        return (self.encoded_tokens, VIEW_COST_RANK.get(self.view, 1))

    @property
    def head(self) -> int:
        return self.max_tokens * 3 // 4  # the 768/256 split of D3, held at every cap

    @property
    def tail(self) -> int:
        return self.max_tokens - self.head


@dataclass(frozen=True, slots=True)
class Measurement:
    """One cell, measured. `per_query` is what the paired bootstrap needs; the mean alone cannot be tested."""

    cell: Cell
    per_query: Mapping[str, float]
    metrics: Mapping[str, float]
    seconds: float
    ledger_run_id: str = ""

    @property
    def ndcg_pts(self) -> float:
        return float(self.metrics.get("ndcg_at_10", 0.0)) * 100.0


@dataclass(frozen=True, slots=True)
class G1Decision:
    """What G1 chose for one route, and why every other cell lost."""

    route: str
    winner: str
    baseline: str
    adopted: bool
    pool: tuple[str, ...]
    rejected: Mapping[str, str]
    comparisons: Mapping[str, Mapping[str, Any]]
    threshold_pts: float
    n_queries: int
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "winner": self.winner,
            "baseline": self.baseline,
            "adopted": self.adopted,
            "pool": list(self.pool),
            "rejected": dict(self.rejected),
            "comparisons": {k: dict(v) for k, v in self.comparisons.items()},
            "threshold_pts": self.threshold_pts,
            "n_queries": self.n_queries,
            "notes": self.notes,
        }


def decide_g1(
    measurements: Sequence[Measurement],
    *,
    route: str,
    baseline: str,
    threshold_pts: float = PRACTICAL_THRESHOLD_PTS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> G1Decision:
    """Apply the G1 rule to a measured grid. Pure: same table in, same decision out."""
    if len(measurements) < 2:
        raise InvalidInput("G1 compares cells: it needs at least two measured cells", n=len(measurements))
    queries = {tuple(sorted(m.per_query)) for m in measurements}
    if len(queries) != 1:
        raise InvalidInput(
            "every cell must be measured on the same queries, or the paired bootstrap compares different things",
            n_distinct_query_sets=len(queries),
        )
    by_key = {m.cell.key: m for m in measurements}
    if baseline not in by_key:
        raise InvalidInput(f"the baseline cell {baseline!r} is not in the measured table", cells=sorted(by_key))

    def compare(system: Measurement, against: Measurement) -> Any:
        return paired_bootstrap(system.per_query, against.per_query, resamples=resamples, seed=seed)

    base = by_key[baseline]
    best = max(measurements, key=lambda m: (m.ndcg_pts, -m.cell.encoded_tokens))

    rejected: dict[str, str] = {}
    pool: list[Measurement] = []
    for m in measurements:
        gap = compare(best, m)
        if m.cell.key != best.cell.key and gap.passes(threshold_pts):
            rejected[m.cell.key] = (
                f"{gap.delta:.2f} pt behind {best.cell.key}, which clears the {threshold_pts} pt gate"
            )
        else:
            pool.append(m)

    # A cell that costs more than the incumbent has to earn it against *the incumbent*, not against the best cell.
    eligible: list[Measurement] = []
    for m in pool:
        if m.cell.cost_rank > base.cell.cost_rank:
            lift = compare(m, base)
            if not lift.passes(threshold_pts):
                rejected[m.cell.key] = (
                    f"{lift.delta:+.2f} pt over the default for {m.cell.encoded_tokens} tokens against "
                    f"{base.cell.encoded_tokens}: below the {threshold_pts} pt gate for a costlier cell"
                )
                continue
        eligible.append(m)

    # Cheapest wins; then accuracy; then the incumbent, because changing the shipped configuration is itself a cost.
    winner = (
        min(eligible, key=lambda m: (m.cell.cost_rank, -m.ndcg_pts, m.cell.key != baseline, m.cell.key))
        if eligible
        else base
    )
    for m in measurements:
        if m.cell.key != winner.cell.key and m.cell.key not in rejected:
            rejected[m.cell.key] = (
                f"ties {winner.cell.key} but encodes {m.cell.encoded_tokens} tokens against "
                f"{winner.cell.encoded_tokens}"
                if m.cell.cost_rank > winner.cell.cost_rank
                else f"no better than the incumbent {winner.cell.key} at the same cost"
            )

    comparisons = {m.cell.key: compare(m, base).as_dict() for m in measurements}
    return G1Decision(
        route=route,
        winner=winner.cell.key,
        baseline=baseline,
        adopted=winner.cell.key != baseline,
        pool=tuple(m.cell.key for m in pool),
        rejected=rejected,
        comparisons=comparisons,
        threshold_pts=threshold_pts,
        n_queries=len(base.per_query),
        notes=(
            f"best={best.cell.key} at {best.ndcg_pts:.2f} pt; winner encodes {winner.cell.encoded_tokens} tokens "
            f"per query"
        ),
    )


def render_table(measurements: Sequence[Measurement], decision: G1Decision) -> str:
    """The G1 table: every cell, its score, what it costs, and its fate."""
    lines = [
        f"**G1 · route `{decision.route}`** (n={decision.n_queries}, threshold {decision.threshold_pts} pt)",
        "",
        "| cell | NDCG@10 (pt) | Δ vs default (pt) | tokens/query | outcome |",
        "|---|---|---|---|---|",
    ]
    for m in sorted(measurements, key=lambda x: -x.ndcg_pts):
        key = m.cell.key
        delta = float(decision.comparisons.get(key, {}).get("delta", 0.0))
        if key == decision.winner:
            outcome = "**selected**" + (" (baseline)" if key == decision.baseline else "")
        elif key == decision.baseline:
            outcome = "baseline · " + decision.rejected.get(key, "unchanged")
        else:
            outcome = decision.rejected.get(key, "in the pool")
        lines.append(f"| {key} | {m.ndcg_pts:.2f} | {delta:+.2f} | {m.cell.encoded_tokens} | {outcome} |")
    return "\n".join(lines)


def record_decision(decision: G1Decision, *, path: str | Path | None = None) -> Path:
    """Write one route's decision into `configs/gates/G1.yaml`. Written once per route; supersede via an ADR.

    Per route rather than per file: deciding `statement_like` must not lock `generic` out of ever being decided.
    """
    target = Path(path) if path else repo_path(*GATE_CONFIG)
    existing = yaml.safe_load(target.read_text(encoding="utf-8")) if target.is_file() else {}
    existing = existing or {}
    routes = dict(existing.get("routes") or {})
    if decision.route in routes:
        raise InvalidInput(
            f"G1 is already decided for route {decision.route!r}; supersede it with an ADR rather than editing it",
            current=routes[decision.route].get("winner"),
        )
    routes[decision.route] = {"winner": decision.winner, "evidence": decision.as_dict()}
    payload = {**existing, "status": "decided", "routes": routes}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


# -- the measuring pass ------------------------------------------------------------------------------------------
def default_grid(
    tasks: Sequence[str] = TASKS, views: Sequence[str] = VIEWS, lengths: Sequence[int] = LENGTHS
) -> list[Cell]:
    return [Cell(task=t, view=v, max_tokens=n) for t in tasks for v in views for n in lengths]


def measure_cell(
    cell: Cell,
    *,
    route: str,
    query_ids: Sequence[str],
    config_path: str = "configs/dev.yaml",
    top_k: int = 1000,
) -> Measurement:
    """Measure one cell through the shipping path: the engine, the same prep, the same metric code.

    The task string comes from the model card (`configs/models/<name>.yaml`, per D4) — the sweep chooses *which*
    task a route uses, never what it says.
    """
    from acis.appsdata import apps  # noqa: PLC0415
    from acis.core.config import load_frozen_config  # noqa: PLC0415
    from acis.embed.factory import build_encoder  # noqa: PLC0415
    from acis.engine import AcisEngine  # noqa: PLC0415
    from acis.eval.dev_task import dev_qrels  # noqa: PLC0415
    from acis.eval.metrics import K_VALUES, per_query, score_run  # noqa: PLC0415
    from acis.rank.compose import rank_derived_scores  # noqa: PLC0415

    config = load_frozen_config(config_path).with_overrides(
        **{
            "prep.query.view": cell.view,
            "prep.query.max_tokens": cell.max_tokens,
            "prep.query.head": cell.head,
            "prep.query.tail": cell.tail,
        }
    )
    encoder = build_encoder(config)
    with_task = getattr(encoder, "with_route_task", None)
    if with_task is not None:
        encoder = with_task(route, cell.task)

    engine = AcisEngine.from_config(config, encoder=encoder)
    snapshot = engine.build_snapshot(apps.load_corpus(), source=f"g1:{route}:{cell.key}")
    queries = apps.load_queries()
    ids = list(query_ids)
    started = time.perf_counter()
    ranked = engine.search_batch(snapshot, ids, [queries[q] for q in ids], top_k=top_k)
    seconds = time.perf_counter() - started

    run = {qid: rank_derived_scores(hits, top_k) for qid, hits in ranked.items()}
    qrels = dev_qrels(ids)
    return Measurement(
        cell=cell,
        per_query=per_query(qrels, run, "ndcg", 10),
        metrics=score_run(qrels, run, K_VALUES),
        seconds=seconds,
    )


def run_g1(
    route: str,
    *,
    query_ids: Sequence[str],
    grid: Sequence[Cell] | None = None,
    baseline: str = "T1/V0/1024",
    config_path: str = "configs/dev.yaml",
) -> tuple[list[Measurement], G1Decision]:
    """Measure the grid for one route and decide. The grid is a product, so keep it small on purpose."""
    cells = list(grid) if grid is not None else default_grid()
    measurements = [measure_cell(c, route=route, query_ids=query_ids, config_path=config_path) for c in cells]
    return measurements, decide_g1(measurements, route=route, baseline=baseline)


__all__ = [
    "GATE_CONFIG",
    "LENGTHS",
    "TASKS",
    "VIEWS",
    "Cell",
    "G1Decision",
    "Measurement",
    "decide_g1",
    "default_grid",
    "measure_cell",
    "record_decision",
    "render_table",
    "run_g1",
]
