"""Calibrated confidence — how likely the top result is relevant (docs/spec/02 §4 stage 11).

Before this module the engine reported confidence from the margin between the first two *final* scores. On the
hybrid channel those scores are rank-derived (0, −1, −2, …), so every query — "configure a kubernetes ingress"
against a corpus of programming-contest solutions included — came back `high`; on the dense channel the relative
cosine margin is tiny, so almost everything came back `low`. Neither said anything.

**The signal** is how far the served #1 stands out from this query's own top-100 dense cosines:

    z = (cos(served #1) − mean(top-100 cos)) / std(top-100 cos)

computed from this query alone (INV-3). A query whose answer is clear has a #1 far above its crowd; a query with no
relevant document — or the packed 0.580–0.596 band reported from the page — has a flat crowd.

**The calibration** maps z to P(served #1 is relevant) per route, by isotonic regression on data the ranking never
saw: APPS dev out-of-fold records for `statement_like`, CosQA's `valid` split for `generic`
(`scripts/bench/calibrate_confidence.py`, which records the fit and its held-out reliability in the ledger).
**The bands** are fixed, not tuned: `high` ≥ 0.6, `medium` ≥ 0.3, else `low`, and `low` sets `no_strong_match`.
Without a calibration file the engine says the confidence is uncalibrated rather than inventing one.

It is a report, never a ranking input: nothing here can reorder a result.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HIGH = 0.6
MEDIUM = 0.3
CROWD = 100


def z_top1(top_cosines: Sequence[float] | np.ndarray, served_top1_cosine: float) -> float:
    """How many standard deviations the served #1 sits above this query's own top-`CROWD` dense cosines."""
    values = np.asarray(top_cosines, dtype=np.float64)[:CROWD]
    if values.size < 2 or not math.isfinite(served_top1_cosine):
        return float("nan")
    sd = float(values.std())
    if sd <= 1e-9:
        return 0.0
    return (float(served_top1_cosine) - float(values.mean())) / sd


def isotonic(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators: a non-decreasing step function of x. Returns (knots, values) for `lookup`."""
    pairs = sorted((float(x), float(y)) for x, y in zip(xs, ys, strict=True) if math.isfinite(x))
    blocks: list[list[float]] = []  # [sum_y, count, max_x]
    for x, y in pairs:
        blocks.append([y, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, mx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] = mx
    return [b[2] for b in blocks], [b[0] / b[1] for b in blocks]


def lookup(knots: Sequence[float], values: Sequence[float], x: float) -> float:
    if not knots or not math.isfinite(x):
        return float("nan")
    i = bisect.bisect_left(list(knots), x)
    return float(values[min(i, len(values) - 1)])


def band(p: float) -> str:
    if not math.isfinite(p):
        return "low"
    return "high" if p >= HIGH else "medium" if p >= MEDIUM else "low"


@dataclass(frozen=True, slots=True)
class Calibration:
    """Per-route isotonic maps from z to P(served #1 relevant), with where each was fitted."""

    routes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    ledger_run_id: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        payload = json.loads(Path(path).read_text("utf-8"))
        return cls(routes=payload["routes"], ledger_run_id=str(payload.get("ledger_run_id", "")))

    def probability(self, route: str, z: float) -> float:
        fit = self.routes.get(route) or self.routes.get("generic")
        if not fit:
            return float("nan")
        return lookup(fit["knots"], fit["values"], z)

    def source(self, route: str) -> str:
        fit = self.routes.get(route) or self.routes.get("generic") or {}
        return str(fit.get("fitted_on", "unknown"))


__all__ = ["CROWD", "HIGH", "MEDIUM", "Calibration", "band", "isotonic", "lookup", "z_top1"]
