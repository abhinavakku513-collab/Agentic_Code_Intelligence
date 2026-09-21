"""Paired bootstrap over queries (docs/spec/03 §7, docs/spec/09 §2).

The gate rule everywhere in ACIS is the same: **ΔNDCG@10 ≥ +0.5 pt and the 95 % CI lower bound > 0**, over the
declared decision set. This module is the only implementation of that rule, so no gate can quietly use a different
one. It resamples *queries* (not scores), pairs the two systems on the same resample, and is seeded, so the same
inputs always give the same verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260920
PRACTICAL_THRESHOLD_PTS = 0.5  # NDCG points, docs/spec/03 §7


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """All numbers are in **points** (0–100), the unit every gate is written in."""

    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    resamples: int
    seed: int

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0

    def passes(self, threshold_pts: float = PRACTICAL_THRESHOLD_PTS) -> bool:
        """The frozen gate rule: practical size **and** statistical confidence."""
        return self.delta >= threshold_pts and self.ci_excludes_zero

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "n": self.n,
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "delta": round(self.delta, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "p_two_sided": round(self.p_two_sided, 6),
            "resamples": self.resamples,
            "seed": self.seed,
            "ci_excludes_zero": self.ci_excludes_zero,
        }


def aligned_vectors(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-query vectors restricted to the queries both systems scored, in a deterministic order."""
    keys = sorted(set(a) & set(b))
    return keys, np.array([a[k] for k in keys], dtype=np.float64), np.array([b[k] for k in keys], dtype=np.float64)


def paired_bootstrap(
    system: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    scale: float = 100.0,
    ci: float = 95.0,
) -> BootstrapResult:
    """Paired percentile bootstrap of `mean(system) - mean(baseline)`, reported in points."""
    keys, xa, xb = aligned_vectors(system, baseline)
    n = len(keys)
    if n == 0:
        return BootstrapResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, resamples, seed)

    diff = (xa - xb) * scale
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diff[idx].mean(axis=1)

    lo = float(np.percentile(means, (100.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 - (100.0 - ci) / 2.0))
    # Two-sided bootstrap p-value: how often the resampled mean falls on the other side of zero.
    centred = means - observed
    p = float((np.abs(centred) >= abs(observed)).mean())
    return BootstrapResult(
        n=n,
        mean_a=float(xa.mean() * scale),
        mean_b=float(xb.mean() * scale),
        delta=observed,
        ci_low=lo,
        ci_high=hi,
        p_two_sided=p,
        resamples=resamples,
        seed=seed,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjustment for the ≤ 10 comparisons a milestone is allowed (docs/spec/06, blueprint §6.4)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "PRACTICAL_THRESHOLD_PTS",
    "BootstrapResult",
    "aligned_vectors",
    "holm_adjust",
    "paired_bootstrap",
]
