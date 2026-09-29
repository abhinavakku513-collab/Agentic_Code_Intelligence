"""Calibrated confidence (`acis.rank.confidence`): monotone, never certain, never reorders anything."""

from __future__ import annotations

import random

from acis.rank.confidence import MIN_BLOCK, band, isotonic, lookup, z_top1


def test_the_calibration_is_monotone_and_never_claims_certainty():
    rng = random.Random(0)
    xs = [rng.gauss(0, 1) for _ in range(600)]
    ys = [int(rng.random() < min(0.95, max(0.02, 0.3 + 0.25 * x))) for x in xs]
    knots, values = isotonic(xs, ys)
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))
    assert values[0] > 0.0 and values[-1] < 1.0
    # A handful of lucky points at the top cannot read as P = 1: every block rests on MIN_BLOCK points.
    knots, values = isotonic([*xs, 9.0, 9.1, 9.2], [*ys, 1, 1, 1])
    assert lookup(knots, values, 9.5) < 1.0


def test_z_is_about_this_query_alone():
    crowd = [0.60 + 0.001 * i for i in range(100)]
    assert z_top1(crowd, 0.70) > 1.0
    assert z_top1([0.5] * 100, 0.5) == 0.0
    assert band(0.61) == "high" and band(0.3) == "medium" and band(0.1) == "low" and band(float("nan")) == "low"
    assert MIN_BLOCK >= 10
