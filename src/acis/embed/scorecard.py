"""The resource scorecard (D2, docs/spec/02 §7, docs/spec/06 §6).

Running time, model size and GPU requirement are **scored** by the organisers (FAQ), so a candidate's cost is part
of its result, not a footnote to it. Every row here is measured on a declared host rather than estimated, and the
cold and warm figures are kept apart because conflating them is the easiest way to publish a number that cannot be
reproduced (D17).

Nothing in this module decides anything; it records. `acis.eval.bakeoff` applies the D4 rule to what it records.
"""

from __future__ import annotations

import gc
import resource
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

SECONDS_PER_HOUR = 3600.0
OFFICIAL_TOKENS = 3_000_000  # index 8,765 documents + 3,765 queries (docs/spec/02 §7)


@dataclass(frozen=True, slots=True)
class Scorecard:
    """One candidate's cost, in the shape the submission and the ledger both want."""

    model: str
    params: int | None
    model_mb: float | None
    numeric_profile: str
    peak_rss_mb: float
    cold_seconds: float
    warm_seconds: float
    latency_ms: dict[str, float]
    throughput_rows_per_s: float
    threads: int
    gpu: str = "none"
    notes: str = ""
    projected_cold_pass_hours: float | None = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def within_slo(self) -> bool | None:
        """`None` when the projection is unknown — an unknown cost is not a passing cost."""
        return None if self.projected_cold_pass_hours is None else self.projected_cold_pass_hours <= 4.0


@dataclass
class Timings:
    """Per-stage wall clock, collected by `stage()`."""

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start)

    @property
    def total(self) -> float:
        return sum(self.stages.values())


def peak_rss_mb() -> float:
    """Peak resident set of this process. `ru_maxrss` is KiB on Linux and bytes on macOS."""
    import sys  # noqa: PLC0415

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(raw) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(raw) / 1024.0


def percentiles(samples: Sequence[float], points: Sequence[int] = (50, 95, 99)) -> dict[str, float]:
    """Latency percentiles from measured samples. Empty input gives an empty dict, never a zero."""
    if not samples:
        return {}
    ordered = sorted(samples)
    out: dict[str, float] = {}
    for p in points:
        index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = round(ordered[index] * 1000.0, 3)
    return out


def project_cold_pass_hours(rows_per_second: float, *, total_rows: int) -> float | None:
    """Extrapolate the official cold pass from a measured rate. `None` when the rate is meaningless."""
    if rows_per_second <= 0:
        return None
    return round(total_rows / rows_per_second / SECONDS_PER_HOUR, 4)


def measure(
    encode: Callable[[Sequence[str]], Any],
    texts: Sequence[str],
    *,
    model: str,
    numeric_profile: str,
    threads: int,
    params: int | None = None,
    model_mb: float | None = None,
    single_query_samples: int = 20,
    total_rows: int = OFFICIAL_TOKENS // 250,
    notes: str = "",
) -> Scorecard:
    """Measure one candidate: a cold pass, a warm pass, and per-query latency.

    The cold pass runs first and deliberately: it is the number the submission quotes, and it only means anything
    before any cache is warm. Garbage is collected between the two so the warm figure is not flattered by memory
    the cold pass happened to leave behind.
    """
    gc.collect()
    cold_start = time.perf_counter()
    encode(list(texts))
    cold = time.perf_counter() - cold_start

    gc.collect()
    warm_start = time.perf_counter()
    encode(list(texts))
    warm = time.perf_counter() - warm_start

    samples: list[float] = []
    for i in range(min(single_query_samples, len(texts))):
        started = time.perf_counter()
        encode([texts[i]])
        samples.append(time.perf_counter() - started)

    rate = len(texts) / cold if cold > 0 else 0.0
    return Scorecard(
        model=model,
        params=params,
        model_mb=model_mb,
        numeric_profile=numeric_profile,
        peak_rss_mb=round(peak_rss_mb(), 2),
        cold_seconds=round(cold, 4),
        warm_seconds=round(warm, 4),
        latency_ms=percentiles(samples),
        throughput_rows_per_s=round(rate, 3),
        threads=threads,
        gpu="none",
        notes=notes,
        projected_cold_pass_hours=project_cold_pass_hours(rate, total_rows=total_rows),
    )


def render_table(cards: Sequence[Scorecard]) -> str:
    """The markdown table the scorecard section of the submission quotes."""
    header = (
        "| model | params | MB | profile | peak RSS (MB) | cold (s) | warm (s) | p95 (ms) | rows/s | "
        "projected cold pass (h) | GPU |"
    )
    lines = [header, "|" + "---|" * 11]
    for c in cards:
        lines.append(
            f"| {c.model} | {c.params or '–'} | {c.model_mb or '–'} | {c.numeric_profile} | {c.peak_rss_mb} | "
            f"{c.cold_seconds} | {c.warm_seconds} | {c.latency_ms.get('p95', '–')} | {c.throughput_rows_per_s} | "
            f"{c.projected_cold_pass_hours if c.projected_cold_pass_hours is not None else '–'} | {c.gpu} |"
        )
    return "\n".join(lines)


__all__ = [
    "OFFICIAL_TOKENS",
    "SECONDS_PER_HOUR",
    "Scorecard",
    "Timings",
    "measure",
    "peak_rss_mb",
    "percentiles",
    "project_cold_pass_hours",
    "render_table",
]
