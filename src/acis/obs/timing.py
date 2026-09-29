"""Per-request stage timings (ms), shared by every layer that does timed work.

A ContextVar, so concurrent API requests never mix their clocks. It lives in `obs` rather than in the engine so
the encoder runtime can report the time a query spent *queued* for the model separately from the time the model
spent on it: folding the two together is how a contended request came to look like a slow encoder.

Stages are **exclusive**: a stage opened inside another is subtracted from its parent, so the stages of one
request add up to the time they cover and a bar chart of them never counts a millisecond twice.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

STAGES: ContextVar[dict[str, float] | None] = ContextVar("acis_stages", default=None)
_OPEN: ContextVar[tuple[str, ...]] = ContextVar("acis_open_stages", default=())


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time one pipeline stage into the current request's `timings_ms`; free when no request is timing."""
    sink = STAGES.get()
    if sink is None:
        yield
        return
    parents = _OPEN.get()
    token = _OPEN.set((*parents, name))
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        _OPEN.reset(token)
        sink[name] = sink.get(name, 0.0) + elapsed
        if parents:
            sink[parents[-1]] = sink.get(parents[-1], 0.0) - elapsed


__all__ = ["STAGES", "stage"]
