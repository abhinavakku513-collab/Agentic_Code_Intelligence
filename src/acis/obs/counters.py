"""Degradation and fallback counters (INV-7).

Every fallback in ACIS goes through `degradation()`. It increments a named counter, records the event for
`diagnostics.degradations`, and — in strict mode — raises `StrictViolation` instead, so an official run can never
silently return a degraded ranking.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from acis.core.errors import StrictViolation


@dataclass
class Counters:
    """Thread-safe named counters plus an ordered, de-duplicated list of degradation events."""

    counts: Counter[str] = field(default_factory=Counter)
    events: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def incr(self, name: str, amount: int = 1) -> int:
        with self._lock:
            self.counts[name] += amount
            return self.counts[name]

    def note(self, event: str) -> None:
        with self._lock:
            if event not in self.events:
                self.events.append(event)

    def get(self, name: str) -> int:
        return int(self.counts.get(name, 0))

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self.counts)

    def degradations(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self.events)

    def reset(self) -> None:
        with self._lock:
            self.counts.clear()
            self.events.clear()

    def __iter__(self) -> Iterator[str]:
        return iter(self.snapshot())


_GLOBAL = Counters()


def global_counters() -> Counters:
    return _GLOBAL


def degradation(name: str, detail: str = "", *, strict: bool = False, counters: Counters | None = None) -> str:
    """Record one degradation. In strict mode this raises instead of returning (INV-7)."""
    sink = counters if counters is not None else _GLOBAL
    event = f"{name}:{detail}" if detail else name
    sink.incr(f"degradation.{name}")
    sink.note(event)
    if strict:
        raise StrictViolation(f"strict mode forbids the fallback {event!r}", degradation=event)
    return event


__all__ = ["Counters", "degradation", "global_counters"]
