"""Routing v1.1 — deciding which pipeline a query deserves (docs/spec/10 §4, D18, INV-15).

The learned ranker is trained on APPS problem statements. On one it is worth nearly two NDCG@10 points; on a
short hands-on question it can be worth less than nothing, because the shape it learned is not the shape it is
being shown. Measured, not assumed: "compute n factorial modulo 10^9+7" ranks better under the frozen dense
order than under the ranker.

So the route decides, and it decides on two signals that are both properties of the query alone:

* **the OOD score** — the mean cosine of the query's embedding to its `k` nearest TRAIN-query embeddings. A
  query that looks like the ones the ranker was trained on scores high; anything else scores low.
* **feature availability** — the fraction of bridge feature groups that actually fired. A query with no numbers,
  no quoted output and no structural cues gives the ranker nothing to work with, whatever it resembles.

`statement_like` needs **both**. Everything else, and every routing failure, takes the generic path, where the
frozen dense order stands. That is R-Q2 in the spec, and it is the difference between a system that is good at
one query shape and one that is not actively bad at the others.

**INV-15 and INV-3.** The bank is query-side only: it selects a pipeline and never offsets a document's score,
so it is not a reference-query offset and a ranking still depends on nothing but its own query. The bank is
built from TRAIN queries, which are the queries we are allowed to look at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Nearest neighbours averaged into the OOD score. Small enough that one twin query cannot carry a stranger in,
#: large enough that the score is not a single cosine.
DEFAULT_K = 8
#: Calibrated on the dev queries so that ≥ 95 % of them qualify (spec 10 §4); `scripts/bench/calibrate_route.py`
#: recomputes it and writes it back, and the value shipped is recorded in `configs/gates/G-OOD.yaml`.
DEFAULT_TAU = 0.55
#: Share of bridge feature groups that must have fired. Matches the ranker's own abstention floor.
DEFAULT_RHO = 0.35


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Why a query took the path it took. Returned to the caller, so a surprising ranking can be explained."""

    route: str
    ood_score: float
    availability: float
    tau: float
    rho: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "ood_score": round(self.ood_score, 4),
            "availability": round(self.availability, 4),
            "tau": self.tau,
            "rho": self.rho,
            "reason": self.reason,
        }


@dataclass(slots=True)
class QueryBank:
    """TRAIN-query embeddings, normalised, used for routing and for nothing else."""

    matrix: np.ndarray
    k: int = DEFAULT_K
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_vectors(cls, vectors: Sequence[np.ndarray] | np.ndarray, *, k: int = DEFAULT_K, **meta: Any) -> QueryBank:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("a query bank needs a non-empty 2-D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return cls(matrix=(matrix / norms).astype(np.float32), k=k, meta=dict(meta))

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])

    def ood_score(self, vector: np.ndarray) -> float:
        """Mean cosine to the `k` nearest bank entries. Higher means "this looks like what we trained on"."""
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return 0.0
        sims = self.matrix @ (query / norm)
        take = min(self.k, sims.shape[0])
        if take <= 0:
            return 0.0
        nearest = np.partition(sims, -take)[-take:]
        return float(nearest.mean())

    def save(self, path: Any) -> Any:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, self.matrix)
        return target

    @classmethod
    def load(cls, path: Any, *, k: int = DEFAULT_K, **meta: Any) -> QueryBank:
        return cls(matrix=np.load(path, mmap_mode=None).astype(np.float32), k=k, meta=dict(meta))


def decide(
    vector: np.ndarray | None,
    availability: float,
    bank: QueryBank | None,
    *,
    tau: float = DEFAULT_TAU,
    rho: float = DEFAULT_RHO,
) -> RouteDecision:
    """Routing v1.1. Anything missing or ambiguous takes the generic path — failure routes down, never up."""
    if bank is None or vector is None:
        return RouteDecision("generic", 0.0, availability, tau, rho, "no query bank: routing is unavailable")

    score = bank.ood_score(vector)
    if score < tau:
        return RouteDecision("generic", score, availability, tau, rho, "unlike the queries the ranker was trained on")
    if availability < rho:
        return RouteDecision("generic", score, availability, tau, rho, "too few feature groups fired")
    return RouteDecision("statement_like", score, availability, tau, rho, "statement-like and well covered")


def categorize(query: str, route: str, symbols: tuple[str, ...], *, weak_match: bool) -> str:
    """A descriptive category for the page and the logs — never a gate (every channel runs for every category).

    By shape only (INV-15): `statement_like` is the router's verdict; `exact_symbol` a short query that names a
    code symbol; `code` one written mostly in code syntax; `vague` a very short query with no symbol; `generic`
    anything else. `out_of_corpus` replaces the category when the calibrated confidence finds no strong match —
    it describes the answer, not the query.
    """
    words = query.split()
    code_chars = sum(query.count(c) for c in "=()[]{}:;<>+*/%")
    if weak_match and route != "statement_like":
        return "out_of_corpus"
    if route == "statement_like":
        return "statement_like"
    if symbols and len(words) <= 6:
        return "exact_symbol"
    if code_chars >= max(4, len(words) // 2):
        return "code"
    if len(words) <= 3:
        return "vague"
    return "generic"


__all__ = ["DEFAULT_K", "DEFAULT_RHO", "DEFAULT_TAU", "QueryBank", "RouteDecision", "categorize", "decide"]
