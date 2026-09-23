"""Request and response schemas for the HTTP surface (docs/spec/06 §2, `.claude/rules/security.md`).

Every field that crosses the network is validated by pydantic before it reaches the engine — lengths, ranges and
enumerations — because the API is the one place where input arrives from somebody who has read none of the rules.
The bounds are the engine's own (`top_k ≤ 1000`, `MAX_QUERY_CHARS`), stated once here so a rejection is a 422 with
a reason rather than a 500 with a traceback.

There is deliberately **no evaluation endpoint** (docs/spec/06 §1): scoring reads labels, and nothing that reads
labels is reachable over a socket.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_CHARS = 16_000
MAX_TOP_K = 1000


class SearchBody(BaseModel):
    """`POST /v1/search`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    repo_id: str = Field(default="-", max_length=64)
    version: str = Field(default="latest", max_length=128)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    mode: Literal["auto", "dense", "lexical", "hybrid"] = "auto"
    explain: bool = False


class EvolveBody(BaseModel):
    """`POST /v1/evolve` — the Bonus surface."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    repo_id: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=10, ge=1, le=100)
    flat: bool = False
    prefer: Literal["best", "latest"] = "best"


class IngestBody(BaseModel):
    """`POST /v1/repos/{repo_id}/ingest`. Local paths only: the service never fetches anything."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["jsonl", "dir", "zip", "git"] = "jsonl"
    location: str = Field(min_length=1, max_length=4096)
    extensions: list[str] = Field(default_factory=list, max_length=16)
    rev: str = Field(default="", max_length=128)


class UnitOut(BaseModel):
    unit_id: str
    key: str
    body_hash: str
    n_bytes: int


class HitOut(BaseModel):
    rank: int
    score: float
    unit: UnitOut
    source: str
    signals: dict[str, float] = Field(default_factory=dict)


class SnapshotOut(BaseModel):
    id: str
    version: str
    complete: bool
    missing_channels: list[str] = Field(default_factory=list)


class SearchOut(BaseModel):
    snapshot: SnapshotOut
    results: list[HitOut]
    route: str
    confidence: str
    no_strong_match: bool
    timings_ms: dict[str, float]
    degradations: list[str]
    interpreted_intent: dict[str, Any] = Field(default_factory=dict)


class EvolveOut(BaseModel):
    groups: list[dict[str, Any]]
    flat_results: list[HitOut] = Field(default_factory=list)
    degradations: list[str] = Field(default_factory=list)


class VersionOut(BaseModel):
    label: str
    snapshot_id: str | None
    n_units: int
    state: str
    active: bool


class DiffOut(BaseModel):
    repo_id: str
    a: str
    b: str
    added: list[str]
    removed: list[str]
    changed: list[str]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    encoder: str
    submission_capable: bool
    numeric_profile: str
    threads: int
    repos: list[str] = Field(default_factory=list)


__all__ = [
    "MAX_QUERY_CHARS",
    "MAX_TOP_K",
    "DiffOut",
    "EvolveBody",
    "EvolveOut",
    "HealthOut",
    "HitOut",
    "IngestBody",
    "SearchBody",
    "SearchOut",
    "SnapshotOut",
    "UnitOut",
    "VersionOut",
]
