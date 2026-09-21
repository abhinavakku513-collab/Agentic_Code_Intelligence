"""Core value types (docs/spec/06 §1). Plain frozen dataclasses: the engine never speaks mteb, HTTP or pydantic.

The `AcisEngine` interface is frozen at the end of Phase 1; these types are part of that contract, so additions are
backwards-compatible (new fields get defaults) and removals need an ADR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

BuildMode = Literal["eager_heads", "eager_all", "lazy"]
Route = Literal["statement_like", "generic"]
Confidence = Literal["high", "medium", "low"]
VersionSelectorKind = Literal["latest", "version", "snapshot", "commit", "as_of", "all", "range"]


@dataclass(frozen=True, slots=True)
class Snippet:
    """One indexable text with an opaque handle. Handles are never features (INV-4)."""

    handle: str
    text: str


@dataclass(frozen=True, slots=True)
class Unit:
    """A stored, content-addressed snippet inside a snapshot."""

    unit_id: str
    key: str
    body_hash: str
    repo_id: str = "-"
    version_id: str = "-"
    snapshot_id: str = "-"
    n_bytes: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    """One ranked result. `source` is re-read from the content store by hash (INV-1)."""

    rank: int
    score: float
    unit: Unit
    source: str
    signals: Mapping[str, float] = field(default_factory=dict)
    also_at: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Limits:
    max_units: int = 2_000_000
    max_bytes: int = 4 << 30
    max_unit_bytes: int = 4 << 20
    max_seconds: float = 3600.0


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Where units come from (Track B1 implements the loaders; the type is part of the frozen interface)."""

    kind: Literal["jsonl", "dir", "zip", "git", "memory"]
    location: str
    version: str | None = None
    digest: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RefSpec:
    refs: Sequence[str] = field(default_factory=tuple)
    include_all: bool = False


@dataclass(frozen=True, slots=True)
class JobHandle:
    job_id: str
    repo_id: str
    state: Literal["queued", "running", "done", "failed"] = "queued"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """An immutable, addressable view of a corpus. Only VALID snapshots are searchable (INV-9)."""

    snapshot_id: str
    repo_id: str
    version_id: str
    n_units: int
    config_hash: str
    source: str = ""
    state: Literal["BUILDING", "VALID", "PARTIAL", "RETIRED"] = "VALID"
    missing_channels: Sequence[str] = field(default_factory=tuple)
    created_ts: float = 0.0

    @property
    def complete(self) -> bool:
        return self.state == "VALID" and not self.missing_channels


@dataclass(frozen=True, slots=True)
class BuildReport:
    snapshot: Snapshot
    units_total: int
    units_new: int
    units_reused: int
    seconds: float
    degradations: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    repo_id: str = "-"
    version: str = "latest"
    top_k: int = 10
    mode: Literal["auto", "dense", "lexical", "hybrid"] = "auto"
    filters: Mapping[str, Any] = field(default_factory=dict)
    investigate: Literal["off", "auto", "on"] = "off"
    allow_partial: bool = False
    explain: bool = False
    diagnostics: bool = False


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    id: str
    version: str
    complete: bool
    missing_channels: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    snapshot: SnapshotRef
    results: Sequence[Hit]
    confidence: Confidence = "low"
    no_strong_match: bool = False
    route: Route = "generic"
    interpreted_intent: Mapping[str, Any] = field(default_factory=dict)
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    degradations: Sequence[str] = field(default_factory=tuple)
    investigation: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class VersionComparison:
    repo_id: str
    a: str
    b: str
    added: Sequence[str] = field(default_factory=tuple)
    removed: Sequence[str] = field(default_factory=tuple)
    changed: Sequence[str] = field(default_factory=tuple)
    query_effect: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvolveRequest:
    query: str
    repo_id: str = "-"
    top_k: int = 10
    flat: bool = False
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvolveResponse:
    groups: Sequence[Mapping[str, Any]]
    flat_results: Sequence[Hit] = field(default_factory=tuple)
    degradations: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Diagnostics:
    config_hash: str
    model_fingerprint: str
    numeric_profile: str
    hardware: Mapping[str, Any]
    tier: str
    degradations: Mapping[str, int] = field(default_factory=dict)
    snapshots: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    agent_calls: int = 0


@dataclass(frozen=True, slots=True)
class EvalSpec:
    kind: Literal["dev", "ladder", "gate", "robustness", "official"]
    config_path: str | None = None
    split: str = "train"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalReport:
    kind: str
    run_id: str
    metrics: Mapping[str, float]
    artifacts: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""


__all__ = [
    "BuildMode",
    "BuildReport",
    "Confidence",
    "Diagnostics",
    "EvalReport",
    "EvalSpec",
    "EvolveRequest",
    "EvolveResponse",
    "Hit",
    "JobHandle",
    "Limits",
    "RefSpec",
    "Route",
    "SearchRequest",
    "SearchResponse",
    "Snapshot",
    "SnapshotRef",
    "Snippet",
    "SourceSpec",
    "Unit",
    "VersionComparison",
    "VersionSelectorKind",
]
