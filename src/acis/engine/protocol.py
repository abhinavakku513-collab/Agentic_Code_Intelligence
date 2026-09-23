"""The `AcisEngine` interface — **frozen at the end of Phase 1** (docs/spec/06 §1, docs/TRIAGE.md).

Track B builds against this file. Changing a signature here needs an ADR, because a worktree that is three phases
behind must still compile against it. Methods whose implementation belongs to a later phase are part of the contract
now and raise `NotReady` until that phase lands — a missing method would silently change the shape of the interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from acis.core.types import (
    BuildMode,
    BuildReport,
    Diagnostics,
    EvalReport,
    EvalSpec,
    EvolveRequest,
    EvolveResponse,
    JobHandle,
    Limits,
    RefSpec,
    SearchRequest,
    SearchResponse,
    Snapshot,
    Snippet,
    SourceSpec,
    VersionComparison,
)


@runtime_checkable
class SearchEngine(Protocol):
    """Everything the API, the CLI, the demo and the mteb adapter are allowed to call."""

    # -- repositories and versions (Track B1) -----------------------------------------------------------------
    def ingest(self, source: SourceSpec, *, repo_id: str, limits: Limits | None = None) -> JobHandle: ...

    def index(self, repo_id: str, *, refs: RefSpec | None = None, mode: BuildMode = "eager_heads") -> BuildReport: ...

    def update_version(self, repo_id: str, delta: SourceSpec, *, expected_active: str | None = None) -> BuildReport: ...

    def compare_versions(self, repo_id: str, a: str, b: str, *, query: str | None = None) -> VersionComparison: ...

    # -- retrieval ---------------------------------------------------------------------------------------------
    def search(self, req: SearchRequest) -> SearchResponse: ...

    def search_version(self, repo_id: str, version: str, query: str, **kw: object) -> SearchResponse: ...

    def retrieve_evolution(self, req: EvolveRequest) -> EvolveResponse: ...

    # -- introspection and evaluation ----------------------------------------------------------------------------
    def diagnostics(self, repo_id: str | None = None) -> Diagnostics: ...

    def evaluate(self, spec: EvalSpec) -> EvalReport: ...

    # -- batch surface used by the mteb adapter (no mteb types cross this line, INV-11) ---------------------------
    def build_snapshot(self, docs: Sequence[Snippet], *, source: str) -> Snapshot: ...

    def search_batch(
        self,
        snap: Snapshot,
        ids: Sequence[str],
        texts: Sequence[str],
        *,
        top_k: int,
        restrict_to: Mapping[str, Sequence[str]] | None = None,
        strict: bool = False,
    ) -> dict[str, list[tuple[str, float]]]: ...


#: `docs/spec/07` calls this "the `AcisEngine` interface". `AcisEngine` is the implementation
#: (`acis.engine.core`); `SearchEngine` is the Protocol it satisfies. The alias exists so that a Track B engineer
#: grepping for the name the spec uses finds the frozen surface rather than only prose.
AcisEngineProtocol = SearchEngine

__all__ = ["AcisEngineProtocol", "SearchEngine"]
