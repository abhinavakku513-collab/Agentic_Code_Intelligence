"""Evolution-aware retrieval: ranking lineages instead of revisions (docs/spec/04 §6, D12, INV-2).

Searching every version of a corpus at once produces a result list that is mostly one thing repeated. Twenty
revisions of the same function, near-identical, crowding out the four genuinely different implementations that
should have been on the page. That is the failure the Bonus exists to fix, and the fix is not deduplication after
the fact — it is ranking the *lineage* and then showing its best revision.

Three decisions carry it:

* **One candidate per unique body.** Identical revisions are one document with a `present_in` span, so the
  expensive channel sees each distinct body once no matter how many versions carry it.
* **A lineage scores as its best member** (`max`), not its mean. A unit that was improved in v5 should be found
  by a query the v5 revision answers, and averaging in four older revisions would bury exactly that.
* **What is returned is the best revision plus its history** — the members, the version span, and a timeline of
  what changed. A lineage with an inferred (S4) link says so, because a guess that is presented as a fact is
  worse than the gap it filled.

Unlinked distinct implementations stay separate: the grouping can only ever merge what the cascade linked above
its confidence floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from acis.lineage.store import Lineage, LineageIndex


@dataclass(frozen=True, slots=True)
class RevisionHit:
    """One revision that the underlying ranking returned, before grouping."""

    version: str
    key: str
    body_hash: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class LineageHit:
    """One lineage as an answer: its best revision, every member, and how it got there."""

    lineage_id: str
    score: float
    best: RevisionHit
    members: tuple[RevisionHit, ...]
    first_seen: str
    last_seen: str
    inferred: bool = False
    timeline: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def n_revisions(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "score": round(self.score, 6),
            "best": {
                "version": self.best.version,
                "key": self.best.key,
                "body_hash": self.best.body_hash,
                "score": round(self.best.score, 6),
            },
            "span": [self.first_seen, self.last_seen],
            "n_revisions": self.n_revisions,
            "inferred": self.inferred,
            "members": [{"version": m.version, "key": m.key, "score": round(m.score, 6)} for m in self.members],
            "timeline": [dict(step) for step in self.timeline],
        }


def group(
    hits: Sequence[RevisionHit],
    index: LineageIndex,
    *,
    prefer: str = "best",
    top_k: int = 10,
) -> list[LineageHit]:
    """Group revision hits into lineage hits, highest-scoring lineage first.

    `prefer="latest"` breaks a tie towards the newest revision; `prefer="best"` keeps the highest-scoring one.
    Ties beyond that fall back to the lineage id, so the order never depends on dictionary iteration.
    """
    if prefer not in ("best", "latest"):
        raise ValueError(f"prefer must be 'best' or 'latest', not {prefer!r}")

    buckets: dict[str, list[RevisionHit]] = {}
    for hit in hits:
        lineage_id = index.of(hit.version, hit.key) or f"unlinked:{hit.version}:{hit.key}"
        buckets.setdefault(lineage_id, []).append(hit)

    out: list[LineageHit] = []
    for lineage_id, members in buckets.items():
        ordered = sorted(members, key=lambda m: (-m.score, m.rank))
        best = ordered[0]
        if prefer == "latest":
            top = max(m.score for m in members)
            latest = [m for m in members if m.score >= top - 1e-9]
            best = sorted(latest, key=lambda m: (m.version, m.rank))[-1]
        lineage: Lineage | None = index.get(lineage_id)
        out.append(
            LineageHit(
                lineage_id=lineage_id,
                score=max(m.score for m in members),  # a lineage is as good as its best revision
                best=best,
                members=tuple(sorted(members, key=lambda m: m.version)),
                first_seen=lineage.first_seen if lineage else best.version,
                last_seen=lineage.last_seen if lineage else best.version,
                inferred=bool(lineage and lineage.has_inferred_link),
                timeline=tuple(lineage.timeline()) if lineage else (),
            )
        )

    out.sort(key=lambda h: (-h.score, h.best.rank, h.lineage_id))
    return out[:top_k]


def duplicate_rate(hits: Sequence[LineageHit] | Sequence[RevisionHit], index: LineageIndex | None = None) -> float:
    """Share of a result list that is a second revision of something already in it (spec 04 §6).

    Grouped results are duplicate-free by construction, so this measures the flat baseline — and the difference
    between the two is the whole claim the Bonus makes.
    """
    if not hits:
        return 0.0
    seen: set[str] = set()
    duplicates = 0
    for hit in hits:
        if isinstance(hit, LineageHit):
            identity = hit.lineage_id
        else:
            identity = (index.of(hit.version, hit.key) if index else None) or f"{hit.version}:{hit.key}"
        if identity in seen:
            duplicates += 1
        seen.add(identity)
    return duplicates / len(hits)


def unique_bodies(revisions: Sequence[RevisionHit]) -> dict[str, list[RevisionHit]]:
    """`body_hash -> the revisions carrying it`: identical revisions are one candidate with a span."""
    out: dict[str, list[RevisionHit]] = {}
    for revision in revisions:
        out.setdefault(revision.body_hash, []).append(revision)
    return out


__all__ = ["LineageHit", "RevisionHit", "duplicate_rate", "group", "unique_bodies"]
