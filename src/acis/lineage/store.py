"""Lineages: turning pairwise alignments into the identity of a unit over time (docs/spec/04 §6, D12).

An alignment says "this revision came from that one". A **lineage** is what you get when those links are closed
transitively: every revision of one unit across the whole history, under one id. That is what the Bonus ranks.

Union-find does the closing, with one rule that matters: only links **above the confidence floor** join anything.
A pair the cascade could not place stays two lineages, and the result list shows two entries — which is the
honest answer. Merging them would hide a genuinely different implementation behind one that merely looked
similar, and that failure is invisible to anyone reading the output (D12: unknown beats a guessed merge).

The canonical id is derived from the earliest member, so a lineage keeps its name as history grows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from acis.core.hashing import hash_obj, short
from acis.lineage.align import FLOOR, Link


@dataclass(frozen=True, slots=True)
class Member:
    """One revision inside a lineage: which version it belongs to, and how it relates to its predecessor."""

    version: str
    key: str
    body_hash: str
    relation: str = "added"
    confidence: float = 1.0
    inferred: bool = False


@dataclass(frozen=True, slots=True)
class Lineage:
    """One unit's history. `members` runs oldest-first, which is what a timeline reads."""

    lineage_id: str
    members: tuple[Member, ...]

    @property
    def first_seen(self) -> str:
        return self.members[0].version

    @property
    def last_seen(self) -> str:
        return self.members[-1].version

    @property
    def head(self) -> Member:
        return self.members[-1]

    @property
    def has_inferred_link(self) -> bool:
        """True when any link in this lineage came from S4 — the stage that guesses, with a floor."""
        return any(m.inferred for m in self.members)

    def at(self, version: str) -> Member | None:
        return next((m for m in self.members if m.version == version), None)

    def timeline(self) -> list[dict[str, Any]]:
        """Change points, oldest first: what happened to this unit and how confident we are that it happened."""
        out: list[dict[str, Any]] = []
        first_seen: dict[str, str] = {}
        for index, member in enumerate(self.members):
            previous = self.members[index - 1] if index else None
            changed = previous is None or previous.body_hash != member.body_hash
            step: dict[str, Any] = {
                "version": member.version,
                "key": member.key,
                "relation": member.relation,
                "confidence": round(member.confidence, 4),
                "inferred": member.inferred,
                "changed": changed,
                "body_hash": member.body_hash,
            }
            # A change back to content this unit already had — a revert — is a fact of the content, not a guess:
            # the bytes are identical to an earlier revision's, so say which one.
            if changed and previous is not None and member.body_hash in first_seen:
                step["same_content_as"] = first_seen[member.body_hash]
            first_seen.setdefault(member.body_hash, member.version)
            out.append(step)
        return out


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, item: tuple[str, str]) -> tuple[str, str]:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def unite(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_lineages(
    versions: Sequence[tuple[str, Mapping[str, str]]],
    links_by_version: Mapping[str, Sequence[Link]],
    *,
    floor: float = FLOOR,
) -> list[Lineage]:
    """Close the alignments transitively into lineages.

    `versions` is `(label, {key: body_hash})` oldest-first; `links_by_version[label]` aligns that version against
    its predecessor. A link below `floor` is ignored entirely — not weakened, ignored — so the two revisions stay
    separate lineages and the caller sees two answers instead of one confident wrong one.
    """
    union = _UnionFind()
    labels = [label for label, _ in versions]
    details: dict[tuple[str, str], tuple[str, float, bool]] = {}

    for index, label in enumerate(labels):
        if index == 0:
            continue
        previous = labels[index - 1]
        for link in links_by_version.get(label, ()):
            if link.from_key is None or link.to_key is None:
                continue
            if link.confidence < floor:
                continue  # unknown beats a guessed merge (D12)
            union.unite((previous, link.from_key), (label, link.to_key))
            details[(label, link.to_key)] = (link.relation, link.confidence, link.inferred)

    grouped: dict[tuple[str, str], list[Member]] = {}
    order = {label: i for i, label in enumerate(labels)}
    for label, units in versions:
        for key, body_hash in units.items():
            relation, confidence, inferred = details.get((label, key), ("added", 1.0, False))
            root = union.find((label, key))
            grouped.setdefault(root, []).append(
                Member(
                    version=label,
                    key=key,
                    body_hash=body_hash,
                    relation=relation,
                    confidence=confidence,
                    inferred=inferred,
                )
            )

    lineages: list[Lineage] = []
    for root, members in grouped.items():
        members.sort(key=lambda m: (order.get(m.version, 0), m.key))
        lineages.append(Lineage(lineage_id=lineage_id_for(root), members=tuple(members)))
    lineages.sort(key=lambda lin: (order.get(lin.first_seen, 0), lin.members[0].key))
    return lineages


def lineage_id_for(root: tuple[str, str]) -> str:
    """Derived from the earliest member, so a lineage's name does not change as history grows."""
    return "lin_" + short(hash_obj({"version": root[0], "key": root[1]}), 12)


@dataclass(frozen=True, slots=True)
class LineageIndex:
    """Lookup from `(version, key)` to the lineage that contains it."""

    lineages: tuple[Lineage, ...]
    by_member: Mapping[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def build(cls, lineages: Sequence[Lineage]) -> LineageIndex:
        mapping = {(m.version, m.key): lin.lineage_id for lin in lineages for m in lin.members}
        return cls(lineages=tuple(lineages), by_member=mapping)

    def of(self, version: str, key: str) -> str | None:
        return self.by_member.get((version, key))

    def get(self, lineage_id: str) -> Lineage | None:
        return next((lin for lin in self.lineages if lin.lineage_id == lineage_id), None)

    @property
    def size(self) -> int:
        return len(self.lineages)


__all__ = ["Lineage", "LineageIndex", "Member", "build_lineages", "lineage_id_for"]
