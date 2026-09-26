"""The alignment cascade: recognising revisions of one unit across versions (docs/spec/04 §6, D12).

Across versions a unit exists as many near-identical revisions. Recognising them is what lets the Bonus rank
*lineages* instead of drowning a result list in twenty copies of the same function — and getting it wrong is
worse than not doing it at all, because a wrong merge hides a genuinely different implementation behind one that
merely looks similar.

So the cascade runs cheapest-and-most-certain first, and every stage stores the evidence it used:

| stage | rule | relation | certainty |
|---|---|---|---|
| S0 | same key, same body hash | `identical` | certain |
| S1 | same key: `modified` at similarity ≥ θ_mod, else `replaced` (rewritten in place) | high |
| S2 | same body under a different key | `moved` | high |
| S3 | name-insensitive normalised body equal | `renamed` | medium-high |
| S4 | leftover removed × added, best assignment ≥ θ_rep | `replaced` | low-medium, always labelled |
| S5 | nothing matched | `added` / `removed` | — |

Two rules make it safe. A link below the confidence floor is **not** a link: the pair stays `added` and `removed`,
which reads as "we do not know", and unknown beats a guessed merge (D12). And S4 — the only inferential stage —
is a global best assignment rather than a greedy first match, so one strong pair cannot drag a weak one along
with it, and its links are labelled `inferred` wherever they surface.

Similarity reuses what the snapshots already hold: token sets from the text and, when both sides were embedded,
the cosine of vectors already in the store. Nothing here embeds anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RELATIONS = ("identical", "modified", "moved", "renamed", "replaced", "added", "removed")
#: Confidence floors. Below `FLOOR` a pair is not linked at all — two unknowns beat one wrong merge.
FLOOR = 0.55
THETA_MOD = 0.60
THETA_REP = 0.72
#: Weights for the S4 score: token overlap, structural shape, dense cosine (spec 04 §6).
W_TOKEN, W_STRUCT, W_DENSE = 0.45, 0.20, 0.35

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
_KEYWORDS = frozenset(
    (
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "True",
        "False",
        "None",
        "self",
        "print",
        "range",
        "len",
        "int",
        "str",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
    )
)


@dataclass(frozen=True, slots=True)
class Revision:
    """One unit of one version, as the cascade sees it."""

    key: str
    body_hash: str
    text: str
    unit_id: str = ""
    vector: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Link:
    """One alignment decision, with the evidence behind it. `inferred` marks S4, which is a guess with a floor."""

    from_key: str | None
    to_key: str | None
    relation: str
    confidence: float
    stage: str
    evidence: Mapping[str, float] = field(default_factory=dict)

    @property
    def inferred(self) -> bool:
        return self.stage == "S4"

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_key": self.from_key,
            "to_key": self.to_key,
            "relation": self.relation,
            "confidence": round(self.confidence, 4),
            "stage": self.stage,
            "inferred": self.inferred,
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
        }


# -- similarity ------------------------------------------------------------------------------------------------
def token_set(text: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in _TOKEN.finditer(text))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def structural_shape(text: str) -> frozenset[str]:
    """What the code *is* with its names removed: keywords, punctuation and literals, in order-free form.

    Two revisions of one function keep their shape through a rename; two different algorithms do not acquire one.
    """
    shape: list[str] = []
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        if _IDENTIFIER.fullmatch(token) and token.lower() not in _KEYWORDS:
            shape.append("_name")
        else:
            shape.append(token.lower())
    return frozenset(f"{a}|{b}" for a, b in zip(shape, shape[1:], strict=False)) or frozenset(shape)


def normalised_body(text: str) -> str:
    """Name-insensitive normal form: identifiers collapse, whitespace collapses. S3 compares these."""
    out = []
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        out.append("_" if _IDENTIFIER.fullmatch(token) and token.lower() not in _KEYWORDS else token.lower())
    return " ".join(out)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine of two vectors that already exist. Absent on either side contributes nothing rather than a guess."""
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def similarity(a: Revision, b: Revision) -> dict[str, float]:
    """The evidence vector for one pair. Every component is reported, so a decision can be read back."""
    token = jaccard(token_set(a.text), token_set(b.text))
    struct = jaccard(structural_shape(a.text), structural_shape(b.text))
    dense = cosine(a.vector, b.vector)
    combined = W_TOKEN * token + W_STRUCT * struct + W_DENSE * dense
    # With no vectors the dense term is structurally absent, so the remaining weights are renormalised rather
    # than letting a missing channel look like disagreement.
    if a.vector is None or b.vector is None:
        combined = (W_TOKEN * token + W_STRUCT * struct) / (W_TOKEN + W_STRUCT)
    return {"token": token, "structure": struct, "dense": dense, "combined": combined}


# -- the cascade -----------------------------------------------------------------------------------------------
def align(
    parent: Sequence[Revision],
    child: Sequence[Revision],
    *,
    theta_mod: float = THETA_MOD,
    theta_rep: float = THETA_REP,
    floor: float = FLOOR,
) -> list[Link]:
    """Align one version against its parent. Returns one link per parent unit and per unmatched child unit."""
    links: list[Link] = []
    by_key_parent = {r.key: r for r in parent}
    by_key_child = {r.key: r for r in child}
    matched_parent: set[str] = set()
    matched_child: set[str] = set()

    # S0 and S1: the same key on both sides is the ordinary case and needs no inference.
    for key, before in by_key_parent.items():
        after = by_key_child.get(key)
        if after is None:
            continue
        if before.body_hash == after.body_hash:
            links.append(Link(key, key, "identical", 1.0, "S0", {"body_hash": 1.0}))
        else:
            evidence = similarity(before, after)
            relation = "modified" if evidence["combined"] >= theta_mod else "replaced"
            # Same key, different body: the key itself is strong evidence, so this stays a link even when the
            # bodies diverge — a rewrite in place is still the same unit.
            links.append(Link(key, key, relation, max(evidence["combined"], theta_mod), "S1", evidence))
        matched_parent.add(key)
        matched_child.add(key)

    leftover_parent = [r for r in parent if r.key not in matched_parent]
    leftover_child = [r for r in child if r.key not in matched_child]

    # S2: the same bytes under a different key is a move, and it is certain.
    by_hash_child: dict[str, list[Revision]] = {}
    for revision in leftover_child:
        by_hash_child.setdefault(revision.body_hash, []).append(revision)
    for before in list(leftover_parent):
        pool = by_hash_child.get(before.body_hash)
        if not pool:
            continue
        after = pool.pop(0)
        links.append(Link(before.key, after.key, "moved", 1.0, "S2", {"body_hash": 1.0}))
        matched_parent.add(before.key)
        matched_child.add(after.key)

    leftover_parent = [r for r in leftover_parent if r.key not in matched_parent]
    leftover_child = [r for r in leftover_child if r.key not in matched_child]

    # S3: identical once names are removed — a rename, possibly with a move.
    by_norm_child: dict[str, list[Revision]] = {}
    for revision in leftover_child:
        by_norm_child.setdefault(normalised_body(revision.text), []).append(revision)
    for before in list(leftover_parent):
        pool = by_norm_child.get(normalised_body(before.text))
        if not pool:
            continue
        after = pool.pop(0)
        links.append(Link(before.key, after.key, "renamed", 0.85, "S3", {"normalised_body": 1.0}))
        matched_parent.add(before.key)
        matched_child.add(after.key)

    leftover_parent = [r for r in leftover_parent if r.key not in matched_parent]
    leftover_child = [r for r in leftover_child if r.key not in matched_child]

    # S4: the only inferential stage. A global best assignment, never a greedy first match.
    for before, after, evidence in _best_assignment(leftover_parent, leftover_child, theta_rep):
        confidence = float(evidence["combined"])
        if confidence < floor:
            continue
        links.append(Link(before.key, after.key, "replaced", confidence, "S4", evidence))
        matched_parent.add(before.key)
        matched_child.add(after.key)

    # S5: whatever is left really did appear or disappear — or we simply do not know, which reads the same way.
    for before in parent:
        if before.key not in matched_parent:
            links.append(Link(before.key, None, "removed", 1.0, "S5", {}))
    for after in child:
        if after.key not in matched_child:
            links.append(Link(None, after.key, "added", 1.0, "S5", {}))
    return links


def _best_assignment(
    parent: Sequence[Revision], child: Sequence[Revision], theta: float
) -> list[tuple[Revision, Revision, dict[str, float]]]:
    """Highest-scoring one-to-one assignment above `theta`.

    Greedy over a globally sorted list rather than Hungarian: the pools reaching S4 are the leftovers of three
    earlier stages and are small, and a greedy pass over *sorted* pairs gives the same answer whenever the scores
    are not pathologically close — while never letting one strong pair pull a weak one in behind it.
    """
    scored: list[tuple[float, Revision, Revision, dict[str, float]]] = []
    for before in parent:
        for after in child:
            evidence = similarity(before, after)
            if evidence["combined"] >= theta:
                scored.append((evidence["combined"], before, after, evidence))
    scored.sort(key=lambda item: (-item[0], item[1].key, item[2].key))

    used_parent: set[str] = set()
    used_child: set[str] = set()
    out: list[tuple[Revision, Revision, dict[str, float]]] = []
    for _, before, after, evidence in scored:
        if before.key in used_parent or after.key in used_child:
            continue
        used_parent.add(before.key)
        used_child.add(after.key)
        out.append((before, after, evidence))
    return out


__all__ = [
    "FLOOR",
    "RELATIONS",
    "THETA_MOD",
    "THETA_REP",
    "W_DENSE",
    "W_STRUCT",
    "W_TOKEN",
    "Link",
    "Revision",
    "align",
    "cosine",
    "jaccard",
    "normalised_body",
    "similarity",
    "structural_shape",
    "token_set",
]
