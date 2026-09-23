"""Apps-Evolve: a version history with ground truth by construction (docs/spec/04 §5, §6).

There is no public benchmark for "did retrieval follow this code as it changed", so P1 and the Bonus need one we
can trust. Buying trust by labelling a real history is expensive and arguable; **generating** one is neither: if
v3's `sort.py` was produced by renaming an identifier in v2's `sort.py`, that is not an opinion, and the lineage
is known before any retrieval runs.

The operators are the edits that actually happen to code, chosen so that each one attacks a different assumption:

| operator | what it changes | what it is designed to break |
|---|---|---|
| `reformat` | whitespace and blank lines | anything that hashes formatting |
| `comment` | a comment line | exact-match and near-duplicate detectors |
| `rename` | one identifier, consistently | lexical overlap; dense should barely move |
| `reorder` | the order of two helpers | order-sensitive similarity |
| `constant` | a numeric literal | "semantically identical" assumptions |
| `branch` | adds a guard clause | structure comparison |
| `replace` | a different solution to the same problem | the assumption that a lineage is textually continuous |
| `rekey` | the unit's key (a rename or move) | key-based identity |
| `add` / `delete` | a unit appears or disappears | matching that assumes a bijection |

Everything is deterministic given a seed: the same inputs produce the same history, so a benchmark number is
reproducible and a failure can be replayed exactly. Nothing here is APPS-specific — the input is any collection of
`(key, text)` pairs (INV-15), and `acis.appsdata` supplies real ones when they are wanted.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from acis.core.errors import InvalidInput

#: Relations a transition can carry. `identical` and `modified` keep the key; `moved` changes it; `replaced` is
#: a different body for the same lineage; `added` and `removed` have no counterpart on one side.
RELATIONS = ("identical", "modified", "moved", "replaced", "added", "removed")
OPERATORS = ("reformat", "comment", "rename", "reorder", "constant", "branch", "replace", "rekey", "add", "delete")
#: Operators that keep the key and change the body — the ordinary case, and the one lineage must get right.
EDIT_OPERATORS = ("reformat", "comment", "rename", "reorder", "constant", "branch")

_IDENTIFIER = re.compile(r"\b([a-z_][a-z0-9_]{2,})\b")
_NUMBER = re.compile(r"(?<![\w.])(\d{1,6})(?![\w.])")
_DEF = re.compile(r"^def\s+\w+\s*\(.*?\):$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Change:
    """One unit's fate between two versions. Ground truth: recorded when the edit is made, not inferred."""

    lineage_id: str
    relation: str
    operator: str
    from_key: str | None
    to_key: str | None


@dataclass(frozen=True, slots=True)
class Evolution:
    """A generated history and everything that is true about it by construction."""

    versions: tuple[tuple[str, Mapping[str, str]], ...]
    changes: Mapping[str, tuple[Change, ...]]
    lineage_of: Mapping[tuple[str, str], str]
    seed: int

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.versions)

    def units(self, label: str) -> Mapping[str, str]:
        for name, units in self.versions:
            if name == label:
                return units
        raise InvalidInput(f"no version {label!r} in this evolution", labels=list(self.labels))

    def as_source_options(self) -> dict[str, Any]:
        """Shape a `SourceSpec(kind="memory", …)` takes, so an evolution can be ingested directly."""
        return {"versions": {label: dict(units) for label, units in self.versions}}

    def key_at(self, lineage_id: str, label: str) -> str | None:
        """Which key this lineage lives under in `label` — `None` when it is not present in that version."""
        for (version, key), lid in self.lineage_of.items():
            if version == label and lid == lineage_id:
                return key
        return None


# -- the operators -------------------------------------------------------------------------------------------
def reformat(text: str, rng: random.Random) -> str:
    """Whitespace only: blank lines collapse or double, and indentation widens. Semantically identical."""
    lines = text.split("\n")
    widened = [("    " + line if line.startswith("    ") and rng.random() < 0.5 else line) for line in lines]
    return "\n".join(widened).replace("\n\n", "\n\n\n" if rng.random() < 0.5 else "\n")


def comment(text: str, rng: random.Random) -> str:
    # Deliberately free of the two markers `src/` is checked for: a generator that planted them in its own
    # comments would make that guard cry wolf on its own output.
    note = rng.choice(["# tidy up", "# revisit later", "# fast path", "# see issue 42"])
    lines = text.split("\n")
    at = rng.randrange(0, max(1, len(lines)))
    return "\n".join([*lines[:at], note, *lines[at:]])


def rename(text: str, rng: random.Random) -> str:
    """Rename one identifier consistently. Token-level, so a partial word is never rewritten."""
    names = sorted({m.group(1) for m in _IDENTIFIER.finditer(text)} - _KEYWORDS)
    if not names:
        return text
    old = rng.choice(names)
    new = f"{old}_v{rng.randrange(2, 99)}"
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def reorder(text: str, rng: random.Random) -> str:
    """Swap two top-level definitions. The file computes the same thing in a different order."""
    blocks = _split_definitions(text)
    if len(blocks) < 2:
        return text
    i = rng.randrange(0, len(blocks) - 1)
    blocks[i], blocks[i + 1] = blocks[i + 1], blocks[i]
    return "".join(blocks)


def constant(text: str, rng: random.Random) -> str:
    """Change a numeric literal — the smallest edit that changes what the code *does*."""
    numbers = list(_NUMBER.finditer(text))
    if not numbers:
        return text
    chosen = rng.choice(numbers)
    replacement = str(int(chosen.group(1)) + rng.choice([1, 2, 10]))
    return text[: chosen.start()] + replacement + text[chosen.end() :]


def branch(text: str, rng: random.Random) -> str:
    """Add a guard clause at the top of the first function body."""
    guard = rng.choice(
        ["    if not locals():\n        return None\n", "    if False:\n        pass\n", "    # guard\n"]
    )
    match = _DEF.search(text)
    if not match:
        return guard + text
    cut = match.end() + 1
    return text[:cut] + guard + text[cut:]


_KEYWORDS = {
    "and",
    "for",
    "not",
    "def",
    "return",
    "import",
    "from",
    "class",
    "while",
    "elif",
    "else",
    "with",
    "try",
    "except",
    "finally",
    "lambda",
    "print",
    "range",
    "input",
    "int",
    "str",
    "len",
    "list",
    "dict",
    "set",
    "sorted",
    "sum",
    "min",
    "max",
    "abs",
    "map",
    "filter",
    "self",
    "none",
    "true",
    "false",
}


def _split_definitions(text: str) -> list[str]:
    """Split into top-level blocks at `def`/`class` boundaries, keeping every character."""
    cuts = [m.start() for m in re.finditer(r"^(?:def|class)\s", text, re.MULTILINE)]
    if len(cuts) < 2:
        return [text]
    # Boundaries, not offsets: the text before the first definition stays attached to it, and the last block runs
    # to the end. `zip(bounds, bounds[1:])` over a single list would drop the final block.
    bounds = [0, *cuts[1:]]
    edges = [*bounds[1:], len(text)]
    return [text[a:b] for a, b in zip(bounds, edges, strict=True) if text[a:b]]


_EDITS = {
    "reformat": reformat,
    "comment": comment,
    "rename": rename,
    "reorder": reorder,
    "constant": constant,
    "branch": branch,
}


# -- generation -------------------------------------------------------------------------------------------------
def generate(
    seeds: Sequence[tuple[str, str]],
    *,
    n_versions: int = 5,
    seed: int = 0,
    edits_per_version: int = 2,
    operators: Sequence[str] = OPERATORS,
    spares: Sequence[str] = (),
) -> Evolution:
    """Build a history from `seeds`, applying `edits_per_version` operators at each step.

    `spares` are alternative bodies used by `replace` and `add`; without them those operators fall back to an
    ordinary edit rather than inventing content, because a generator that invents bodies is a generator whose
    ground truth nobody can check.
    """
    if not seeds:
        raise InvalidInput("Apps-Evolve needs at least one seed unit")
    unknown = sorted(set(operators) - set(OPERATORS))
    if unknown:
        raise InvalidInput("unknown operator(s)", unknown=unknown, known=list(OPERATORS))
    if n_versions < 1:
        raise InvalidInput("a history needs at least one version", n_versions=n_versions)

    rng = random.Random(seed)
    lineage_of: dict[tuple[str, str], str] = {}
    current: dict[str, str] = {}
    lineage_by_key: dict[str, str] = {}

    for index, (key, text) in enumerate(seeds):
        lineage_id = f"L{index:04d}"
        current[key] = text
        lineage_by_key[key] = lineage_id
        lineage_of[("v1", key)] = lineage_id

    versions: list[tuple[str, Mapping[str, str]]] = [("v1", dict(current))]
    changes: dict[str, tuple[Change, ...]] = {"v1": ()}
    spare_pool = list(spares)
    added = 0

    for version in range(2, n_versions + 1):
        label = f"v{version}"
        step: list[Change] = []
        for _ in range(edits_per_version):
            operator = rng.choice(list(operators))
            change = _apply(operator, current, lineage_by_key, rng, spare_pool, added)
            if change is None:
                continue
            if change.relation == "added":
                added += 1
            step.append(change)
        for key in current:
            lineage_of[(label, key)] = lineage_by_key[key]
        versions.append((label, dict(current)))
        changes[label] = tuple(step)

    return Evolution(versions=tuple(versions), changes=changes, lineage_of=lineage_of, seed=seed)


def _apply(
    operator: str,
    current: dict[str, str],
    lineage_by_key: dict[str, str],
    rng: random.Random,
    spares: list[str],
    added: int,
) -> Change | None:
    """Apply one operator in place and return the ground-truth record of what it did."""
    keys = sorted(current)
    if not keys:
        return None

    if operator == "delete" and len(keys) > 1:
        key = rng.choice(keys)
        lineage_id = lineage_by_key.pop(key)
        del current[key]
        return Change(lineage_id, "removed", operator, key, None)

    if operator == "add":
        key = f"added{added}.py"
        body = spares.pop(0) if spares else f"def added_{added}():\n    return {added}\n"
        current[key] = body
        lineage_id = f"A{added:04d}"
        lineage_by_key[key] = lineage_id
        return Change(lineage_id, "added", operator, None, key)

    if operator == "rekey":
        key = rng.choice(keys)
        new_key = f"moved_{key}"
        if new_key in current:
            return None
        current[new_key] = current.pop(key)
        lineage_by_key[new_key] = lineage_by_key.pop(key)
        return Change(lineage_by_key[new_key], "moved", operator, key, new_key)

    if operator == "replace":
        key = rng.choice(keys)
        if not spares:
            return _apply(rng.choice(list(EDIT_OPERATORS)), current, lineage_by_key, rng, spares, added)
        current[key] = spares.pop(0)
        return Change(lineage_by_key[key], "replaced", operator, key, key)

    if operator not in _EDITS:
        # `delete` on a one-unit version, or any operator that could not apply this time. A step that changes
        # nothing is a legitimate outcome — a history where every edit lands is not one code ever has — and it is
        # recorded as no change rather than forced into some other operator.
        return None

    key = rng.choice(keys)
    before = current[key]
    after = _EDITS[operator](before, rng)
    current[key] = after
    relation = "identical" if after == before else "modified"
    return Change(lineage_by_key[key], relation, operator, key, key)


__all__ = [
    "EDIT_OPERATORS",
    "OPERATORS",
    "RELATIONS",
    "Change",
    "Evolution",
    "branch",
    "comment",
    "constant",
    "generate",
    "reformat",
    "rename",
    "reorder",
]
