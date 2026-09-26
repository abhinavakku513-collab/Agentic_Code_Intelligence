"""Apps-Evolve, the generated version history (docs/spec/04 §5).

The generator exists so P1 and the Bonus can be measured against ground truth rather than against opinion, which
only works if the ground truth is actually true. So these tests check the generator the way one checks a ruler:
the same seed gives the same history, every recorded relation matches what the text did, an operator that claims
to be semantics-preserving preserves them, and a lineage is traceable from the first version to the last.
"""

from __future__ import annotations

import random

import pytest

from acis.core.errors import InvalidInput
from acis.eval import appsevolve as ae

SEEDS = [
    ("sort.py", "def bubble(xs):\n    n = 5\n    return sorted(xs)\n"),
    ("search.py", "def binary(xs, target):\n    lo = 0\n    return lo\n\n\ndef helper(x):\n    return x + 1\n"),
    ("graph.py", "def dijkstra(graph, start):\n    dist = {}\n    return dist\n"),
]


# -- determinism ---------------------------------------------------------------------------------------------
def test_the_same_seed_gives_the_same_history():
    """A benchmark number nobody can reproduce is not evidence."""
    a = ae.generate(SEEDS, seed=7)
    b = ae.generate(SEEDS, seed=7)
    assert a.versions == b.versions
    assert a.changes == b.changes


def test_a_different_seed_gives_a_different_history():
    assert ae.generate(SEEDS, seed=1).versions != ae.generate(SEEDS, seed=2).versions


def test_the_requested_number_of_versions_is_what_comes_out():
    assert ae.generate(SEEDS, n_versions=4, seed=0).labels == ("v1", "v2", "v3", "v4")


def test_the_first_version_is_the_seeds_unchanged():
    evolution = ae.generate(SEEDS, seed=3)
    assert dict(evolution.units("v1")) == dict(SEEDS)


# -- ground truth --------------------------------------------------------------------------------------------
def test_every_unit_of_every_version_belongs_to_a_lineage():
    evolution = ae.generate(SEEDS, n_versions=5, seed=5, edits_per_version=3)
    for label in evolution.labels:
        for key in evolution.units(label):
            assert (label, key) in evolution.lineage_of


def test_a_lineage_keeps_its_identity_through_a_rename():
    """The point of `rekey`: the key changed, the lineage did not — and it may be renamed again later."""
    evolution = ae.generate(SEEDS, n_versions=3, seed=0, edits_per_version=1, operators=["rekey"])
    moves = [(label, c) for label, step in evolution.changes.items() for c in step if c.relation == "moved"]
    assert moves
    for label, move in moves:
        assert move.from_key != move.to_key
        assert evolution.lineage_of[(label, move.to_key)] == move.lineage_id
        assert evolution.key_at(move.lineage_id, label) == move.to_key


def test_a_deleted_unit_is_gone_from_every_later_version():
    evolution = ae.generate(SEEDS, n_versions=4, seed=11, edits_per_version=1, operators=["delete"])
    removals = [(label, c) for label, step in evolution.changes.items() for c in step if c.relation == "removed"]
    assert removals
    label, change = removals[0]
    later = evolution.labels[evolution.labels.index(label) :]
    assert all(change.from_key not in evolution.units(name) for name in later)


def test_an_added_unit_is_present_from_the_version_that_added_it():
    evolution = ae.generate(SEEDS, n_versions=4, seed=13, edits_per_version=1, operators=["add"])
    additions = [(label, c) for label, step in evolution.changes.items() for c in step if c.relation == "added"]
    assert additions
    label, change = additions[0]
    assert change.to_key in evolution.units(label)
    assert change.to_key not in evolution.units("v1")


def test_a_recorded_modification_really_changed_the_text():
    evolution = ae.generate(SEEDS, n_versions=5, seed=17, edits_per_version=2, operators=list(ae.EDIT_OPERATORS))
    for index, label in enumerate(evolution.labels[1:], start=1):
        previous = evolution.units(evolution.labels[index - 1])
        current = evolution.units(label)
        for change in evolution.changes[label]:
            if change.relation == "modified" and change.from_key in previous:
                assert current[change.to_key] != previous[change.from_key]


def test_nothing_changes_that_no_operator_touched():
    """Every difference between consecutive versions is accounted for by a recorded change."""
    evolution = ae.generate(SEEDS, n_versions=5, seed=23, edits_per_version=2)
    for index, label in enumerate(evolution.labels[1:], start=1):
        before = evolution.units(evolution.labels[index - 1])
        after = evolution.units(label)
        touched = {c.from_key for c in evolution.changes[label]} | {c.to_key for c in evolution.changes[label]}
        for key in set(before) | set(after):
            if before.get(key) != after.get(key):
                assert key in touched, f"{key} changed in {label} with no recorded operator"


# -- the operators themselves -----------------------------------------------------------------------------------
def test_reformat_changes_only_whitespace():
    text = SEEDS[1][1]
    out = ae.reformat(text, random.Random(0))
    assert out.split() == text.split()


def test_comment_adds_a_comment_line():
    out = ae.comment("def f():\n    return 1\n", random.Random(0))
    assert any(line.strip().startswith("#") for line in out.split("\n"))


def test_rename_is_consistent_and_token_aligned():
    """A partial word must never be rewritten: renaming `lo` must not touch `lower`."""
    text = "def f(lo, lower):\n    return lo + lower\n"
    out = ae.rename(text, random.Random(2))
    assert "lower" in out or "lower_v" in out
    assert out.count("def f(") == 1


def test_reorder_keeps_every_definition():
    text = SEEDS[1][1]
    out = ae.reorder(text, random.Random(0))
    assert sorted(out.split()) == sorted(text.split())
    assert out != text


def test_constant_changes_a_number():
    out = ae.constant("def f():\n    return 5\n", random.Random(0))
    assert "return 5" not in out and "return" in out


def test_branch_adds_a_guard_inside_the_function():
    out = ae.branch("def f():\n    return 1\n", random.Random(0))
    assert out.startswith("def f():") and len(out) > len("def f():\n    return 1\n")


def test_replace_swaps_in_a_spare_body_and_says_so():
    spare = "def completely_different():\n    return 99\n"
    evolution = ae.generate(SEEDS, n_versions=2, seed=0, edits_per_version=1, operators=["replace"], spares=[spare])
    replaced = [c for step in evolution.changes.values() for c in step if c.relation == "replaced"]
    assert replaced and spare in evolution.units("v2").values()


def test_replace_without_a_spare_falls_back_rather_than_inventing_content():
    evolution = ae.generate(SEEDS, n_versions=2, seed=0, edits_per_version=1, operators=["replace"])
    assert all(c.relation != "replaced" for step in evolution.changes.values() for c in step)


# -- input handling -------------------------------------------------------------------------------------------
def test_an_unknown_operator_is_refused():
    with pytest.raises(InvalidInput, match="unknown operator"):
        ae.generate(SEEDS, operators=["teleport"])


def test_no_seeds_is_refused():
    with pytest.raises(InvalidInput, match="at least one"):
        ae.generate([])


def test_an_evolution_can_be_handed_straight_to_a_memory_source():
    evolution = ae.generate(SEEDS, n_versions=3, seed=0)
    options = evolution.as_source_options()
    assert set(options["versions"]) == set(evolution.labels)
    assert options["versions"]["v1"] == dict(SEEDS)


def test_rename_never_breaks_an_import_or_an_attribute():
    """A refactor renames *our* names. Renaming `heapq` or `.sort` produces code that no longer runs, and it is the
    first thing a reader of the demo sees (`import heapq_v85`)."""
    text = (
        "import heapq\nfrom collections import deque\nimport numpy as np\n"
        "def solve(graph, start):\n    queue = deque([start])\n    graph.sort()\n"
        "    return heapq.heappop(np.array(queue))\n"
    )
    protected = {"heapq", "collections", "deque", "numpy", "sort", "heappop", "array"}
    for seed in range(200):
        out = ae.rename(text, random.Random(seed))
        for name in protected:
            assert f"{name}_v" not in out, (seed, name)
        assert out.count("import heapq\n") == 1
