"""The Bonus: alignment, lineages and evolution-aware grouping (docs/spec/04 §6, D12).

Two failures matter here and they pull in opposite directions. **Missing** a lineage floods a result list with
near-identical revisions and buries the genuinely different implementations. **Inventing** one hides a different
implementation behind something that merely looked similar — and that is invisible to whoever reads the output.
So the tests check both: that ordinary edits stay one lineage through renames and moves, and that two unrelated
units are never merged, whatever the thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from acis.eval import appsevolve as ae
from acis.lineage import align, search, store

SORT = "def bubble_sort(xs):\n    for i in range(len(xs)):\n        xs.sort()\n    return xs\n"
SORT_EDITED = "def bubble_sort(xs):\n    # tidy\n    for i in range(len(xs)):\n        xs.sort()\n    return xs\n"
SORT_RENAMED = "def sort_list(ys):\n    for j in range(len(ys)):\n        ys.sort()\n    return ys\n"
GRAPH = (
    "import heapq\ndef dijkstra(g, s):\n    pq = [(0, s)]\n    while pq:\n        heapq.heappop(pq)\n    return {}\n"
)


def rev(key, text, vector=None):
    from acis.core.hashing import sha256_text

    return align.Revision(key=key, body_hash=sha256_text(text), text=text, vector=vector)


# -- the cascade, stage by stage ---------------------------------------------------------------------------------
def test_identical_content_under_the_same_key_is_certain():
    links = align.align([rev("a.py", SORT)], [rev("a.py", SORT)])
    assert [(link.relation, link.stage, link.confidence) for link in links] == [("identical", "S0", 1.0)]


def test_an_edit_in_place_stays_the_same_unit():
    links = align.align([rev("a.py", SORT)], [rev("a.py", SORT_EDITED)])
    assert links[0].relation == "modified" and links[0].stage == "S1"
    assert not links[0].inferred


def test_the_same_bytes_under_a_new_name_is_a_move_and_it_is_certain():
    links = align.align([rev("old.py", SORT)], [rev("new.py", SORT)])
    moved = [link for link in links if link.relation == "moved"]
    assert moved and moved[0].from_key == "old.py" and moved[0].to_key == "new.py"
    assert moved[0].confidence == 1.0


def test_a_rename_of_identifiers_is_recognised_without_the_bytes_matching():
    links = align.align([rev("a.py", SORT)], [rev("b.py", SORT_RENAMED)])
    renamed = [link for link in links if link.relation == "renamed"]
    assert renamed and renamed[0].stage == "S3"


def test_two_unrelated_units_are_never_merged():
    """The failure that matters: a wrong merge hides a different implementation and nobody can see it happened."""
    links = align.align([rev("sort.py", SORT)], [rev("graph.py", GRAPH)])
    assert {link.relation for link in links} == {"removed", "added"}
    assert not any(link.relation in ("modified", "moved", "renamed", "replaced") for link in links)


def test_a_replacement_is_inferred_only_above_the_threshold_and_is_labelled():
    similar = SORT.replace("bubble_sort", "quick_sort").replace("xs.sort()", "xs.sort(reverse=False)")
    links = align.align([rev("a.py", SORT)], [rev("z.py", similar)])
    replaced = [link for link in links if link.relation == "replaced"]
    if replaced:  # whether it fires depends on the thresholds; if it does, it must be honest about being a guess
        assert replaced[0].stage == "S4" and replaced[0].inferred
        assert replaced[0].confidence >= align.FLOOR


def test_an_added_and_a_removed_unit_are_reported_as_such():
    links = align.align([rev("gone.py", SORT)], [rev("new.py", GRAPH)])
    assert {link.relation for link in links} == {"removed", "added"}


def test_dense_similarity_is_used_when_vectors_exist_and_ignored_when_they_do_not():
    a, b = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    with_vectors = align.similarity(rev("a", SORT, a), rev("b", SORT_EDITED, b))
    without = align.similarity(rev("a", SORT), rev("b", SORT_EDITED))
    assert with_vectors["dense"] == pytest.approx(1.0)
    assert without["dense"] == 0.0
    assert without["combined"] > 0.0  # the missing channel is renormalised away, not counted as disagreement


# -- lineages -------------------------------------------------------------------------------------------------
def build(versions):
    """Align consecutive versions and close the links into lineages."""
    revs = {label: [rev(k, t) for k, t in units.items()] for label, units in versions}
    links = {}
    for i, (label, _) in enumerate(versions):
        if i:
            links[label] = align.align(revs[versions[i - 1][0]], revs[label])
    tree = [(label, {k: rev(k, t).body_hash for k, t in units.items()}) for label, units in versions]
    return store.build_lineages(tree, links)


def test_a_unit_edited_across_three_versions_is_one_lineage():
    lineages = build(
        [("v1", {"a.py": SORT}), ("v2", {"a.py": SORT_EDITED}), ("v3", {"a.py": SORT_EDITED + "# more\n"})]
    )
    assert len(lineages) == 1
    assert [m.version for m in lineages[0].members] == ["v1", "v2", "v3"]
    assert lineages[0].first_seen == "v1" and lineages[0].last_seen == "v3"


def test_a_lineage_survives_a_rename_midway():
    lineages = build([("v1", {"a.py": SORT}), ("v2", {"b.py": SORT}), ("v3", {"b.py": SORT_EDITED})])
    assert len(lineages) == 1
    assert [m.key for m in lineages[0].members] == ["a.py", "b.py", "b.py"]


def test_distinct_implementations_stay_distinct():
    lineages = build(
        [("v1", {"sort.py": SORT, "graph.py": GRAPH}), ("v2", {"sort.py": SORT_EDITED, "graph.py": GRAPH})]
    )
    assert len(lineages) == 2
    assert {len(lin.members) for lin in lineages} == {2}


def test_a_lineage_that_ends_does_not_reappear():
    lineages = build([("v1", {"a.py": SORT}), ("v2", {"b.py": GRAPH})])
    assert len(lineages) == 2
    assert [lin.last_seen for lin in lineages] == ["v1", "v2"]


def test_a_timeline_marks_the_versions_where_content_changed():
    lineages = build([("v1", {"a.py": SORT}), ("v2", {"a.py": SORT}), ("v3", {"a.py": SORT_EDITED})])
    timeline = lineages[0].timeline()
    assert [step["changed"] for step in timeline] == [True, False, True]


def test_a_link_below_the_floor_never_joins_a_lineage():
    """D12: two unknowns beat one wrong merge."""
    links = {"v2": [align.Link("a.py", "b.py", "replaced", 0.10, "S4", {"combined": 0.10})]}
    lineages = store.build_lineages([("v1", {"a.py": "h1"}), ("v2", {"b.py": "h2"})], links, floor=align.FLOOR)
    assert len(lineages) == 2


def test_lineage_ids_are_stable_as_history_grows():
    short_history = build([("v1", {"a.py": SORT}), ("v2", {"a.py": SORT_EDITED})])
    long_history = build([("v1", {"a.py": SORT}), ("v2", {"a.py": SORT_EDITED}), ("v3", {"a.py": SORT_EDITED})])
    assert short_history[0].lineage_id == long_history[0].lineage_id


# -- grouping -------------------------------------------------------------------------------------------------
def hits(*triples):
    return [search.RevisionHit(v, k, f"h{k}{v}", s, i + 1) for i, (v, k, s) in enumerate(triples)]


def test_grouping_collapses_revisions_of_one_lineage_into_one_answer():
    lineages = build([("v1", {"a.py": SORT}), ("v2", {"a.py": SORT_EDITED}), ("v3", {"a.py": SORT_EDITED})])
    index = store.LineageIndex.build(lineages)
    grouped = search.group(hits(("v1", "a.py", 0.9), ("v2", "a.py", 0.8), ("v3", "a.py", 0.7)), index)
    assert len(grouped) == 1
    assert grouped[0].n_revisions == 3 and grouped[0].best.version == "v1"


def test_a_lineage_scores_as_its_best_revision_not_its_average():
    """A unit improved in v5 must be findable by a query only the v5 revision answers."""
    lineages = build([("v1", {"a.py": SORT}), ("v2", {"a.py": SORT_EDITED})])
    index = store.LineageIndex.build(lineages)
    grouped = search.group(hits(("v1", "a.py", 0.10), ("v2", "a.py", 0.95)), index)
    assert grouped[0].score == pytest.approx(0.95)


def test_preferring_the_latest_revision_breaks_a_tie_towards_the_newest():
    lineages = build([("v1", {"a.py": SORT}), ("v2", {"a.py": SORT_EDITED})])
    index = store.LineageIndex.build(lineages)
    tied = hits(("v1", "a.py", 0.9), ("v2", "a.py", 0.9))
    assert search.group(tied, index, prefer="best")[0].best.version == "v1"
    assert search.group(tied, index, prefer="latest")[0].best.version == "v2"


def test_grouping_removes_the_duplication_a_flat_list_has():
    lineages = build([("v1", {"a.py": SORT, "g.py": GRAPH}), ("v2", {"a.py": SORT_EDITED, "g.py": GRAPH})])
    index = store.LineageIndex.build(lineages)
    flat = hits(("v1", "a.py", 0.9), ("v2", "a.py", 0.88), ("v1", "g.py", 0.5), ("v2", "g.py", 0.49))
    assert search.duplicate_rate(flat, index) == 0.5
    assert search.duplicate_rate(search.group(flat, index)) == 0.0


def test_an_inferred_link_is_reported_in_the_answer():
    links = {"v2": [align.Link("a.py", "b.py", "replaced", 0.80, "S4", {"combined": 0.80})]}
    lineages = store.build_lineages([("v1", {"a.py": "h1"}), ("v2", {"b.py": "h2"})], links)
    index = store.LineageIndex.build(lineages)
    grouped = search.group(hits(("v2", "b.py", 0.9)), index)
    assert grouped[0].inferred is True


def test_identical_revisions_are_one_candidate_with_a_span():
    revisions = hits(("v1", "a.py", 0.9), ("v2", "a.py", 0.9))
    same = [search.RevisionHit(r.version, r.key, "same-body", r.score, r.rank) for r in revisions]
    assert list(search.unique_bodies(same)) == ["same-body"]


# -- against generated ground truth --------------------------------------------------------------------------
def test_the_cascade_recovers_lineages_the_generator_created():
    """Apps-Evolve knows the answer by construction, so this measures rather than asserts a hope."""
    seeds = [("sort.py", SORT), ("graph.py", GRAPH)]
    evolution = ae.generate(seeds, n_versions=4, seed=3, edits_per_version=1, operators=list(ae.EDIT_OPERATORS))
    lineages = build([(label, dict(units)) for label, units in evolution.versions])

    # Every seed is one lineage running the whole history: only in-place edits were applied.
    assert len(lineages) == len(seeds)
    assert all(len(lin.members) == len(evolution.labels) for lin in lineages)
