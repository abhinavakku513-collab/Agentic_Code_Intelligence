"""Version selectors (docs/spec/04 §4, INV-2).

A selector is how a request names *which* version it is asking about, and the rule it lives by is short: resolve
it once, or fail. There is no guessing. A label that does not exist is `NotFound`; one that exists but was never
built is `IndexRequired`; an `as_of` instant that two versions could answer is `VersionConflict` with both
candidates named. Every one of those is a different thing for the caller to do, which is why they are different
errors rather than one.
"""

from __future__ import annotations

import time

import pytest

from acis.core.errors import IndexRequired, InvalidInput, NotFound, VersionConflict
from acis.store import catalog, selectors


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))


@pytest.fixture
def repo():
    """Three built versions and one that was recorded but never built."""
    base = time.time() - 3600
    with catalog.open_catalog() as db:
        catalog.register_repo(db, "apps", source_kind="jsonl")
        for i, label in enumerate(("v1", "v2", "v3")):
            catalog.record_version(db, "apps", label, snapshot_id=f"s_{label}", state="ACTIVE")
            db.execute(
                "UPDATE versions SET created_ts = ? WHERE repo_id = ? AND label = ?", (base + i * 60, "apps", label)
            )
        catalog.record_version(db, "apps", "v4", snapshot_id=None, state="BUILDING")
        db.execute("UPDATE versions SET created_ts = ? WHERE repo_id = ? AND label = ?", (base + 180, "apps", "v4"))
    return "apps"


def resolve(repo_id, text):
    return selectors.resolve(repo_id, text)


# -- parsing --------------------------------------------------------------------------------------------------
def test_a_bare_label_is_the_same_as_naming_the_version(repo):
    assert resolve(repo, "v2").snapshot_ids == ("s_v2",)
    assert resolve(repo, "version:v2").snapshot_ids == ("s_v2",)


def test_latest_is_the_newest_built_version(repo):
    resolution = resolve(repo, "latest")
    assert resolution.snapshot_ids == ("s_v3",) and resolution.labels == ("v3",)


def test_all_returns_every_built_version_oldest_first(repo):
    resolution = resolve(repo, "all")
    assert resolution.labels == ("v1", "v2", "v3")  # v4 was never built, so it is not an answer


def test_a_range_is_inclusive_at_both_ends(repo):
    assert resolve(repo, "range:v1..v2").labels == ("v1", "v2")
    assert resolve(repo, "range:v2..v3").labels == ("v2", "v3")


def test_a_reversed_range_is_an_error_not_a_silent_swap(repo):
    with pytest.raises(InvalidInput, match="order"):
        resolve(repo, "range:v3..v1")


def test_a_snapshot_can_be_named_directly(repo):
    assert resolve(repo, "snapshot:s_v2").snapshot_ids == ("s_v2",)


def test_an_unparseable_selector_says_what_is_supported(repo):
    with pytest.raises(InvalidInput, match="latest"):
        resolve(repo, "whenever:soonish")


def test_an_empty_selector_is_refused(repo):
    with pytest.raises(InvalidInput):
        resolve(repo, "   ")


# -- the three failure modes ---------------------------------------------------------------------------------------
def test_an_unknown_label_is_not_found(repo):
    with pytest.raises(NotFound):
        resolve(repo, "v99")


def test_a_version_that_was_never_built_needs_an_index(repo):
    """The version exists; it has no snapshot. That is a different instruction to the caller than 'no such thing'."""
    with pytest.raises(IndexRequired):
        resolve(repo, "v4")


def test_a_repository_with_nothing_built_needs_an_index():
    with catalog.open_catalog() as db:
        catalog.register_repo(db, "empty", source_kind="jsonl")
    with pytest.raises(IndexRequired):
        resolve("empty", "latest")


def test_an_unknown_repository_is_not_found():
    with pytest.raises(NotFound):
        resolve("never-heard-of-it", "latest")


# -- as_of ---------------------------------------------------------------------------------------------------------
def test_as_of_picks_the_newest_version_at_or_before_the_instant(repo):
    with catalog.open_catalog() as db:
        rows = {r["label"]: r["created_ts"] for r in catalog.list_versions(db, repo)}
    instant = selectors.to_iso(rows["v2"] + 30)
    assert resolve(repo, f"as_of:{instant}").labels == ("v2",)


def test_as_of_before_the_first_version_is_not_found(repo):
    with catalog.open_catalog() as db:
        first = min(r["created_ts"] for r in catalog.list_versions(db, repo))
    with pytest.raises(NotFound, match="before"):
        resolve(repo, f"as_of:{selectors.to_iso(first - 600)}")


def test_as_of_exactly_on_a_boundary_takes_that_version(repo):
    with catalog.open_catalog() as db:
        rows = {r["label"]: r["created_ts"] for r in catalog.list_versions(db, repo)}
    assert resolve(repo, f"as_of:{selectors.to_iso(rows['v2'])}").labels == ("v2",)


def test_two_versions_at_the_same_instant_are_a_conflict_not_a_coin_flip(repo):
    with catalog.open_catalog() as db:
        row = catalog.get_version(db, repo, "v2")
        db.execute(
            "UPDATE versions SET created_ts = ? WHERE repo_id = ? AND label = ?", (row["created_ts"], repo, "v3")
        )
        instant = selectors.to_iso(row["created_ts"])
    with pytest.raises(VersionConflict) as excinfo:
        resolve(repo, f"as_of:{instant}")
    assert "v2" in str(excinfo.value.context) and "v3" in str(excinfo.value.context)


def test_an_unparseable_instant_is_refused(repo):
    with pytest.raises(InvalidInput, match="ISO"):
        resolve(repo, "as_of:last tuesday")


# -- what a resolution carries --------------------------------------------------------------------------------------
def test_a_resolution_says_how_it_was_reached(repo):
    resolution = resolve(repo, "latest")
    assert resolution.kind == "latest" and resolution.selector == "latest"
    assert resolution.single  # exactly one snapshot: the ordinary search path


def test_a_multi_version_resolution_is_not_single(repo):
    assert not resolve(repo, "all").single
