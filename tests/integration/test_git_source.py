"""The git source (docs/spec/04 §2, `.claude/rules/security.md`).

A repository is the most hostile source we accept: it carries hooks that want to run, submodules and LFS pointers
that want to fetch, symlinks that point anywhere, and a history somebody else wrote. So the source reads it the
way a forensic tool would — `ls-tree` and `cat-file`, no checkout, no hooks, no network — and these tests are
against a real repository built in a temporary directory, because a mock would prove nothing about `git`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from acis.core.errors import InvalidInput
from acis.core.types import Limits, SourceSpec
from acis.ingest import sources

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(repo), "GIT_CONFIG_NOSYSTEM": "1"},
    )


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """Three commits: add two files, change one, add one and delete another."""
    root = tmp_path_factory.mktemp("gitrepo")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "Test")

    (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "first")

    (root / "a.py").write_text("def a():\n    return 11\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "second")

    (root / "c.py").write_text("def c():\n    return 3\n", encoding="utf-8")
    (root / "b.py").unlink()
    (root / "notes.md").write_text("not code", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "third")
    git(root, "tag", "v3")
    return root


def read(spec, **kw):
    return list(sources.open_source(spec, **kw).versions())


def test_every_commit_becomes_a_version_oldest_first(repo):
    versions = read(SourceSpec(kind="git", location=str(repo)))
    assert len(versions) == 3
    assert [sorted(u.key for u in v.units) for v in versions] == [["a.py", "b.py"], ["a.py", "b.py"], ["a.py", "c.py"]]


def test_a_version_label_is_the_commit_it_came_from(repo):
    versions = read(SourceSpec(kind="git", location=str(repo)))
    for version in versions:
        assert len(version.label) == 12 and all(c in "0123456789abcdef" for c in version.label)
    assert len({v.label for v in versions}) == 3


def test_content_is_read_from_the_object_database_not_the_working_tree(repo):
    """No checkout happens, so a dirty working tree cannot change what a historical version contains."""
    (repo / "a.py").write_text("SCRIBBLED OVER", encoding="utf-8")
    try:
        versions = read(SourceSpec(kind="git", location=str(repo)))
        assert all("SCRIBBLED" not in u.text for v in versions for u in v.units)
    finally:
        (repo / "a.py").write_text("def a():\n    return 11\n", encoding="utf-8")


def test_only_the_configured_extensions_are_ingested(repo):
    versions = read(SourceSpec(kind="git", location=str(repo)))
    assert all(u.key.endswith(".py") for v in versions for u in v.units)


def test_the_history_can_be_limited_to_the_most_recent_commits(repo):
    versions = read(SourceSpec(kind="git", location=str(repo), options={"max_commits": 2}))
    assert len(versions) == 2  # the newest two, still oldest-first
    assert sorted(u.key for u in versions[-1].units) == ["a.py", "c.py"]


def test_a_single_revision_can_be_named(repo):
    versions = read(SourceSpec(kind="git", location=str(repo), options={"rev": "v3"}))
    assert len(versions) == 1 and sorted(u.key for u in versions[0].units) == ["a.py", "c.py"]


def test_an_unknown_revision_is_refused_rather_than_guessed(repo):
    with pytest.raises(InvalidInput, match="revision"):
        read(SourceSpec(kind="git", location=str(repo), options={"rev": "no-such-ref"}))


def test_a_directory_that_is_not_a_repository_is_refused(tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(InvalidInput, match="git repository"):
        read(SourceSpec(kind="git", location=str(tmp_path / "plain")))


def test_limits_apply_to_a_repository_too(repo):
    from acis.core.errors import ResourceLimit

    with pytest.raises(ResourceLimit):
        read(SourceSpec(kind="git", location=str(repo)), limits=Limits(max_unit_bytes=4))


def test_hooks_never_run(repo, tmp_path):
    """A repository's hooks are somebody else's code. Reading a repository must not execute any of it."""
    sentinel = tmp_path / "HOOK_RAN"
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    for name in ("post-checkout", "post-index-change", "pre-commit"):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\ntouch {sentinel}\n", encoding="utf-8")
        hook.chmod(0o755)

    read(SourceSpec(kind="git", location=str(repo)))
    assert not sentinel.exists()


def test_a_symlink_in_the_tree_is_not_followed(repo):
    """`ls-tree` reports the link's mode; the target is not ours to read."""
    (repo / "link.py").symlink_to("/etc/passwd")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "link")
    try:
        versions = read(SourceSpec(kind="git", location=str(repo)))
        assert all("link.py" not in [u.key for u in v.units] for v in versions)
    finally:
        git(repo, "reset", "-q", "--hard", "HEAD~1")


def test_the_source_never_writes_to_the_repository(repo):
    before = sorted(p.name for p in repo.iterdir())
    read(SourceSpec(kind="git", location=str(repo)))
    assert sorted(p.name for p in repo.iterdir()) == before
