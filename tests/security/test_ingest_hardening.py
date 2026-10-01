"""Hostile sources (docs/spec/05 §2, INV-5, docs/DESIGN_RULES.md).

Everything a source reads was written by somebody else. The archive, the directory tree and the JSONL file are
all attacker-controlled in the threat model, so each of these tests is an attack that a naive loader falls for:
an archive member that writes outside the destination, a symlink that reaches out of the tree, a small file that
expands to gigabytes, and a Python file that would run code the moment anyone was careless enough to import it.

The rule the last one checks is absolute: untrusted code is **parsed**, never imported, executed, compiled for
execution or `eval`'d.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from acis.core.errors import InvalidInput, ResourceLimit
from acis.core.types import Limits, SourceSpec
from acis.ingest import sources


def read(spec, **kw):
    return list(sources.open_source(spec, **kw).versions())


# -- archives ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "member",
    [
        "../escape.py",
        "v1/../../escape.py",
        "/absolute.py",
        "v1/sub/../../../escape.py",
        "C:\\windows\\system32\\evil.py",
    ],
)
def test_an_archive_member_cannot_name_a_path_outside_the_archive(tmp_path, member):
    path = tmp_path / "hostile.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, "x = 1")
    with pytest.raises(InvalidInput):
        read(SourceSpec(kind="zip", location=str(path)))


def test_a_symlink_member_is_refused(tmp_path):
    """A symlink in an archive is a request to read something the archive does not contain."""
    path = tmp_path / "link.zip"
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("v1/link.py")
        info.external_attr = 0o120777 << 16  # S_IFLNK
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(InvalidInput, match="symlink"):
        read(SourceSpec(kind="zip", location=str(path)))


def test_a_zip_bomb_is_refused_before_it_is_read(tmp_path):
    """The declared size is checked first: a 1 MB archive that claims 4 GB never gets decompressed."""
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v1/big.py", "0" * (8 << 20))
    with pytest.raises(ResourceLimit):
        read(SourceSpec(kind="zip", location=str(path)), limits=Limits(max_unit_bytes=1 << 20))


def test_an_archive_with_absurdly_many_members_is_refused(tmp_path):
    path = tmp_path / "many.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(50):
            zf.writestr(f"v1/f{i}.py", "x = 1")
    with pytest.raises(ResourceLimit):
        read(SourceSpec(kind="zip", location=str(path)), limits=Limits(max_units=10))


def test_a_file_that_is_not_an_archive_is_refused_clearly(tmp_path):
    path = tmp_path / "not.zip"
    path.write_text("this is not a zip file", encoding="utf-8")
    with pytest.raises(InvalidInput, match="archive"):
        read(SourceSpec(kind="zip", location=str(path)))


# -- directories ---------------------------------------------------------------------------------------------------
def test_a_symlink_in_a_directory_source_is_not_followed(tmp_path):
    root = tmp_path / "repo" / "v1"
    root.mkdir(parents=True)
    (root / "real.py").write_text("a = 1", encoding="utf-8")
    secret = tmp_path / "secret.py"
    secret.write_text("password = 'hunter2'", encoding="utf-8")
    (root / "link.py").symlink_to(secret)

    units = read(SourceSpec(kind="dir", location=str(tmp_path / "repo")))[0].units
    assert [u.key for u in units] == ["real.py"]
    assert all("hunter2" not in u.text for u in units)


def test_a_version_directory_that_is_a_symlink_is_not_followed(tmp_path):
    root = tmp_path / "repo"
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "a.py").write_text("a = 1", encoding="utf-8")
    outside = tmp_path / "outside"
    (outside).mkdir()
    (outside / "b.py").write_text("b = 2", encoding="utf-8")
    (root / "v2").symlink_to(outside, target_is_directory=True)

    assert [v.label for v in read(SourceSpec(kind="dir", location=str(root)))] == ["v1"]


# -- content -------------------------------------------------------------------------------------------------------
def test_untrusted_python_is_parsed_never_executed(tmp_path):
    """INV-5. The file writes a sentinel at import time; ingesting it must leave no sentinel."""
    sentinel = tmp_path / "SIDE_EFFECT"
    hostile = f"import pathlib\npathlib.Path({str(sentinel)!r}).write_text('executed')\n"
    spec = SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl"))
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "evil", "version": "v1", "text": hostile}) + "\n")

    units = read(spec)[0].units
    assert not sentinel.exists()
    assert units[0].text == hostile  # stored verbatim: it is data


def test_unparseable_python_is_still_ingested(tmp_path):
    """Spec 04 §2: parse failure is a flag, never an error — the unit stays retrievable."""
    broken = "def f(:\n  this is not python\n"
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "broken", "version": "v1", "text": broken}) + "\n")
    unit = read(SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl")))[0].units[0]
    assert unit.text == broken
    assert unit.meta.get("parse_ok") is False


def test_valid_python_is_marked_as_parsed(tmp_path):
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "ok", "version": "v1", "text": "def f():\n    pass\n"}) + "\n")
    unit = read(SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl")))[0].units[0]
    assert unit.meta.get("parse_ok") is True


def test_a_pathological_nesting_depth_does_not_crash_the_parser(tmp_path):
    """`ast.parse` raises `RecursionError` (or `MemoryError`) on deep nesting; neither may escape as a crash."""
    deep = "x = " + "(" * 200 + "1" + ")" * 200
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "deep", "version": "v1", "text": deep}) + "\n")
    unit = read(SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl")))[0].units[0]
    assert unit.text == deep  # ingested either way; only the flag differs


def test_a_null_byte_in_content_does_not_reach_the_store(tmp_path):
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "nul", "version": "v1", "text": "a\x00b"}) + "\n")
    unit = read(SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl")))[0].units[0]
    assert "\x00" not in unit.text


def test_a_hostile_unit_key_cannot_become_a_path(tmp_path):
    """Unit keys are opaque labels, not paths — but they are long-lived, so they are bounded and sanitised."""
    rows = [{"id": "../../etc/passwd", "version": "v1", "text": "t"}, {"id": "x" * 5000, "version": "v1", "text": "t"}]
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(InvalidInput):
        read(SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl")))


def test_a_hostile_version_label_is_refused(tmp_path):
    """Version labels become directory names downstream. `../` is not a version."""
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "a", "version": "../../etc", "text": "t"}) + "\n")
    with pytest.raises(InvalidInput):
        read(SourceSpec(kind="jsonl", location=str(tmp_path / "c.jsonl")))
