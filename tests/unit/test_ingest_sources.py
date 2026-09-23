"""Sources: JSONL, directory sequences and archives (docs/spec/04 §2, D11, INV-5).

A source's only job is to turn somebody else's layout into `(version, units)` without ever trusting it. The
interesting cases are not the happy ones — they are a version directory named `v10`, an archive member called
`../../etc/passwd`, a line of JSON with no text, and a file that would take a gigabyte of RAM to read. Each of
those has a defined answer here, because each of them decides whether P1 can ingest a corpus it did not write.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from acis.core.errors import InvalidInput, ResourceLimit
from acis.core.types import Limits, SourceSpec
from acis.ingest import sources

ROWS = [
    {"id": "a", "version": "v1", "text": "def a():\n    return 1\n"},
    {"id": "b", "version": "v1", "text": "def b():\n    return 2\n"},
    {"id": "a", "version": "v2", "text": "def a():\n    return 11\n"},
]


def jsonl(tmp_path, rows=ROWS, name="corpus.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return SourceSpec(kind="jsonl", location=str(path))


def read(spec, **kw):
    return list(sources.open_source(spec, **kw).versions())


# -- JSONL -------------------------------------------------------------------------------------------------------
def test_jsonl_groups_rows_into_versions_in_the_order_they_first_appear(tmp_path):
    versions = read(jsonl(tmp_path))
    assert [v.label for v in versions] == ["v1", "v2"]
    assert [u.key for u in versions[0].units] == ["a", "b"]
    assert versions[1].units[0].text.endswith("return 11\n")


def test_jsonl_accepts_either_spelling_of_the_identifier(tmp_path):
    spec = jsonl(tmp_path, [{"snippet_id": "x", "version": "v1", "text": "t"}])
    assert read(spec)[0].units[0].key == "x"


def test_a_row_without_a_version_belongs_to_the_first_one(tmp_path):
    """The P1 shape may be a flat corpus. That is one version, not an error."""
    spec = jsonl(tmp_path, [{"id": "a", "text": "t"}, {"id": "b", "text": "u"}])
    versions = read(spec)
    assert len(versions) == 1 and versions[0].label == sources.DEFAULT_VERSION


def test_blank_lines_are_skipped_but_malformed_json_names_its_line(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a", "text": "t"}\n\n   \nnot json at all\n', encoding="utf-8")
    with pytest.raises(InvalidInput, match="line 4"):
        read(SourceSpec(kind="jsonl", location=str(path)))


def test_a_row_without_text_is_refused_rather_than_stored_as_empty(tmp_path):
    with pytest.raises(InvalidInput, match="text"):
        read(jsonl(tmp_path, [{"id": "a", "version": "v1"}]))


def test_a_row_without_an_identifier_is_refused(tmp_path):
    with pytest.raises(InvalidInput, match="identifier"):
        read(jsonl(tmp_path, [{"version": "v1", "text": "t"}]))


def test_the_same_key_twice_in_one_version_is_refused(tmp_path):
    """A version is a set of units keyed by id; two rows for one key means the source is ambiguous, not merged."""
    rows = [{"id": "a", "version": "v1", "text": "one"}, {"id": "a", "version": "v1", "text": "two"}]
    with pytest.raises(InvalidInput, match="duplicate"):
        read(jsonl(tmp_path, rows))


def test_a_missing_file_is_an_error_a_reader_can_act_on(tmp_path):
    with pytest.raises(InvalidInput, match="does not exist"):
        read(SourceSpec(kind="jsonl", location=str(tmp_path / "nope.jsonl")))


# -- directory sequences --------------------------------------------------------------------------------------------
def make_tree(tmp_path, tree):
    root = tmp_path / "repo"
    for version, files in tree.items():
        for name, text in files.items():
            path = root / version / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return SourceSpec(kind="dir", location=str(root))


def test_a_directory_sequence_orders_versions_naturally(tmp_path):
    """`v10` comes after `v9`, which lexicographic order gets wrong and every real history depends on."""
    spec = make_tree(tmp_path, {f"v{i}": {"m.py": f"x = {i}"} for i in (1, 2, 9, 10, 11)})
    assert [v.label for v in read(spec)] == ["v1", "v2", "v9", "v10", "v11"]


def test_nested_paths_become_unit_keys(tmp_path):
    spec = make_tree(tmp_path, {"v1": {"pkg/mod.py": "a = 1", "top.py": "b = 2"}})
    assert [u.key for u in read(spec)[0].units] == ["pkg/mod.py", "top.py"]


def test_only_the_configured_extensions_are_ingested(tmp_path):
    spec = make_tree(tmp_path, {"v1": {"keep.py": "a = 1", "skip.md": "# no", "skip.txt": "no"}})
    assert [u.key for u in read(spec)[0].units] == ["keep.py"]


def test_a_directory_with_no_version_subdirectories_is_one_version(tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    (root / "a.py").write_text("a = 1", encoding="utf-8")
    versions = read(SourceSpec(kind="dir", location=str(root)))
    assert len(versions) == 1 and [u.key for u in versions[0].units] == ["a.py"]


def test_a_version_with_no_matching_files_is_reported_not_silently_empty(tmp_path):
    spec = make_tree(tmp_path, {"v1": {"a.py": "x = 1"}, "v2": {"notes.md": "nothing"}})
    with pytest.raises(InvalidInput, match="no units"):
        read(spec)


# -- archives ----------------------------------------------------------------------------------------------------
def make_zip(tmp_path, members, name="repo.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member, text in members.items():
            zf.writestr(member, text)
    return SourceSpec(kind="zip", location=str(path))


def test_an_archive_yields_the_same_versions_as_the_directory_would(tmp_path):
    spec = make_zip(tmp_path, {"v1/a.py": "a = 1", "v1/b.py": "b = 2", "v2/a.py": "a = 11"})
    versions = read(spec)
    assert [v.label for v in versions] == ["v1", "v2"]
    assert [u.key for u in versions[0].units] == ["a.py", "b.py"]


def test_an_archive_is_streamed_and_never_extracted(tmp_path):
    spec = make_zip(tmp_path, {"v1/a.py": "a = 1"})
    read(spec)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["repo.zip"]


def test_directories_inside_an_archive_are_not_units(tmp_path):
    path = tmp_path / "d.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("v1/", "")
        zf.writestr("v1/a.py", "a = 1")
    versions = read(SourceSpec(kind="zip", location=str(path)))
    assert [u.key for u in versions[0].units] == ["a.py"]


# -- limits ------------------------------------------------------------------------------------------------------
def test_a_unit_larger_than_the_limit_is_refused_not_truncated(tmp_path):
    spec = jsonl(tmp_path, [{"id": "a", "version": "v1", "text": "x" * 5000}])
    with pytest.raises(ResourceLimit):
        read(spec, limits=Limits(max_unit_bytes=1000))


def test_more_units_than_the_limit_is_refused(tmp_path):
    rows = [{"id": f"u{i}", "version": "v1", "text": "t"} for i in range(20)]
    with pytest.raises(ResourceLimit):
        read(jsonl(tmp_path, rows), limits=Limits(max_units=5))


def test_more_bytes_than_the_limit_is_refused(tmp_path):
    rows = [{"id": f"u{i}", "version": "v1", "text": "x" * 500} for i in range(20)]
    with pytest.raises(ResourceLimit):
        read(jsonl(tmp_path, rows), limits=Limits(max_bytes=1000))


# -- identity ------------------------------------------------------------------------------------------------------
def test_the_source_identity_covers_the_filters_that_changed_what_was_read(tmp_path):
    """Filters are part of the build config (spec 04 §2): ingesting `.py` only is a different corpus."""
    base = make_tree(tmp_path, {"v1": {"a.py": "x = 1"}})
    other = SourceSpec(kind="dir", location=base.location, options={"extensions": [".py", ".md"]})
    assert sources.source_id(base) != sources.source_id(other)


def test_the_same_source_read_twice_is_identical(tmp_path):
    spec = jsonl(tmp_path)
    first, second = read(spec), read(spec)
    assert [(v.label, [(u.key, u.text) for u in v.units]) for v in first] == [
        (v.label, [(u.key, u.text) for u in v.units]) for v in second
    ]


def test_an_unknown_source_kind_names_what_is_supported(tmp_path):
    with pytest.raises(InvalidInput, match="jsonl"):
        read(SourceSpec(kind="carrier-pigeon", location=str(tmp_path)))  # type: ignore[arg-type]


def test_memory_sources_exist_for_tests_and_the_demo():
    spec = SourceSpec(kind="memory", location="demo", options={"versions": {"v1": {"a": "text"}}})
    versions = read(spec)
    assert [v.label for v in versions] == ["v1"] and versions[0].units[0].text == "text"
