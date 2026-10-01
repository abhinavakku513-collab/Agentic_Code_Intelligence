"""Documentation must not cite evidence that does not exist (docs/DESIGN_RULES.md, INV-14).

Two classes of false claim are cheap to catch mechanically and were both found by hand during Phase 1:

* a contract row or report citing a test file that was renamed or never written;
* a number about *our* system in a report with no `[ledger:<run_id>]`, test id or artifact path behind it.

Package contracts and phase reports are the documents a gatekeeper reads first, so they are the ones checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from acis.core.paths import acis_root

TEST_PATH = re.compile(r"`?(tests/[A-Za-z0-9_/]+\.py)(::[A-Za-z0-9_]+)?`?")
SRC_PATH = re.compile(r"`(src/acis/[A-Za-z0-9_/]+\.py)`")
LEDGER_CITATION = re.compile(r"\[ledger:[a-z0-9-]+\]")

DOCS = sorted(
    [*acis_root().glob("src/acis/*/README.md"), *acis_root().glob("docs/history/*_REPORT.md")],
    key=lambda p: str(p),
)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(acis_root())))
def test_cited_test_files_exist(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    missing = sorted({m.group(1) for m in TEST_PATH.finditer(text) if not (acis_root() / m.group(1)).is_file()})
    assert not missing, f"{doc.relative_to(acis_root())} cites test files that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(acis_root())))
def test_cited_source_files_exist(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    missing = sorted({m.group(1) for m in SRC_PATH.finditer(text) if not (acis_root() / m.group(1)).is_file()})
    assert not missing, f"{doc.relative_to(acis_root())} cites source files that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(acis_root())))
def test_cited_test_functions_exist(doc: Path) -> None:
    """A `file.py::test_name` citation has to name a test that is actually defined there."""
    text = doc.read_text(encoding="utf-8")
    problems: list[str] = []
    for match in TEST_PATH.finditer(text):
        path, func = match.group(1), (match.group(2) or "")[2:]
        if not func:
            continue
        target = acis_root() / path
        if target.is_file() and f"def {func}(" not in target.read_text(encoding="utf-8"):
            problems.append(f"{path}::{func}")
    assert not problems, f"{doc.relative_to(acis_root())} cites tests that are not defined: {problems}"


def test_every_ledger_citation_names_a_real_row() -> None:
    """INV-14: a number citing `[ledger:<id>]` must be traceable to a row that exists."""
    from acis.eval import ledger

    known = {row.run_id for row in ledger.read_rows()}
    if not known:
        pytest.skip("no ledger rows yet")
    problems: list[str] = []
    for doc in DOCS:
        for citation in LEDGER_CITATION.findall(doc.read_text(encoding="utf-8")):
            run_id = citation[len("[ledger:") : -1]
            if run_id not in known:
                problems.append(f"{doc.relative_to(acis_root())}: {citation}")
    assert not problems, f"ledger citations with no matching row: {problems}"


def test_every_package_has_a_contract() -> None:
    """docs/DESIGN_RULES.md: a package directory gets a README.md stating its contract."""
    packages = [p for p in (acis_root() / "src" / "acis").iterdir() if p.is_dir() and (p / "__init__.py").is_file()]
    missing = sorted(p.name for p in packages if not (p / "README.md").is_file())
    assert not missing, f"packages without a contract: {missing}"


def test_no_todo_markers_in_source() -> None:
    """Definition of Done: no TODO/FIXME in `src/`."""
    offenders: list[str] = []
    for path in (acis_root() / "src" / "acis").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line):
                offenders.append(f"{path.relative_to(acis_root())}:{lineno}")
    assert not offenders, f"TODO/FIXME markers in src/: {offenders}"
