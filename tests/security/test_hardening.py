"""Source-level and runtime hardening (docs/spec/05 §2, `.claude/rules/security.md`, INV-5, INV-12).

`safe_path` is the single door to the filesystem, logs never carry user text by default, and the banned constructs
scan is a cheap standing check that nothing re-introduces `eval`, `pickle` or `shell=True` into `src/`.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from acis.core.errors import InvalidInput, ResourceLimit
from acis.core.paths import acis_root
from acis.sec.paths import assert_no_symlink, is_sealed_path, safe_component, safe_path, safe_relpath, within_limit

SRC = acis_root() / "src" / "acis"


# -- safe_path (S-1…S-3) -------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidate",
    [
        "../etc/passwd",
        "a/../../b",
        "/etc/passwd",
        "C:\\Windows\\system32",
        "~/secrets",
        "a/\x00b",
        "a//../..//b",
        "..",
        ".",
        "",
        "   ",
        "\\\\server\\share",
    ],
)
def test_traversal_and_absolute_paths_are_rejected(candidate, tmp_path):
    with pytest.raises(InvalidInput):
        safe_path(tmp_path, candidate)


@pytest.mark.parametrize("candidate", ["a.txt", "dir/file.py", "a/b/c/d.json", "ünïcödé.txt"])
def test_ordinary_relative_paths_are_accepted(candidate, tmp_path):
    resolved = safe_path(tmp_path, candidate)
    assert resolved.is_relative_to(tmp_path.resolve())


def test_component_rules():
    assert safe_component("file.py") == "file.py"
    for bad in ("", ".", "..", "a/b", "a\\b", "con", "NUL", "x" * 300, "a\x00b"):
        with pytest.raises(InvalidInput):
            safe_component(bad)


def test_depth_limit():
    with pytest.raises(InvalidInput, match="nested too deeply"):
        safe_relpath("/".join("abcdefgh"[i % 8] for i in range(200)))


def test_symlinked_components_are_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    assert_no_symlink(tmp_path / "real" / "f.txt", tmp_path)
    with pytest.raises(InvalidInput, match="symlink"):
        assert_no_symlink(tmp_path / "link" / "f.txt", tmp_path)


def test_limits_raise_rather_than_truncate():
    assert within_limit(10, 10) == 10
    with pytest.raises(ResourceLimit):
        within_limit(11, 10, what="archive entry")


def test_sealed_path_detection():
    assert is_sealed_path("/home/u/.acis-sealed/hf/x.parquet")
    assert is_sealed_path("data/sealed/labels.tsv")
    assert not is_sealed_path("src/acis/appsdata/apps.py")


# -- banned constructs in src/ (INV-5, supply chain) -------------------------------------------------------------------
BANNED = [
    (re.compile(r"(?<![\w.])(eval|exec)\s*\("), "eval/exec are forbidden (INV-5)"),
    (re.compile(r"\bpickle\.loads?\(|\bcPickle\b|\bmarshal\.loads?\("), "pickle/marshal deserialisation"),
    (re.compile(r"\btorch\.load\((?![^)]*weights_only\s*=\s*True)"), "torch.load without weights_only=True"),
    (re.compile(r"\byaml\.load\((?![^)]*SafeLoader)"), "yaml.load without SafeLoader"),
    (re.compile(r"shell\s*=\s*True"), "subprocess shell=True"),
    (re.compile(r"trust_remote_code\s*=\s*True"), "trust_remote_code=True"),
    (re.compile(r"\bos\.system\(|\bcommands\.getoutput\("), "os.system"),
]


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Source lines with comments stripped — a rule quoted in a comment is documentation, not code."""
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = re.sub(r"#.*$", "", line)
        if stripped.strip():
            out.append((i, stripped))
    return out


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_banned_constructs_in_source(path):
    for lineno, line in _code_lines(path):
        for pattern, why in BANNED:
            assert not pattern.search(line), f"{path.relative_to(SRC)}:{lineno} {why}: {line.strip()[:80]}"


def test_the_adapter_never_defines_predict_in_source():
    source = (SRC / "mteb_adapter.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*def\s+predict\s*\(", source, re.MULTILINE)


def test_untrusted_code_is_parsed_not_executed():
    """The audit parses corpus code with `ast`; nothing imports, compiles or runs it (INV-5)."""
    source = (SRC / "appsdata" / "apps.py").read_text(encoding="utf-8")
    assert "ast.parse(" in source
    assert "compile(" not in source and "importlib" not in source


def test_no_query_time_network_calls_in_the_engine():
    """D15: the only network path is `acis fetch`."""
    for path in (SRC / "engine").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "requests." not in source and "urllib" not in source and "httpx" not in source


#: The complete list of modules allowed to leave the offline default. Two supply chains, two fetchers, and
#: nothing else: `appsdata/fetch.py` for the dataset (D19, the seal lives on the other side of it) and
#: `embed/modelfetch.py` for model weights (D4/G0.4, safetensors only). Adding a third is a decision, and this
#: assertion is where that decision has to be made rather than noticed later.
ONLINE_MODULES = ["appsdata/fetch.py", "embed/modelfetch.py"]


def test_only_the_declared_modules_go_online():
    online = sorted(
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if "HF_HUB_OFFLINE" in p.read_text(encoding="utf-8")
        and 'os.environ["HF_HUB_OFFLINE"] = "0"' in p.read_text(encoding="utf-8")
    )
    assert online == sorted(ONLINE_MODULES)


# -- log redaction (O-1) --------------------------------------------------------------------------------------------
def test_logs_carry_lengths_not_query_text(monkeypatch):
    from acis.obs import log

    monkeypatch.delenv(log.TEXT_ALLOWED, raising=False)
    buffer = io.StringIO()
    event = log._redact(None, "info", {"event": "search", "query": "a secret problem statement", "text": "body"})
    assert "query" not in event and "text" not in event
    assert event["query_len"] == len("a secret problem statement")
    assert event["text_len"] == 4
    buffer.close()


def test_opt_in_text_logging_is_truncated(monkeypatch):
    from acis.obs import log

    monkeypatch.setenv(log.TEXT_ALLOWED, "1")
    event = log._redact(None, "info", {"query": "x" * 1000})
    assert len(event["query_redacted"]) <= 200


def test_a_structured_log_line_is_json(capsys, monkeypatch):
    from acis.obs.log import configure, get_logger

    configure(level="INFO")
    get_logger("acis.test").info("unit_event", snapshot_id="s_1", stage_ms=3)
    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "unit_event" and payload["snapshot_id"] == "s_1"


# -- INV-11: the engine knows nothing about mteb -------------------------------------------------------------------
ENGINE_SIDE = ("engine", "core", "embed", "rank", "prep", "lexical")


@pytest.mark.parametrize("package", ENGINE_SIDE)
def test_the_engine_never_imports_mteb(package):
    """INV-11: the adapter translates; the engine must not know the harness exists.

    Checked on import statements only — a docstring mentioning mteb is documentation, not a dependency.
    """
    offenders: list[str] = []
    for path in (SRC / package).rglob("*.py"):
        for lineno, line in _code_lines(path):
            if re.match(r"\s*(import\s+mteb|from\s+mteb[\s.])", line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}")
    assert not offenders, f"{package} imports mteb (INV-11): {offenders}"


def test_the_engine_package_imports_cleanly_without_mteb_loaded():
    """A stronger form of INV-11: importing the engine must not pull mteb in transitively."""
    import json
    import subprocess
    import sys

    code = (
        "import json, sys\n"
        "import acis.engine\n"
        "print(json.dumps([m for m in sys.modules if m.split('.')[0] == 'mteb']))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == []


def test_the_robustness_fixture_meets_its_declared_minimum():
    """docs/spec/10 §8: a corpus below ~200 documents cannot expose a template-coupled ranking flip."""
    from acis import robust_hook

    assert robust_hook.corpus_size() >= robust_hook.MIN_DOCS


def test_logging_cannot_break_the_operation_it_observes():
    """A closed or replaced stderr must not raise inside the code being logged (O-1's quieter sibling).

    structlog binds the stream when logging is configured; pytest's capture replaces and closes it routinely, and
    a daemonised run can too. A log call is an observation, never a failure mode.
    """
    import io
    import sys

    from acis.obs.log import configure, get_logger

    configure(level="INFO")
    closed = io.StringIO()
    closed.close()
    original = sys.stderr
    sys.stderr = closed
    try:
        get_logger("acis.test").info("into_the_void", snapshot_id="s_1")  # must not raise
    finally:
        sys.stderr = original
