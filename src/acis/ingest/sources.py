"""Sources: turning somebody else's layout into versions of units (docs/spec/04 §2, INV-5).

Four shapes, one internal model. A source yields `VersionUnits(label, units)` in history order and nothing else;
what it does *not* do is as important:

* it never extracts an archive — members are streamed, and a member that names a path outside the archive, or
  that is a symlink, is refused rather than skipped, because an archive that contains one is not a corpus;
* it never follows a symlink out of a directory tree;
* it never imports, executes, compiles-for-execution or `eval`s what it reads. Python is parsed with `ast` for one
  boolean (`parse_ok`), and a file that will not parse is still ingested — a corpus we cannot parse is still a
  corpus we must be able to search (spec 04 §2).

Limits are enforced while reading, before the bytes are held: a declared size is checked against the cap before
anything is decompressed, so a small archive claiming four gigabytes is refused rather than materialised.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from acis.core.errors import InvalidInput, ResourceLimit
from acis.core.hashing import hash_obj, short
from acis.core.types import Limits, SourceSpec
from acis.obs.log import get_logger
from acis.sec.paths import safe_relpath
from acis.store.layout import NAME as LABEL_PATTERN
from acis.store.snapshots import UnitInput

log = get_logger("acis.ingest")

DEFAULT_VERSION = "v1"
DEFAULT_EXTENSIONS = (".py",)
MAX_KEY_CHARS = 512
#: `v2` sorts before `v10` only if the digits are compared as numbers. Every real history depends on this.
_NATURAL = re.compile(r"(\d+)")


@dataclass(frozen=True, slots=True)
class VersionUnits:
    """One version of a corpus: an ordered list of units under a label the store can use as a directory name."""

    label: str
    units: tuple[UnitInput, ...]
    source_id: str = ""

    @property
    def n_bytes(self) -> int:
        return sum(len(u.text.encode("utf-8")) for u in self.units)


class Source(Protocol):
    """Everything the builder needs from a source."""

    def versions(self) -> Iterator[VersionUnits]: ...


def natural_key(name: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in _NATURAL.split(name))


def source_id(spec: SourceSpec) -> str:
    """Stable identity of a source *including its filters* — they changed what was read (spec 04 §2)."""
    return short(
        hash_obj(
            {
                "kind": spec.kind,
                "location": str(Path(spec.location).expanduser()),
                "options": dict(spec.options),
            }
        ),
        16,
    )


# -- shared checks --------------------------------------------------------------------------------------------
@dataclass(slots=True)
class _Budget:
    """Running totals, so a limit is hit while reading rather than after everything is in memory."""

    limits: Limits
    units: int = 0
    n_bytes: int = 0

    def account(self, key: str, text: str) -> None:
        size = len(text.encode("utf-8"))
        if size > self.limits.max_unit_bytes:
            raise ResourceLimit(
                "unit exceeds the maximum size", unit=key[:80], size=size, limit=self.limits.max_unit_bytes
            )
        self.units += 1
        self.n_bytes += size
        if self.units > self.limits.max_units:
            raise ResourceLimit("source has more units than the limit allows", limit=self.limits.max_units)
        if self.n_bytes > self.limits.max_bytes:
            raise ResourceLimit("source is larger than the limit allows", limit=self.limits.max_bytes)


def clean_text(text: str) -> str:
    """NUL bytes never reach the store: they are not content, and they break every downstream tool."""
    return text.replace("\x00", "")


def check_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise InvalidInput("a unit needs a non-empty identifier")
    if len(key) > MAX_KEY_CHARS:
        raise InvalidInput("unit identifier is too long", length=len(key), limit=MAX_KEY_CHARS)
    if ".." in key or key.startswith("/") or "\x00" in key:
        raise InvalidInput("unit identifier looks like a path, which it may never be", key=key[:80])
    return key


def check_label(label: str) -> str:
    if not LABEL_PATTERN.match(str(label)):
        raise InvalidInput(
            "a version label must be 1-64 characters of [A-Za-z0-9._-] and start alphanumerically",
            label=str(label)[:80],
        )
    return str(label)


def parse_meta(text: str, key: str) -> dict[str, Any]:
    """Parse-only inspection (INV-5): `ast.parse` builds a tree, it never runs anything.

    `compile(..., "exec")` and `eval` are the two things that would, and neither appears here or anywhere else in
    the ingest path. A file that will not parse is data all the same, so the failure is a flag.
    """
    if not key.endswith(".py") and not _looks_like_python(text):
        return {"parse_ok": None}
    try:
        # Other people's code is full of `SyntaxWarning`s (invalid escapes, mostly): true facts about the file,
        # noise in our logs. We read its shape; we are not responsible for its string literals.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return {"parse_ok": False}
    return {"parse_ok": True}


def _looks_like_python(text: str) -> bool:
    head = text[:2000]
    return any(token in head for token in ("def ", "class ", "import ", "print(", "return "))


def _unit(key: str, text: str, *, budget: _Budget) -> UnitInput:
    cleaned = clean_text(text)
    budget.account(key, cleaned)
    return UnitInput(key=check_key(key), text=cleaned, meta=parse_meta(cleaned, key))


def _assemble(label: str, pairs: Sequence[tuple[str, str]], *, budget: _Budget, sid: str) -> VersionUnits:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise InvalidInput("duplicate unit keys in one version", version=label, keys=duplicates[:5])
    if not pairs:
        raise InvalidInput("version contains no units after filtering", version=label)
    return VersionUnits(
        label=check_label(label),
        units=tuple(_unit(k, t, budget=budget) for k, t in pairs),
        source_id=sid,
    )


# -- JSONL ---------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class JsonlSource:
    """`{id|snippet_id, version, text}` per line — the shape P1 and Bonus data is most likely to arrive in."""

    path: Path
    limits: Limits = field(default_factory=Limits)
    sid: str = ""

    def versions(self) -> Iterator[VersionUnits]:
        budget = _Budget(self.limits)
        grouped: dict[str, list[tuple[str, str]]] = {}
        with open(self.path, encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InvalidInput(f"{self.path.name} line {number} is not valid JSON", detail=str(exc)) from exc
                if not isinstance(row, dict):
                    raise InvalidInput(f"{self.path.name} line {number} is not a JSON object")
                key = row.get("id") or row.get("snippet_id") or row.get("unit_id")
                if key is None:
                    raise InvalidInput(f"{self.path.name} line {number} has no identifier (id / snippet_id)")
                if "text" not in row or not isinstance(row["text"], str):
                    raise InvalidInput(f"{self.path.name} line {number} has no text")
                grouped.setdefault(check_label(str(row.get("version") or DEFAULT_VERSION)), []).append(
                    (str(key), row["text"])
                )
        for label, pairs in grouped.items():  # insertion order = the order versions first appeared
            yield _assemble(label, pairs, budget=budget, sid=self.sid)


# -- directory sequences ----------------------------------------------------------------------------------------
@dataclass(slots=True)
class DirSequenceSource:
    """`v1/`, `v2/`, … or a single flat directory. Symlinks are never followed, in or out."""

    root: Path
    limits: Limits = field(default_factory=Limits)
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    sid: str = ""

    def versions(self) -> Iterator[VersionUnits]:
        budget = _Budget(self.limits)
        subdirs = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: natural_key(p.name)
        )
        if not subdirs:
            yield _assemble(DEFAULT_VERSION, self._read(self.root), budget=budget, sid=self.sid)
            return
        for directory in subdirs:
            yield _assemble(directory.name, self._read(directory), budget=budget, sid=self.sid)

    def _read(self, directory: Path) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for path in sorted(directory.rglob("*"), key=lambda p: natural_key(str(p))):
            if path.is_symlink() or not path.is_file() or path.suffix not in self.extensions:
                continue
            if any(parent.is_symlink() for parent in path.relative_to(directory).parents if parent != Path()):
                continue
            pairs.append((path.relative_to(directory).as_posix(), path.read_text(encoding="utf-8", errors="replace")))
        return pairs


# -- archives -------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class ArchiveSequenceSource:
    """A ZIP laid out like a directory sequence. Streamed: nothing is ever written to disk."""

    path: Path
    limits: Limits = field(default_factory=Limits)
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    sid: str = ""

    def versions(self) -> Iterator[VersionUnits]:
        budget = _Budget(self.limits)
        try:
            archive = zipfile.ZipFile(self.path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise InvalidInput(f"{self.path.name} is not a readable archive", detail=str(exc)) from exc

        grouped: dict[str, list[tuple[str, str]]] = {}
        with archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative = self._member_path(info)
                if relative.suffix not in self.extensions:
                    continue
                # The *declared* size is checked before decompressing: a bomb is refused, never materialised.
                if info.file_size > self.limits.max_unit_bytes:
                    raise ResourceLimit(
                        "archive member declares a size over the limit",
                        member=info.filename[:80],
                        size=info.file_size,
                        limit=self.limits.max_unit_bytes,
                    )
                parts = relative.parts
                label = parts[0] if len(parts) > 1 else DEFAULT_VERSION
                key = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
                with archive.open(info) as fh:
                    text = fh.read().decode("utf-8", errors="replace")
                grouped.setdefault(check_label(label), []).append((key, text))

        for label, pairs in grouped.items():
            yield _assemble(label, pairs, budget=budget, sid=self.sid)

    @staticmethod
    def _member_path(info: zipfile.ZipInfo) -> Path:
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise InvalidInput("archive member is a symlink, which is never followed", member=info.filename[:80])
        return Path(safe_relpath(info.filename))


# -- in memory ---------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class MemorySource:
    """`{label: {key: text}}`. For tests and the scripted demo; never a production ingest path."""

    versions_in: Mapping[str, Mapping[str, str]]
    limits: Limits = field(default_factory=Limits)
    sid: str = ""

    def versions(self) -> Iterator[VersionUnits]:
        budget = _Budget(self.limits)
        for label, units in self.versions_in.items():
            yield _assemble(label, list(units.items()), budget=budget, sid=self.sid)


# -- the factory ---------------------------------------------------------------------------------------------------
def open_source(spec: SourceSpec, *, limits: Limits | None = None) -> Source:
    """Build the source a `SourceSpec` names. Unknown kinds say what is supported rather than guessing."""
    bounds = limits or Limits()
    sid = source_id(spec)
    options = dict(spec.options)
    extensions = tuple(str(e) for e in options.get("extensions", DEFAULT_EXTENSIONS))

    if spec.kind == "memory":
        versions = options.get("versions") or {}
        if not isinstance(versions, Mapping):
            raise InvalidInput("a memory source needs options['versions'] = {label: {key: text}}")
        return MemorySource(versions_in=versions, limits=bounds, sid=sid)

    location = Path(spec.location).expanduser()
    if spec.kind in ("jsonl", "zip") and not location.is_file():
        raise InvalidInput(f"{spec.kind} source does not exist", location=str(location))
    if spec.kind == "dir" and not location.is_dir():
        raise InvalidInput("dir source does not exist", location=str(location))

    if spec.kind == "jsonl":
        return JsonlSource(path=location, limits=bounds, sid=sid)
    if spec.kind == "dir":
        return DirSequenceSource(root=location, limits=bounds, extensions=extensions, sid=sid)
    if spec.kind == "zip":
        return ArchiveSequenceSource(path=location, limits=bounds, extensions=extensions, sid=sid)
    if spec.kind == "git":
        from acis.ingest.git import DEFAULT_MAX_COMMITS, GitSource  # noqa: PLC0415 — only this path needs git

        if not location.is_dir():
            raise InvalidInput("git source does not exist", location=str(location))
        return GitSource(
            root=location,
            limits=bounds,
            extensions=extensions,
            rev=str(options.get("rev") or spec.version or ""),
            max_commits=int(options.get("max_commits", DEFAULT_MAX_COMMITS)),
            sid=sid,
        )
    raise InvalidInput(
        f"unknown source kind {spec.kind!r}; supported: jsonl, dir, zip, git, memory", kind=str(spec.kind)
    )


__all__ = [
    "DEFAULT_EXTENSIONS",
    "DEFAULT_VERSION",
    "MAX_KEY_CHARS",
    "ArchiveSequenceSource",
    "DirSequenceSource",
    "JsonlSource",
    "MemorySource",
    "Source",
    "VersionUnits",
    "check_key",
    "check_label",
    "clean_text",
    "natural_key",
    "open_source",
    "parse_meta",
    "source_id",
]
