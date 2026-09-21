"""One `safe_path()` for every filesystem access (docs/spec/05 §2, `.claude/rules/security.md`).

Untrusted archives, git trees and JSONL records name their own paths. Every one of them is resolved through this
module: no escape from the destination root, no symlink traversal, no absolute paths, no NUL bytes, no Windows
reserved names, no path component games.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path, PurePosixPath

from acis.core.errors import InvalidInput

_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_COMPONENT = 255
MAX_DEPTH = 64


def safe_component(name: str) -> str:
    """Validate a single path component. Raises `InvalidInput` for anything that is not an ordinary file name."""
    if not name or name in (".", ".."):
        raise InvalidInput("path component is empty or relative", component=name)
    if _CONTROL.search(name):
        raise InvalidInput("path component contains control characters")
    if "/" in name or "\\" in name:
        raise InvalidInput("path component contains a separator", component=name)
    if len(name.encode("utf-8")) > MAX_COMPONENT:
        raise InvalidInput("path component is too long", length=len(name))
    if _WINDOWS_RESERVED.match(name):
        raise InvalidInput("path component is a reserved device name", component=name)
    return name


def safe_relpath(candidate: str) -> PurePosixPath:
    """Normalise an untrusted *relative* path. Raises for absolute paths, drive letters, `..` and control characters."""
    if not isinstance(candidate, str) or not candidate.strip():
        raise InvalidInput("empty path")
    text = unicodedata.normalize("NFC", candidate).replace("\\", "/")
    if _CONTROL.search(text):
        raise InvalidInput("path contains control characters")
    if text.startswith("/") or re.match(r"^[a-zA-Z]:", text) or text.startswith("~"):
        raise InvalidInput("absolute paths are rejected", path=text[:80])
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts:
        raise InvalidInput("path resolves to nothing", path=text[:80])
    if len(parts) > MAX_DEPTH:
        raise InvalidInput("path is nested too deeply", depth=len(parts))
    for part in parts:
        safe_component(part)
    return PurePosixPath(*parts)


def safe_path(root: str | Path, candidate: str) -> Path:
    """Resolve `candidate` **inside** `root`. The result is guaranteed to be under the real (symlink-free) root."""
    root_path = Path(root).resolve()
    target = (root_path / safe_relpath(candidate)).resolve()
    if target != root_path and root_path not in target.parents:
        raise InvalidInput("path escapes its root", root=str(root_path))
    return target


def assert_no_symlink(path: str | Path, root: str | Path) -> Path:
    """Reject a path whose own name, or any component between `root` and it, is a symlink."""
    p = Path(path).absolute()
    root_path = Path(root).resolve()
    chain = [node for node in (p, *p.parents) if node == root_path or root_path in node.parents]
    for node in chain:
        if node != root_path and node.is_symlink():
            raise InvalidInput("symlinks are not followed", path=str(node))
    return p


def within_limit(n_bytes: int, limit: int, *, what: str = "payload") -> int:
    from acis.core.errors import ResourceLimit  # noqa: PLC0415 — keeps the import graph shallow

    if n_bytes > limit:
        raise ResourceLimit(f"{what} exceeds the configured limit", size=n_bytes, limit=limit)
    return n_bytes


def is_sealed_path(path: str | Path) -> bool:
    """True if `path` points at the physical TEST seal (`~/.acis-sealed`, `data/sealed/`) — see INV-8."""
    text = str(Path(path)).replace(os.sep, "/")
    return (
        "/.acis-sealed" in text
        or text.startswith(".acis-sealed")
        or "/data/sealed/" in text
        or text.startswith("data/sealed/")
    )


__all__ = [
    "MAX_COMPONENT",
    "MAX_DEPTH",
    "assert_no_symlink",
    "is_sealed_path",
    "safe_component",
    "safe_path",
    "safe_relpath",
    "within_limit",
]
