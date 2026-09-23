"""The git source (docs/spec/04 §2, `.claude/rules/security.md`, INV-5).

A repository is read the way a forensic tool reads one: `rev-list` for the history, `ls-tree` for each commit's
files, `cat-file --batch` for their contents. There is **no checkout**, so nothing in the working tree matters and
nothing of ours is ever written into somebody else's repository.

Every invocation is hardened the same way, because each flag turns off a way a repository can run code or reach
the network when someone merely *reads* it:

* `-c core.hooksPath=/dev/null` — hooks are the repository author's shell scripts;
* `-c core.symlinks=false` and a mode check — a symlink in a tree points outside what the repository contains;
* `-c protocol.*.allow=never`, `GIT_TERMINAL_PROMPT=0`, no submodule and no LFS — reading a local history is not
  a reason to open a socket or ask for a password;
* a scrubbed environment and `GIT_CONFIG_NOSYSTEM`, so a global `~/.gitconfig` alias cannot redefine what these
  commands do.

Commits become versions labelled by their short hash, oldest first — the order P1 replays them in.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acis.core.errors import InvalidInput
from acis.core.types import Limits
from acis.obs.log import get_logger

if TYPE_CHECKING:  # a cycle at runtime, a type at check time
    from acis.ingest.sources import VersionUnits

log = get_logger("acis.ingest")

LABEL_CHARS = 12
DEFAULT_MAX_COMMITS = 200
GIT_TIMEOUT = 120.0

#: Flags that make `git` a reader rather than a runtime. Every call carries them.
HARDENING = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.symlinks=false",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "protocol.allow=never",
    "-c",
    "uploadpack.allowFilter=false",
    "-c",
    "advice.detachedHead=false",
)
#: A scrubbed environment: no credential helpers, no prompts, no system or global configuration.
ENV = {
    "PATH": "/usr/bin:/bin:/usr/local/bin",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "/bin/false",
    "GIT_ALLOW_PROTOCOL": "",
    "LC_ALL": "C",
    "HOME": "/nonexistent",
}


def _run(repo: Path, args: list[str], *, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(  # noqa: S603 — fixed argv, never a shell, scrubbed environment
            ["git", *HARDENING, "-C", str(repo), *args],
            capture_output=True,
            check=True,
            timeout=GIT_TIMEOUT,
            env=dict(ENV),
            text=not binary,
        )
    except FileNotFoundError as exc:
        raise InvalidInput("git is not installed, so a git source cannot be read") from exc
    except subprocess.TimeoutExpired as exc:
        raise InvalidInput("git did not finish in time", command=args[0]) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")).strip()
        raise InvalidInput(f"git {args[0]} failed", detail=detail[:200]) from exc


@dataclass(slots=True)
class GitSource:
    """Commits of a local repository as versions of units. Read-only, offline, no checkout."""

    root: Path
    limits: Limits = field(default_factory=Limits)
    extensions: tuple[str, ...] = (".py",)
    rev: str = ""
    max_commits: int = DEFAULT_MAX_COMMITS
    sid: str = ""

    def __post_init__(self) -> None:
        if not (self.root / ".git").exists() and not (self.root / "HEAD").is_file():
            raise InvalidInput("not a git repository", location=str(self.root))

    def versions(self) -> Iterator[VersionUnits]:  # noqa: F821 — imported lazily to avoid a cycle
        from acis.ingest.sources import _assemble, _Budget  # noqa: PLC0415

        budget = _Budget(self.limits)
        for commit in self._commits():
            pairs = self._tree(commit)
            if not pairs:
                continue
            yield _assemble(commit[:LABEL_CHARS], pairs, budget=budget, sid=self.sid)

    def _commits(self) -> list[str]:
        """Oldest first. A named revision is exactly one version; otherwise the last `max_commits` of history."""
        if self.rev:
            try:
                resolved = _run(self.root, ["rev-parse", "--verify", f"{self.rev}^{{commit}}"]).stdout.strip()
            except InvalidInput as exc:
                # `rev-parse` failing means this repository has no such revision. Say that, rather than passing
                # git's own wording up: an unresolvable selector is a defined answer, not an internal error.
                raise InvalidInput(f"unknown revision {self.rev!r} in this repository", rev=self.rev) from exc
            if not resolved:
                raise InvalidInput(f"unknown revision {self.rev!r} in this repository", rev=self.rev)
            return [resolved]
        out = _run(self.root, ["rev-list", f"--max-count={max(1, self.max_commits)}", "--all"]).stdout
        return list(reversed([line.strip() for line in out.splitlines() if line.strip()]))

    def _tree(self, commit: str) -> list[tuple[str, str]]:
        """`ls-tree -r -z`: NUL-separated, so a filename containing a newline cannot forge a record."""
        listing = _run(self.root, ["ls-tree", "-r", "-z", "--full-tree", commit]).stdout
        wanted: list[tuple[str, str]] = []
        for record in listing.split("\0"):
            if not record:
                continue
            meta, _, path = record.partition("\t")
            parts = meta.split()
            if len(parts) < 3:
                continue
            mode, kind, oid = parts[0], parts[1], parts[2]
            # 120000 is a symlink and 160000 a submodule (gitlink): both point at something this repository does
            # not contain, so neither is a unit.
            if kind != "blob" or mode in ("120000", "160000"):
                continue
            if not any(path.endswith(ext) for ext in self.extensions):
                continue
            wanted.append((path, oid))

        return [(path, self._blob(oid)) for path, oid in wanted]

    def _blob(self, oid: str) -> str:
        raw = _run(self.root, ["cat-file", "blob", oid], binary=True).stdout
        return bytes(raw).decode("utf-8", errors="replace")


__all__ = ["DEFAULT_MAX_COMMITS", "ENV", "HARDENING", "LABEL_CHARS", "GitSource"]
