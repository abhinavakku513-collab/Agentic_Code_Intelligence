"""Repo-root and ACIS_HOME resolution (D16: paths resolve from the repo root, never the CWD)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def acis_root() -> Path:
    """Repository root: `$ACIS_ROOT` if set, else the nearest ancestor holding `pyproject.toml`."""
    env = os.environ.get("ACIS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[3]


def acis_home() -> Path:
    """Mutable data root: dataset assets, CAS, caches. `$ACIS_HOME`, else `~/.acis/home`.

    Deliberately **outside the repository** (CLAUDE.md §7): worktrees share one copy instead of each carrying
    gigabytes, and a cache inside the tree would keep tripping the working-tree seal check for files that are not
    ours to begin with.
    """
    env = os.environ.get("ACIS_HOME")
    home = Path(env).resolve() if env else Path.home() / ".acis" / "home"
    home.mkdir(parents=True, exist_ok=True)
    return home


def repo_path(*parts: str) -> Path:
    """A path inside the repository, resolved from the repo root."""
    return acis_root().joinpath(*parts)


def home_path(*parts: str, create_parent: bool = False) -> Path:
    p = acis_home().joinpath(*parts)
    if create_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def reg_home() -> Path:
    """Cache root for third-party REG benchmarks — deliberately **outside** `ACIS_HOME`.

    REG tasks (CosQA, StackOverflowQA, CodeSearchNet, …) ship their own `qrels/test-*` files. Those are other
    benchmarks' labels, not ours, but they match the sealed-name patterns, and the seal check scans `ACIS_HOME`
    with deliberately broad patterns. Keeping them apart means the check can stay broad without crying wolf, and
    `ACIS_HOME` holds only the allow-listed APPS assets.
    """
    env = os.environ.get("ACIS_REG_HOME")
    home = Path(env).resolve() if env else Path.home() / ".acis" / "reg"
    home.mkdir(parents=True, exist_ok=True)
    return home


def sealed_root() -> Path:
    """Physical TEST seal (D19). Read only by `acis.eval.final` / `acis eval official`."""
    return Path(os.environ.get("ACIS_SEALED_HOME", str(Path.home() / ".acis-sealed"))).resolve()


__all__ = ["acis_home", "acis_root", "home_path", "reg_home", "repo_path", "sealed_root"]
