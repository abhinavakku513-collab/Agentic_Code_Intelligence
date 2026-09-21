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
    """Mutable data root (outside git): data, CAS, caches. `$ACIS_HOME`, else `<root>/.acis-home`."""
    env = os.environ.get("ACIS_HOME")
    home = Path(env).resolve() if env else acis_root() / ".acis-home"
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


def sealed_root() -> Path:
    """Physical TEST seal (D19). Read only by `acis.eval.final` / `acis eval official`."""
    return Path(os.environ.get("ACIS_SEALED_HOME", str(Path.home() / ".acis-sealed"))).resolve()


__all__ = ["acis_home", "acis_root", "home_path", "repo_path", "sealed_root"]
