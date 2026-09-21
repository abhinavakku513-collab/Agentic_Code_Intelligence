"""Frozen, hashed configuration (docs/spec/06 §4).

One rule decides everything here: **ranking-relevant keys change `config_hash`; operational knobs do not.** The hash
goes into the ModelMeta revision (D16), every cache key (INV-2), every ledger row and every manifest, so it must be
stable across machines and insensitive to how many threads a judge happens to have.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from acis.core.errors import InvalidInput
from acis.core.hashing import hash_obj, short
from acis.core.paths import acis_root

# Keys that describe *how fast* we run, never *what we return*. Excluded from the hash at any depth.
OPERATIONAL_KEYS = frozenset(
    {
        "threads",
        "num_proc",
        "workers",
        "batch_size",
        "host",
        "port",
        "log_level",
        "out",
        "cold_official",
        "test_touch_budget",
        "progress",
        "cache_dir",
        "device_map",
    }
)
# Sections that only record provenance (which ledger row decided what).
OPERATIONAL_SECTIONS = frozenset({"gates", "notes", "meta"})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _ranking_relevant(value: Any) -> Any:
    """Project a config onto its ranking-relevant part.

    A section that holds nothing but operational knobs collapses to `{}` and is then dropped entirely: otherwise
    `run: {threads: 8}` and no `run:` section at all would hash differently, and setting a thread count would move
    the model revision.
    """
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if str(key) in OPERATIONAL_KEYS:
                continue
            projected = _ranking_relevant(item)
            if isinstance(projected, dict) and not projected and isinstance(item, Mapping):
                continue  # the whole sub-section was operational
            out[str(key)] = projected
        return out
    if isinstance(value, (list, tuple)):
        return [_ranking_relevant(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """An immutable configuration tree plus its identity."""

    raw: Mapping[str, Any]
    source_path: str
    config_hash: str

    # -- access ---------------------------------------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: object) -> bool:
        return key in self.raw

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def get(self, dotted: str, default: Any = None) -> Any:
        """`cfg.get("prep.query.max_tokens", 1024)` — dotted lookup with a default, never a KeyError."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise InvalidInput(f"configuration key {dotted!r} is required", config=self.source_path)
        return value

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, Mapping) else MappingProxyType({})

    def as_dict(self) -> dict[str, Any]:
        thawed: dict[str, Any] = _thaw(self.raw)
        return thawed

    def with_overrides(self, **dotted_values: Any) -> FrozenConfig:
        """Return a new config with dotted keys replaced (used by sweeps and tests, never by the official run)."""
        data = self.as_dict()
        for dotted, value in dotted_values.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return freeze_config(data, source_path=self.source_path)

    # -- identity -------------------------------------------------------------------------------------------------
    @property
    def config12(self) -> str:
        return short(self.config_hash, 12)

    @property
    def strict(self) -> bool:
        return bool(self.get("run.strict", False))

    @property
    def mode(self) -> str:
        return str(self.get("run.mode", "A")).upper()

    @property
    def numeric_profile(self) -> str:
        return str(self.get("model.numeric_profile", "cpu-fp32"))

    @property
    def seed(self) -> int:
        return int(self.get("run.seed", 0))


def compute_config_hash(data: Mapping[str, Any]) -> str:
    """Hash of the ranking-relevant projection of `data` (docs/spec/06 §4)."""
    relevant: dict[str, Any] = {}
    for key, value in data.items():
        if key in OPERATIONAL_SECTIONS:
            continue
        projected = _ranking_relevant(value)
        if isinstance(projected, dict) and not projected and isinstance(value, Mapping):
            continue
        relevant[str(key)] = projected
    return hash_obj(relevant)


def freeze_config(data: Mapping[str, Any], *, source_path: str = "<memory>") -> FrozenConfig:
    plain = _thaw(data)
    if not isinstance(plain, dict):
        raise InvalidInput("configuration must be a mapping", source=source_path)
    return FrozenConfig(raw=_freeze(plain), source_path=source_path, config_hash=compute_config_hash(plain))


def resolve_config_path(path: str | Path | None) -> Path:
    """Resolve a config path **from the repo root** (D16), never the CWD — a judge may run from anywhere."""
    if path is None:
        path = os.environ.get("ACIS_CONFIG", "configs/official.yaml")
    p = Path(path)
    return p if p.is_absolute() else (acis_root() / p)


def load_frozen_config(path: str | Path | None = None) -> FrozenConfig:
    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise InvalidInput(f"configuration file not found: {resolved}")
    with open(resolved, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise InvalidInput(f"configuration root must be a mapping: {resolved}")
    return freeze_config(
        data, source_path=str(resolved.relative_to(acis_root()) if resolved.is_relative_to(acis_root()) else resolved)
    )


__all__ = [
    "OPERATIONAL_KEYS",
    "OPERATIONAL_SECTIONS",
    "FrozenConfig",
    "compute_config_hash",
    "freeze_config",
    "load_frozen_config",
    "resolve_config_path",
]
