"""One place that turns configuration into an encoder (docs/spec/02 §3, D4, D15, D16).

Mode A, Mode B, the dev harness and the G-M bake-off must all run the *same* model over the *same* cache, or the
numbers they produce describe different systems while carrying one name. So there is exactly one constructor, and
it reads the frozen config rather than being told:

    model: {encoder: <card key> | hashing, dim: …, numeric_profile: cpu-fp32}

Weights and vectors live under `ACIS_HOME`, deliberately outside the repository (CLAUDE.md §7): worktrees share
one copy, and the working-tree seal scan stays free of gigabytes that are not ours.

`hashing` is the model-free stand-in that validated the harness in Phase 1. It reports `submission_capable =
False`, so a strict run refuses it; it is not a fallback for a missing model, and nothing here will substitute it
for one. A real model whose weights are not on disk raises `NotReady` naming the step that fetches them.
"""

from __future__ import annotations

from pathlib import Path

from acis.core.config import FrozenConfig
from acis.core.errors import NotReady
from acis.core.hashing import hash_obj, short
from acis.core.numeric import REFERENCE_PROFILE
from acis.core.paths import acis_home
from acis.embed.base import Encoder
from acis.embed.cache import VectorCache
from acis.embed.registry import available_cards, load_card
from acis.obs.log import get_logger

log = get_logger("acis.embed")

STAND_IN = "hashing"


def encoder_name(config: FrozenConfig) -> str:
    return str(config.get("model.encoder", STAND_IN))


def profile_of(config: FrozenConfig) -> str:
    return str(config.get("model.numeric_profile", REFERENCE_PROFILE))


def prep_hash(config: FrozenConfig) -> str:
    """Identity of the preprocessing a vector was produced under (D3).

    Truncating at 512 tokens instead of 1024 changes what the model reads, so it changes the vector — and a cache
    that ignored it would serve the old one for the rest of the sweep.
    """
    return short(hash_obj(config.as_dict().get("prep", {})), 12)


def model_dir(key: str) -> Path:
    """Where pinned weights live: one directory per card key under `ACIS_HOME`, shared by every worktree."""
    return acis_home() / "models" / key


def vector_cache_dir(key: str) -> Path:
    return acis_home() / "cache" / "vectors" / key


def build_encoder(config: FrozenConfig, *, cache: bool = True) -> Encoder:
    """The encoder this configuration names. Never silently substitutes one model for another."""
    name = encoder_name(config)
    if name == STAND_IN:
        from acis.embed.hashing import HashingEncoder  # noqa: PLC0415

        return HashingEncoder(dim=int(config.get("model.dim", 4096)))

    known = available_cards()
    if name not in known:
        raise NotReady(
            f"no model card for {name!r}; cards live in configs/models/ (known: {', '.join(known) or 'none'})"
        )
    card = load_card(name)
    directory = model_dir(name)
    if not directory.is_dir() or not any(directory.glob("*.safetensors")):
        raise NotReady(
            f"no weights for {name} at {directory}: they are fetched once, offline-capable, by the owner "
            f"(G0.4 pins them). Until then the dense channel cannot run with {card.name}.",
        )

    from acis.embed.runtime import load_runtime  # noqa: PLC0415 — torch is imported only when weights exist

    store = VectorCache.open(vector_cache_dir(name)) if cache else None
    runtime = load_runtime(card, directory, profile=profile_of(config), cache=store, prep_hash=prep_hash(config))
    log.info("encoder.built", model=card.name, profile=profile_of(config), cached=cache)
    return runtime


__all__ = [
    "STAND_IN",
    "build_encoder",
    "encoder_name",
    "model_dir",
    "prep_hash",
    "profile_of",
    "vector_cache_dir",
]
