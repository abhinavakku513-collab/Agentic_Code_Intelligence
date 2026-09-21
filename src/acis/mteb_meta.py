"""ModelMeta identity for the mteb adapter (D16, docs/spec/03 §2.5).

Why this matters more than it looks: `mteb.evaluate()` defaults to a persistent result cache with
`overwrite_strategy="only-missing"` (docs/spec/01 V-04), and the cache key is the model's *identity*. A model whose
name and revision never change can therefore return a **stale result from a previous run** while looking perfectly
healthy. So the revision is `<git12>+<config12>`: it moves whenever the code or any ranking-relevant config key
moves, and the official script additionally passes `cache=None, overwrite_strategy="always"`.

`ModelMeta.create_empty(overwrites=…)` exists in mteb 2.21.0 but not in every 2.x, so an introspecting fallback fills
the required fields from the model's own signature rather than from a hard-coded list.
"""

from __future__ import annotations

import subprocess
from typing import Any

from acis.core.config import FrozenConfig
from acis.core.hashing import short
from acis.core.paths import acis_root

MODEL_NAME = "acis/acis-apps"
UNKNOWN_GIT = "nogit0000000"


def git_sha12() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=acis_root(), capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_GIT
    return short(sha, 12) if sha else UNKNOWN_GIT


def revision_for(config: FrozenConfig) -> str:
    """`<git12>+<config12>` — the identity that makes a stale cache hit impossible (D16)."""
    return f"{git_sha12()}+{config.config12}"


def _overwrites(config: FrozenConfig) -> dict[str, Any]:
    # `similarity_fn_name` must be the enum member: mteb compares it with `is`, so the bare string "cosine"
    # silently falls through to "Similarity function not specified" in Mode B.
    from mteb.models.model_meta import ScoringFunction  # noqa: PLC0415

    return {
        "name": MODEL_NAME,
        "revision": revision_for(config),
        "release_date": "2026-09-21",
        "languages": ["eng-Latn", "python-Code"],
        "n_parameters": None,
        "memory_usage_mb": None,
        "max_tokens": float(config.get("prep.doc.max_tokens", 1024)),
        "embed_dim": None,
        "license": None,
        "open_weights": True,
        "public_training_code": None,
        "public_training_data": None,
        "framework": [],
        "reference": None,
        "similarity_fn_name": ScoringFunction.COSINE,
        "use_instructions": True,
        "training_datasets": None,
    }


def make_model_meta(config: FrozenConfig) -> Any:
    """Build a real `ModelMeta`, preferring `create_empty` and falling back to signature introspection."""
    from mteb.models.model_meta import ModelMeta  # noqa: PLC0415

    overwrites = _overwrites(config)
    create_empty = getattr(ModelMeta, "create_empty", None)
    if callable(create_empty):
        try:
            return create_empty(overwrites=overwrites)
        except TypeError:  # an older signature: fall through to introspection
            pass

    fields = getattr(ModelMeta, "model_fields", {})
    kwargs: dict[str, Any] = {}
    for field_name, info in fields.items():
        if field_name in overwrites:
            kwargs[field_name] = overwrites[field_name]
        elif getattr(info, "is_required", lambda: False)():
            kwargs[field_name] = [] if str(getattr(info, "annotation", "")).startswith("list") else None
    kwargs.setdefault("name", MODEL_NAME)
    kwargs.setdefault("revision", overwrites["revision"])
    return ModelMeta(**kwargs)


__all__ = ["MODEL_NAME", "git_sha12", "make_model_meta", "revision_for"]
