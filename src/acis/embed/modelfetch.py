"""Fetching and pinning model weights — the model supply chain (D4, D15, G0.4).

With `acis fetch` this is the only code that touches the network, and it is the easier of the two to get
dangerously wrong, so the rules are narrow:

* **Safetensors only.** A `.bin` checkpoint is a pickle, and loading one executes whatever is inside it. The
  defence is not to load it carefully; it is not to have the file (CLAUDE.md §4).
* **A commit, never a tag.** A moving revision can change under a pinned run. The commit is resolved before
  anything downloads and recorded with the per-file hashes, so what was fetched can be named afterwards and
  re-verified offline for the life of the project.
* **Licence up front.** A non-permissively licensed model may be *measured* for the gap the permissive-only rule
  costs us, never shipped (D4). Fetching one therefore has to be asked for explicitly, and what comes back says
  so — otherwise the distinction survives only in somebody's memory.

`--pin` writes the commit and the file hashes back into `configs/models/<key>.yaml`: that is G0.4's pinning step,
and from then on `verify_pinned_files` refuses to encode with weights that do not match.

The hub is injected so every path here is tested without a network.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from acis.core.errors import InvalidInput, NotReady
from acis.core.hashing import sha256_file
from acis.embed.factory import model_dir
from acis.embed.registry import card_path, licence_is_permissive, load_card, verify_pinned_files
from acis.obs.log import get_logger

log = get_logger("acis.modelfetch")

#: What a model *is*: weights, tokeniser, configuration. Nothing else is downloaded, whatever the repo contains.
ALLOW = ("*.safetensors", "*.safetensors.index.json", "*.json", "*.txt", "*.model", "*.tiktoken")
#: Formats that execute on load, or that we simply have no use for. Listed to explain the refusal, not to gate it.
PICKLE_FORMATS = ("*.bin", "*.pt", "*.pth", "*.pkl", "*.ckpt", "*.h5", "*.msgpack")


@dataclass(frozen=True, slots=True)
class FetchedModel:
    """What landed on disk, and what it is: enough to pin the card and to re-verify offline later."""

    key: str
    repo: str
    commit: str
    model_dir: Path
    files: tuple[str, ...]
    sha256: Mapping[str, str]
    total_mb: float
    reference_only: bool = False


def plan_files(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split a repository listing into what we download and what we deliberately leave behind."""
    wanted, skipped = [], []
    for name in sorted(names):
        base = name.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(base, pattern) for pattern in ALLOW):
            wanted.append(name)
        else:
            skipped.append(name)
    return wanted, skipped


def _hub(injected: Any = None) -> Any:
    if injected is not None:
        return injected
    os.environ["HF_HUB_OFFLINE"] = "0"  # this function is the network path; everything else stays offline (D15)
    import huggingface_hub  # noqa: PLC0415

    return huggingface_hub


def fetch_model(
    key: str,
    *,
    hub: Any = None,
    reference: bool = False,
    force: bool = False,
) -> FetchedModel:
    """Download one carded model into `ACIS_HOME/models/<key>`, hashing everything that lands."""
    card = load_card(key)
    if not licence_is_permissive(card) and not reference:
        raise InvalidInput(
            f"{card.name} is licensed {card.licence!r}, which may be measured for reference but never shipped "
            "(D4). Pass --reference to fetch it deliberately.",
            model=card.name,
        )

    client = _hub(hub)
    listing = list(client.list_repo_files(card.name, revision=card.base_commit))
    wanted, skipped = plan_files(listing)
    if not any(name.endswith(".safetensors") for name in wanted):
        raise NotReady(
            f"{card.name} publishes no safetensors weights; a pickle checkpoint is not an acceptable substitute "
            "(CLAUDE.md §4)",
            skipped=[n for n in skipped if any(fnmatch.fnmatch(n, p) for p in PICKLE_FORMATS)][:5],
        )

    # A tag can move; a commit cannot. Resolve once, before anything downloads, and use it for every file.
    commit = card.base_commit or str(getattr(client.model_info(card.name, revision=card.base_commit), "sha", ""))
    if not commit:
        raise NotReady(f"could not resolve a commit for {card.name}; refusing to fetch a moving revision")

    directory = model_dir(key)
    directory.mkdir(parents=True, exist_ok=True)
    log.info("modelfetch.plan", model=card.name, commit=commit[:12], files=len(wanted), skipped=len(skipped))

    digests: dict[str, str] = {}
    total = 0
    for name in wanted:
        local = Path(
            client.hf_hub_download(
                repo_id=card.name,
                filename=name,
                revision=commit,
                local_dir=str(directory),
                force_download=force,
            )
        )
        digests[name] = sha256_file(local)
        total += local.stat().st_size

    return FetchedModel(
        key=key,
        repo=card.name,
        commit=commit,
        model_dir=directory,
        files=tuple(wanted),
        sha256=digests,
        total_mb=round(total / (1024.0 * 1024.0), 2),
        reference_only=not licence_is_permissive(card),
    )


def pin_card(fetched: FetchedModel, *, path: str | Path | None = None) -> Path:
    """Write the commit and the file hashes into the card (G0.4). A different pin is a decision, not an edit."""
    target = Path(path) if path else card_path(fetched.key)
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    existing = raw.get("base_commit")
    if existing and str(existing) != fetched.commit:
        raise InvalidInput(
            f"{fetched.key} is already pinned to {str(existing)[:12]}; changing a pin needs an ADR",
            pinned=str(existing),
            fetched=fetched.commit,
        )
    raw["base_commit"] = fetched.commit
    raw["file_sha256"] = dict(sorted(fetched.sha256.items()))
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    log.info("modelfetch.pinned", model=fetched.repo, commit=fetched.commit[:12], files=len(fetched.sha256))
    return target


def verify_model(key: str) -> list[str]:
    """Re-hash what is on disk against the card's pins. Offline; returns the problems, empty when as pinned."""
    directory = model_dir(key)
    if not directory.is_dir() or not any(directory.glob("*.safetensors")):
        raise NotReady(f"no weights for {key} at {directory}: fetch them first (`acis fetch --models {key}`)")
    return verify_pinned_files(load_card(key), directory)


__all__ = ["ALLOW", "PICKLE_FORMATS", "FetchedModel", "fetch_model", "pin_card", "plan_files", "verify_model"]
