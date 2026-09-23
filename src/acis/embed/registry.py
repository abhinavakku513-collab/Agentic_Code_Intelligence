"""Model cards, pinning and instruction formats (docs/spec/02 §3, D4, D15).

Three rules this module exists to enforce:

* **Instruction strings, pooling and truncation come from `configs/models/<name>.yaml`, never from code.** A
  Qwen3-style model wants `Instruct: {task}\\nQuery: {query}` and last-token pooling; the next candidate will want
  something else, and the bake-off compares candidates only if each is used as its own card says.
* **A model is pinned by commit SHA and per-file SHA-256.** `HF_HUB_OFFLINE=1` means the weights that load are
  whatever is on disk, so the identity has to be checked rather than assumed (D15).
* **Per-route instructions (INV-15).** Only the `statement_like` route uses the task string G1 tuned on APPS
  statements; every other query takes the generic one, or none at all when the model has no instruction format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from acis.core.errors import InvalidInput
from acis.core.hashing import hash_obj, sha256_file
from acis.core.paths import repo_path

MODEL_CONFIG_DIR = ("configs", "models")
GENERIC_TASK = "T3"
STATEMENT_TASK = "T1"
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Everything the runtime needs to use a model the way its authors intended."""

    key: str
    name: str
    base_commit: str | None
    licence: str
    pooling: str
    normalize: bool
    padding_side: str
    query_template: str
    document_template: str
    tasks: Mapping[str, str]
    max_tokens: int
    trust_remote_code: bool = False
    file_sha256: Mapping[str, str] = field(default_factory=dict)
    #: Which task key each route uses, when gate G1 has decided one. Empty means the D4 default mapping below.
    route_tasks: Mapping[str, str] = field(default_factory=dict)

    # -- instruction formatting ---------------------------------------------------------------------------------
    def task_key_for(self, route: str) -> str:
        """The task *key* a route uses: G1's choice when there is one, else the D4 default (INV-15).

        The default is deliberately conservative: only `statement_like` gets the APPS-tuned instruction, and
        everything else — including any route we have never seen — gets the generic one.
        """
        chosen = self.route_tasks.get(route)
        if chosen:
            return str(chosen)
        return STATEMENT_TASK if route == "statement_like" else GENERIC_TASK

    def task_string(self, route: str) -> str:
        """The task description for a route. Unknown routes fall back to the generic one (INV-15)."""
        task = self.tasks.get(self.task_key_for(route)) or self.tasks.get(GENERIC_TASK) or ""
        return str(task)

    def with_route_task(self, route: str, task_key: str) -> ModelCard:
        """A copy that encodes `route` with `task_key` — how the G1 sweep varies the instruction.

        The task *strings* still come from the card (D4); the sweep only chooses which of them a route uses.
        """
        if task_key not in self.tasks:
            raise InvalidInput(
                f"model card {self.key!r} has no task {task_key!r}", known=sorted(self.tasks), route=route
            )
        return replace(self, route_tasks={**dict(self.route_tasks), route: task_key})

    def format_query(self, text: str, *, route: str = "generic") -> str:
        """Render a query in the model's documented input format.

        A model with no instruction format gets the bare text: inventing a prefix for a model that was not
        trained with one is a silent accuracy change, not a neutral default.
        """
        if not self.query_template:
            return text
        task = self.task_string(route)
        if "{task}" in self.query_template and not task:
            return text
        return self.query_template.format(task=task, query=text)

    def format_document(self, text: str) -> str:
        return self.document_template.format(text=text) if self.document_template else text

    @property
    def is_pinned(self) -> bool:
        return bool(self.base_commit)

    @property
    def fingerprint(self) -> str:
        """Identity of the *card*. The runtime combines it with the weights digest (docs/spec/02 §3)."""
        return hash_obj(
            {
                "name": self.name,
                "commit": self.base_commit,
                "pooling": self.pooling,
                "normalize": self.normalize,
                "padding_side": self.padding_side,
                "query_template": self.query_template,
                "document_template": self.document_template,
                "tasks": dict(self.tasks),
                "route_tasks": dict(self.route_tasks),
                "max_tokens": self.max_tokens,
            }
        )


def card_path(key: str) -> Path:
    return repo_path(*MODEL_CONFIG_DIR, f"{key}.yaml")


def available_cards() -> list[str]:
    directory = repo_path(*MODEL_CONFIG_DIR)
    return sorted(p.stem for p in directory.glob("*.yaml")) if directory.is_dir() else []


def load_card(key: str) -> ModelCard:
    """Read `configs/models/<key>.yaml`. Raises rather than guessing any field the runtime depends on."""
    path = card_path(key)
    if not path.is_file():
        raise InvalidInput(f"no model card for {key!r}", looked_in=str(path), known=available_cards())
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise InvalidInput(f"model card must be a mapping: {path}")

    missing = [k for k in ("name", "pooling", "normalize") if k not in raw]
    if missing:
        raise InvalidInput(f"model card {key!r} is missing required fields", missing=missing)
    if raw.get("trust_remote_code"):
        raise InvalidInput(
            f"model card {key!r} requests trust_remote_code, which is forbidden (CLAUDE.md §4)", card=str(path)
        )

    return ModelCard(
        key=key,
        name=str(raw["name"]),
        base_commit=(str(raw["base_commit"]) if raw.get("base_commit") else None),
        licence=str(raw.get("licence", "unknown")),
        pooling=str(raw["pooling"]),
        normalize=bool(raw["normalize"]),
        padding_side=str(raw.get("padding_side", "right")),
        query_template=str(raw.get("query_template", "")),
        document_template=str(raw.get("document_template", "{text}")),
        tasks={str(k): str(v) for k, v in (raw.get("tasks") or {}).items()},
        max_tokens=int(raw.get("max_tokens", DEFAULT_MAX_TOKENS)),
        trust_remote_code=False,
        file_sha256={str(k): str(v) for k, v in (raw.get("file_sha256") or {}).items()},
        route_tasks={str(k): str(v) for k, v in (raw.get("route_tasks") or {}).items()},
    )


# -- pinning ---------------------------------------------------------------------------------------------------
PERMISSIVE_LICENCES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd"})


def licence_is_permissive(card: ModelCard) -> bool:
    """D4: the shipped encoder needs a permissive licence. Others may be *measured* as reference only."""
    return card.licence.strip().lower() in PERMISSIVE_LICENCES


def verify_pinned_files(card: ModelCard, model_dir: str | Path) -> list[str]:
    """Re-hash the weights on disk against the card. Returns the list of problems (empty == as pinned).

    With `HF_HUB_OFFLINE=1` whatever sits in the cache is what loads, so this is the only thing standing between a
    pinned model and a different one with the same name.
    """
    if not card.file_sha256:
        return ["the card pins no file checksums (G0.4 records them)"]
    directory = Path(model_dir)
    problems: list[str] = []
    for name, expected in sorted(card.file_sha256.items()):
        target = directory / name
        if not target.is_file():
            problems.append(f"missing: {name}")
        elif sha256_file(target) != expected:
            problems.append(f"checksum mismatch: {name}")
    return problems


def assert_shippable(card: ModelCard) -> None:
    """The envelope a *submitted* encoder must satisfy (D4). Reference-only candidates skip this deliberately."""
    if not card.is_pinned:
        raise InvalidInput(f"model {card.key!r} is not pinned to a commit (G0.4 does this)", name=card.name)
    if card.trust_remote_code:
        raise InvalidInput(f"model {card.key!r} requires remote code", name=card.name)
    if not licence_is_permissive(card):
        raise InvalidInput(
            f"model {card.key!r} has a non-permissive licence, so it may be measured but not shipped (D4)",
            licence=card.licence,
        )


def describe(card: ModelCard) -> dict[str, Any]:
    """The row a scorecard and a ledger entry both want."""
    return {
        "key": card.key,
        "name": card.name,
        "base_commit": card.base_commit,
        "licence": card.licence,
        "permissive": licence_is_permissive(card),
        "pinned": card.is_pinned,
        "pooling": card.pooling,
        "max_tokens": card.max_tokens,
        "card_fingerprint": card.fingerprint,
    }


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "GENERIC_TASK",
    "MODEL_CONFIG_DIR",
    "PERMISSIVE_LICENCES",
    "STATEMENT_TASK",
    "ModelCard",
    "assert_shippable",
    "available_cards",
    "card_path",
    "describe",
    "licence_is_permissive",
    "load_card",
    "verify_pinned_files",
]
