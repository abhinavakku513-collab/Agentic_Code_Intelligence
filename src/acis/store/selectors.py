"""Version selectors: naming which version a request is about (docs/spec/04 §4, INV-2).

    latest · version:<label> · <label> · snapshot:<id> · commit:<sha|prefix> · tag:<name> · as_of:<ISO-8601>
    all · range:<a>..<b>

The rule is that a selector is resolved **once**, at the start of a request, and either yields snapshots or
fails. It never guesses, and the three ways it can fail are three different instructions to the caller:

* `NotFound` — there is no such version. Nothing to do but ask for a different one.
* `IndexRequired` — the version exists and has never been built. Build it.
* `VersionConflict` — an `as_of` instant that two versions answer equally well. Both candidates are named, and
  the caller decides; picking one for them would be a coin flip presented as an answer.

Resolving once is what keeps INV-2 true: every channel in a request sees the same snapshot, so results cannot mix
versions even when a build activates midway through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from acis.core.errors import IndexRequired, InvalidInput, NotFound, VersionConflict
from acis.store import catalog, snapshots

KINDS = ("latest", "version", "snapshot", "commit", "tag", "as_of", "all", "range")
RANGE = re.compile(r"^range:(?P<a>[^.]+)\.\.(?P<b>.+)$")


@dataclass(frozen=True, slots=True)
class Resolution:
    """What a selector resolved to. `snapshot_ids` and `labels` are parallel and in history order."""

    selector: str
    kind: str
    snapshot_ids: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def single(self) -> bool:
        return len(self.snapshot_ids) == 1

    @property
    def snapshot_id(self) -> str:
        if not self.single:
            raise InvalidInput(
                "this selector names more than one version; use the evolution surface or pin one version",
                selector=self.selector,
                versions=len(self.snapshot_ids),
            )
        return self.snapshot_ids[0]


def to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _parse_instant(text: str) -> float:
    try:
        parsed = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise InvalidInput("as_of takes an ISO-8601 instant, e.g. as_of:2026-09-23T10:00:00Z", got=text[:40]) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _built(rows: list[Any]) -> list[Any]:
    return [r for r in rows if r["snapshot_id"]]


def resolve(repo_id: str, selector: str | None = None) -> Resolution:
    """Resolve `selector` against a repository's recorded versions."""
    text = (selector or "latest").strip()
    if not text:
        raise InvalidInput("an empty selector names nothing; use `latest` or a version label")

    with catalog.open_catalog() as db:
        known = catalog.get_repo_exists(db, repo_id)
        rows = catalog.list_versions(db, repo_id)
    if not known:
        raise NotFound(f"no repository {repo_id!r}")

    built = _built(rows)
    by_label = {str(r["label"]): r for r in rows}

    if text in ("latest", "version:latest"):
        if not built:
            raise IndexRequired(f"{repo_id} has no built version yet; run `acis index`")
        # `latest` is whatever `refs/ACTIVE` points at, not simply the newest build: after a rollback those are
        # two different snapshots, and the one that is *serving* is the honest answer (spec 04 §3).
        active = snapshots.active_snapshot_id(repo_id)
        chosen = next((r for r in built if str(r["snapshot_id"]) == active), None) or built[-1]
        return Resolution(text, "latest", (str(chosen["snapshot_id"]),), (str(chosen["label"]),))

    if text == "all":
        if not built:
            raise IndexRequired(f"{repo_id} has no built version yet; run `acis index`")
        return Resolution(
            text, "all", tuple(str(r["snapshot_id"]) for r in built), tuple(str(r["label"]) for r in built)
        )

    if text.startswith("snapshot:"):
        snapshot_id = text.split(":", 1)[1].strip()
        match = [r for r in rows if str(r["snapshot_id"]) == snapshot_id]
        if not match:
            raise NotFound(f"no snapshot {snapshot_id!r} in {repo_id}")
        return Resolution(text, "snapshot", (snapshot_id,), (str(match[0]["label"]),))

    if text.startswith("as_of:"):
        return _resolve_as_of(repo_id, text, built)

    if text.startswith("range:"):
        return _resolve_range(repo_id, text, rows, built)

    # `version:`, `commit:`, `tag:` and a bare label all name one version. Commits and tags *are* labels here:
    # the git source labels a version by its short commit hash, and a tag is recorded under its own name.
    label = text.split(":", 1)[1].strip() if ":" in text and text.split(":", 1)[0] in KINDS else text
    if ":" in text and text.split(":", 1)[0] not in KINDS:
        raise InvalidInput(
            f"unknown selector {text!r}; supported: latest, all, version:<label>, snapshot:<id>, commit:<sha>, "
            "tag:<name>, as_of:<ISO-8601>, range:<a>..<b>",
        )
    row = by_label.get(label) or _by_commit_prefix(rows, label)
    if row is None:
        raise NotFound(f"no version {label!r} in {repo_id}")
    if not row["snapshot_id"]:
        raise IndexRequired(f"version {label!r} of {repo_id} exists but has not been built; run `acis index`")
    return Resolution(text, "version", (str(row["snapshot_id"]),), (str(row["label"]),))


def _by_commit_prefix(rows: list[Any], label: str) -> Any | None:
    """A commit selector may be a prefix. An ambiguous prefix is a conflict, never the first match."""
    if len(label) < 4:
        return None
    matches = [r for r in rows if str(r["label"]).startswith(label)]
    if len(matches) > 1:
        raise VersionConflict(
            "this commit prefix matches more than one version",
            prefix=label,
            candidates=[str(r["label"]) for r in matches[:5]],
        )
    return matches[0] if matches else None


#: ISO-8601 carries microseconds, while a timestamp is a float with more precision than that. Comparing them
#: exactly makes `as_of:<the instant a version was created>` answer "the version before it" for some values and
#: not others — the same query giving different answers on different days. One microsecond of tolerance is the
#: resolution of the input format, so an instant includes the microsecond it names.
INSTANT_TOLERANCE_S = 1e-6


def _resolve_as_of(repo_id: str, text: str, built: list[Any]) -> Resolution:
    instant = _parse_instant(text.split(":", 1)[1])
    candidates = [r for r in built if float(r["created_ts"]) <= instant + INSTANT_TOLERANCE_S]
    if not candidates:
        raise NotFound(f"{repo_id} has no version at or before that instant (it is before the first one)")
    newest = max(float(r["created_ts"]) for r in candidates)
    tied = [r for r in candidates if float(r["created_ts"]) == newest]
    if len(tied) > 1:
        raise VersionConflict(
            "more than one version was created at that instant; name one explicitly",
            candidates=[str(r["label"]) for r in tied],
        )
    row = tied[0]
    return Resolution(text, "as_of", (str(row["snapshot_id"]),), (str(row["label"]),))


def _resolve_range(repo_id: str, text: str, rows: list[Any], built: list[Any]) -> Resolution:
    match = RANGE.match(text)
    if not match:
        raise InvalidInput("a range looks like range:<a>..<b>", got=text[:40])
    a, b = match.group("a").strip(), match.group("b").strip()
    order = {str(r["label"]): int(r["ordinal"]) for r in rows}
    if a not in order or b not in order:
        raise NotFound(f"range endpoint not found in {repo_id}", endpoints=[a, b])
    if order[a] > order[b]:
        raise InvalidInput("a range runs in history order: its first endpoint must not be newer than its second")
    chosen = [r for r in built if order[a] <= int(r["ordinal"]) <= order[b]]
    if not chosen:
        raise IndexRequired(f"no version in that range of {repo_id} has been built; run `acis index`")
    return Resolution(
        text, "range", tuple(str(r["snapshot_id"]) for r in chosen), tuple(str(r["label"]) for r in chosen)
    )


__all__ = ["INSTANT_TOLERANCE_S", "KINDS", "Resolution", "resolve", "to_iso"]
