"""Decontamination of training pairs (docs/spec/02 §6, integrity rule R6).

Before anything is fitted on dev data, training inputs whose text is near-duplicate of a held-out **input** are
dropped. Only query *texts* are compared — they are public inputs that ship in the same file as the dev queries. No
label is read here, and the result can only ever *remove* training material, never add or reweight it.

Method: MinHash over character 5-grams, 128 permutations, drop at estimated Jaccard ≥ 0.8. The dropped list is logged
so the decision is auditable.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NGRAM = 5
NUM_PERM = 128
DROP_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class DecontamReport:
    n_train: int
    n_reference: int
    dropped: tuple[str, ...]
    threshold: float
    ngram: int
    num_perm: int
    matches: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def kept(self) -> int:
        return self.n_train - self.n_dropped

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_train": self.n_train,
                "n_reference": self.n_reference,
                "n_dropped": self.n_dropped,
                "kept": self.kept,
                "threshold": self.threshold,
                "ngram": self.ngram,
                "num_perm": self.num_perm,
                "dropped": list(self.dropped),
                "matches": dict(self.matches),
            },
            indent=2,
            sort_keys=True,
        )

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p


def normalise(text: str) -> str:
    """Comparison form: NFKC, case-folded, whitespace collapsed. Structure noise must not hide a duplicate."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text: str, n: int = NGRAM) -> set[bytes]:
    """Character n-grams of the normalised text (word n-grams miss reformatted code)."""
    norm = normalise(text)
    if len(norm) < n:
        return {norm.encode("utf-8")} if norm else set()
    return {norm[i : i + n].encode("utf-8") for i in range(len(norm) - n + 1)}


def jaccard(a: set[bytes], b: set[bytes]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _minhash(text: str, *, num_perm: int, ngram: int) -> Any:
    from datasketch import MinHash  # noqa: PLC0415 — optional on the hot path

    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, ngram):
        m.update(sh)
    return m


def decontaminate(
    train: Mapping[str, str],
    reference: Iterable[str],
    *,
    threshold: float = DROP_THRESHOLD,
    ngram: int = NGRAM,
    num_perm: int = NUM_PERM,
    validate_ids: bool = False,
) -> DecontamReport:
    """Drop every training entry whose text is ≥ `threshold` Jaccard-similar to any reference input.

    `train` is `{id: text}`; `reference` is the held-out **input** texts. The estimate comes from MinHash LSH and is
    confirmed exactly on the candidates, so a near-miss is never dropped on an approximation alone.

    `validate_ids=True` (what every real training export must pass) asserts that the keys are dev-pool query ids
    and that no held-out id reached this path. It is off by default so the function stays usable on arbitrary
    `{id: text}` mappings in tests and on non-APPS corpora.
    """
    from acis.eval.guard import assert_dev_pool, assert_no_holdout_ids  # noqa: PLC0415 — avoids an import cycle

    # The programmatic half of the seal, wired in at the one place that currently fits training material:
    # every key must be a dev-pool query, and no held-out id may reach it (docs/spec/03 §5).
    if validate_ids and train:
        assert_no_holdout_ids(train.keys(), context="decontaminate")
        assert_dev_pool(train.keys(), context="decontaminate")

    reference_texts = [r for r in reference if r and r.strip()]
    if not train or not reference_texts:
        return DecontamReport(len(train), len(reference_texts), (), threshold, ngram, num_perm)

    from datasketch import MinHashLSH  # noqa: PLC0415

    from acis.core.hashing import sha256_text  # noqa: PLC0415

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ref_shingles: list[set[bytes]] = []
    # The reference key is a digest of the text, never its position in the held-out list. A positional index is
    # reversible in the dev environment, which would turn the drop report into a held-out ↔ train alignment —
    # an artifact we generated that points at specific held-out queries.
    ref_keys: list[str] = []
    for i, text in enumerate(reference_texts):
        lsh.insert(f"r{i}", _minhash(text, num_perm=num_perm, ngram=ngram))
        ref_shingles.append(shingles(text, ngram))
        ref_keys.append(sha256_text(normalise(text))[:16])

    dropped: list[str] = []
    matches: dict[str, str] = {}
    for key, text in train.items():
        candidates = lsh.query(_minhash(text, num_perm=num_perm, ngram=ngram))
        if not candidates:
            continue
        own = shingles(text, ngram)
        for cand in candidates:
            idx = int(cand[1:])
            if jaccard(own, ref_shingles[idx]) >= threshold:
                dropped.append(key)
                matches[key] = ref_keys[idx]
                break

    return DecontamReport(
        n_train=len(train),
        n_reference=len(reference_texts),
        dropped=tuple(sorted(dropped)),
        threshold=threshold,
        ngram=ngram,
        num_perm=num_perm,
        matches=matches,
    )


def exact_duplicate_groups(texts: Mapping[str, str]) -> dict[str, list[str]]:
    """`{normalised_text_key: [id, …]}` for every group with more than one member (duplicate audit, D9 ordering)."""
    from acis.core.hashing import normalized_text_key  # noqa: PLC0415

    groups: dict[str, list[str]] = {}
    for key, text in texts.items():
        groups.setdefault(normalized_text_key(text), []).append(key)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def holdout_reference_texts() -> Sequence[str]:
    """Held-out query **texts** (never labels), used only to remove contaminated training pairs."""
    from acis.appsdata import apps  # noqa: PLC0415
    from acis.eval.splits import holdout_query_ids  # noqa: PLC0415

    return [apps.query_text(qid) for qid in holdout_query_ids()]


__all__ = [
    "DROP_THRESHOLD",
    "NGRAM",
    "NUM_PERM",
    "DecontamReport",
    "decontaminate",
    "exact_duplicate_groups",
    "holdout_reference_texts",
    "jaccard",
    "normalise",
    "shingles",
]
