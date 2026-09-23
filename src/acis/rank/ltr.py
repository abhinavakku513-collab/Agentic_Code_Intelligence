"""The learned ranker: LightGBM LambdaRank, cross-fitted and abstaining (docs/spec/02 §4, gate G5).

The ranker exists to combine evidence the channels produce separately, and the two design decisions that matter
are both about not trusting it too far:

* **Cross-fitting.** A ranker trained on the same queries it is scored on reports its own memory. Training runs
  K-fold: each fold's queries are scored by a model that never saw them, and the out-of-fold scores are what the
  gate reads (`docs/spec/09` §2). The shipped model is a refit on everything with the hyper-parameters frozen.
* **Abstention.** When fewer than ρ of the feature groups fired — a short query with no numbers, no quoted
  output and no lexical match — the ranker has nothing to add, and it returns the dense order untouched rather
  than inventing one. That is what makes it safe on arbitrary queries (INV-15, spec 10 §5), and it is counted as
  a degradation so the run manifest says how often it happened (INV-7).

Group dropout during training (≈25 % of groups masked per sample) is the third: it trains the model to survive a
missing family of evidence, which is exactly what an unfamiliar query produces.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from acis.core.errors import InvalidInput, NotReady
from acis.rank.candidates import FEATURE_NAMES, GROUPS, MONOTONE, group_availability, mask_group

#: Frozen hyper-parameters (spec 02 §4). Changing one is a gate decision, not a tweak.
PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "lambdarank_truncation_level": 20,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": 0,
    "deterministic": True,
    "force_row_wise": True,
}
DEFAULT_ROUNDS = 400
DROPOUT_RATE = 0.25
#: Below this share of groups firing, the ranker abstains and the dense order stands.
DEFAULT_RHO = 0.35


@dataclass(frozen=True, slots=True)
class TrainingGroup:
    """One query's candidates: features, labels and the dense order to fall back on."""

    query_id: str
    features: np.ndarray
    labels: np.ndarray
    doc_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.shape[0] != len(self.labels) or self.features.shape[0] != len(self.doc_ids):
            raise InvalidInput("a training group's features, labels and ids must line up")


@dataclass(slots=True)
class Ranker:
    """A trained LambdaRank model plus the contract it is served under."""

    booster: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES
    rho: float = DEFAULT_RHO
    rounds: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] != len(self.feature_names):
            raise InvalidInput(
                "feature width does not match the trained model",
                expected=len(self.feature_names),
                got=int(features.shape[1]),
            )
        return np.asarray(self.booster.predict(features), dtype=np.float64)

    def should_abstain(self, features: np.ndarray) -> bool:
        """True when too little evidence fired for the model to add anything (spec 10 §5)."""
        availability = group_availability(features)
        fired = sum(1 for value in availability.values() if value > 0.0)
        return (fired / max(1, len(GROUPS))) < self.rho

    def rerank(self, doc_ids: Sequence[str], features: np.ndarray) -> tuple[list[str], bool]:
        """Return `(order, abstained)`. Abstaining returns the input order unchanged — the dense order."""
        if not len(doc_ids):
            return [], False
        if self.should_abstain(features):
            return list(doc_ids), True
        scores = self.score(features)
        order = np.argsort(-scores, kind="stable")
        return [doc_ids[i] for i in order], False

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.booster.model_to_string(), encoding="utf-8")
        target.with_suffix(".meta.json").write_text(
            json.dumps(
                {"feature_names": list(self.feature_names), "rho": self.rho, "rounds": self.rounds, **self.meta},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Ranker:
        import lightgbm as lgb  # noqa: PLC0415

        target = Path(path)
        if not target.is_file():
            raise NotReady(f"no ranker at {target}; train one with `acis eval gate --gate G5`")
        booster = lgb.Booster(model_str=target.read_text(encoding="utf-8"))
        meta_path = target.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        return cls(
            booster=booster,
            feature_names=tuple(meta.get("feature_names", FEATURE_NAMES)),
            rho=float(meta.get("rho", DEFAULT_RHO)),
            rounds=int(meta.get("rounds", 0)),
            meta=meta,
        )


def _monotone_constraints() -> list[int]:
    return [MONOTONE.get(name, 0) for name in FEATURE_NAMES]


def _augment(groups: Sequence[TrainingGroup], *, seed: int, rate: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stack groups into LightGBM's flat form, masking one feature group in `rate` of the queries.

    The dropout is applied per *query*, not per row: masking half a query's candidates would teach the model that
    a missing feature marks a bad document, which is the opposite of the lesson.
    """
    rng = np.random.default_rng(seed)
    names = list(GROUPS)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sizes: list[int] = []
    for group in groups:
        matrix = group.features
        if rate > 0 and rng.random() < rate:
            matrix = mask_group(matrix, names[int(rng.integers(0, len(names)))])
        blocks.append(matrix)
        labels.append(group.labels)
        sizes.append(matrix.shape[0])
    return np.vstack(blocks), np.concatenate(labels), sizes


def train(
    groups: Sequence[TrainingGroup],
    *,
    rounds: int = DEFAULT_ROUNDS,
    seed: int = 0,
    dropout: float = DROPOUT_RATE,
    rho: float = DEFAULT_RHO,
) -> Ranker:
    """Fit one LambdaRank model. Deterministic for a seed: the same groups give the same booster."""
    import lightgbm as lgb  # noqa: PLC0415

    if not groups:
        raise InvalidInput("the ranker needs at least one training group")
    features, labels, sizes = _augment(groups, seed=seed, rate=dropout)
    dataset = lgb.Dataset(
        features,
        label=labels,
        group=sizes,
        feature_name=list(FEATURE_NAMES),
        params={"verbosity": -1},
        free_raw_data=False,
    )
    booster = lgb.train(
        {**PARAMS, "seed": seed, "monotone_constraints": _monotone_constraints()},
        dataset,
        num_boost_round=rounds,
    )
    return Ranker(
        booster=booster,
        rho=rho,
        rounds=rounds,
        meta={"n_groups": len(groups), "seed": seed, "dropout": dropout},
    )


def cross_fit(
    groups: Sequence[TrainingGroup],
    *,
    folds: Mapping[str, int],
    rounds: int = DEFAULT_ROUNDS,
    seed: int = 0,
    dropout: float = DROPOUT_RATE,
) -> dict[str, list[tuple[str, float]]]:
    """Out-of-fold scores: every query is scored by a model that never saw it (`docs/spec/09` §2).

    This is what a gate reads. A ranker scored on its own training queries reports its memory, and the difference
    is large enough to turn a regression into an apparent improvement.
    """
    by_fold: dict[int, list[TrainingGroup]] = {}
    for group in groups:
        by_fold.setdefault(int(folds.get(group.query_id, 0)), []).append(group)
    if len(by_fold) < 2:
        raise InvalidInput("cross-fitting needs at least two folds", folds=sorted(by_fold))

    out: dict[str, list[tuple[str, float]]] = {}
    for held_out, members in sorted(by_fold.items()):
        training = [g for fold, gs in by_fold.items() if fold != held_out for g in gs]
        model = train(training, rounds=rounds, seed=seed, dropout=dropout)
        for group in members:
            scores = model.score(group.features)
            out[group.query_id] = list(zip(group.doc_ids, (float(s) for s in scores), strict=True))
    return out


__all__ = [
    "DEFAULT_RHO",
    "DEFAULT_ROUNDS",
    "DROPOUT_RATE",
    "PARAMS",
    "Ranker",
    "TrainingGroup",
    "cross_fit",
    "train",
]
