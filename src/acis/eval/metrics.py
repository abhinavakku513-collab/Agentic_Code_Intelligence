"""Retrieval metrics — an **independent** implementation that must equal mteb/pytrec_eval to 1e-9 (Phase 1, B0).

Why re-implement what pytrec_eval already computes: the official number is only trustworthy if we can re-derive it
from a run file without the library that produced it (parity P3, `verify-submission`). So this module computes
NDCG@k, MAP@k, Recall@k, P@k, Success@k and MRR@k from first principles, and `tests/metamorphic` pins it against
mteb's own pipeline on random, oracle and BM25 runs.

Two behaviours are copied deliberately, because they are what the grading harness actually does:

* **Ranking order** is `(score, doc_id)` descending — mteb's MRR sorts that way, and trec_eval breaks ties by
  document id in reverse lexicographic order.
* **Scores are compared at float32 resolution.** pytrec_eval collapses scores that are closer than the float32
  spacing at their magnitude (docs/spec/01 V-08). Rank-derived scores (D9) stay far outside that zone; raw model
  scores do not, which is exactly why D9 exists.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]

K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 20, 100, 1000)
METRIC_FAMILIES = ("ndcg", "map", "recall", "precision", "mrr", "hit_rate")


def as_float32(score: float) -> float:
    """The value pytrec_eval actually compares. Two scores that collapse here are a tie for NDCG."""
    return float(np.float32(score))


def rank_order(doc_scores: Mapping[str, float]) -> list[str]:
    """trec_eval order: **float32** score descending, ties broken by doc id descending (NDCG/MAP/Recall/P/Success)."""
    return [
        d
        for d, _ in sorted(
            ((d, as_float32(s)) for d, s in doc_scores.items()), key=lambda kv: (kv[1], kv[0]), reverse=True
        )
    ]


def mrr_order(doc_scores: Mapping[str, float]) -> list[str]:
    """mteb's MRR order: **exact float** score descending, ties broken by doc id descending.

    The difference from `rank_order` is not cosmetic. Inside the float32 collapse zone the two disagree, so the same
    ranking can score MRR@10 = 1.0 and NDCG@10 = 0.0 (docs/spec/01 V-08). Rank-derived scores (D9) keep every gap at
    1e-3, far outside that zone — which is the entire reason D9 exists.
    """
    return [d for d, _ in sorted(doc_scores.items(), key=lambda kv: (float(kv[1]), kv[0]), reverse=True)]


def _gains(order: Sequence[str], rels: Mapping[str, int]) -> list[int]:
    return [max(0, int(rels.get(doc, 0))) for doc in order]


def ndcg_at_k(order: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """trec_eval's `ndcg_cut.k`: linear gains, log2(rank+1) discount, ideal DCG over all judged relevant documents."""
    gains = _gains(order[:k], rels)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains) if g)
    ideal = sorted((max(0, int(v)) for v in rels.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal) if g)
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(order: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """trec_eval's `map_cut.k`: precision summed at relevant ranks within k, divided by the total relevant count."""
    n_rel = sum(1 for v in rels.values() if int(v) > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    total = 0.0
    for i, doc in enumerate(order[:k], start=1):
        if int(rels.get(doc, 0)) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


def recall_at_k(order: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    n_rel = sum(1 for v in rels.values() if int(v) > 0)
    if n_rel == 0:
        return 0.0
    found = sum(1 for doc in order[:k] if int(rels.get(doc, 0)) > 0)
    return found / n_rel


def precision_at_k(order: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for doc in order[:k] if int(rels.get(doc, 0)) > 0) / k


def success_at_k(order: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """trec_eval's `success.k`, reported by mteb as `hit_rate_at_k`."""
    return 1.0 if any(int(rels.get(doc, 0)) > 0 for doc in order[:k]) else 0.0


def rr_at_k(order: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """Reciprocal rank as mteb computes MRR@k."""
    for rank, doc in enumerate(order[:k], start=1):
        if int(rels.get(doc, 0)) > 0:
            return 1.0 / rank
    return 0.0


_FUNCS = {
    "ndcg": ndcg_at_k,
    "map": map_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": success_at_k,
    "mrr": rr_at_k,
}
# Which ordering each family is graded under (see `rank_order` vs `mrr_order`).
_ORDERERS = {fam: (mrr_order if fam == "mrr" else rank_order) for fam in _FUNCS}


def shared_query_ids(qrels: Qrels, run: Run) -> list[str]:
    """Queries scored by the harness: present in the run **and** in the qrels (pytrec_eval's intersection rule)."""
    return sorted(set(run) & set(qrels))


def per_query(qrels: Qrels, run: Run, metric: str, k: int) -> dict[str, float]:
    """Per-query metric vector — the input of every paired bootstrap (docs/spec/03 §7)."""
    if metric not in _FUNCS:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(_FUNCS)}")
    fn, order_of = _FUNCS[metric], _ORDERERS[metric]
    return {qid: fn(order_of(run[qid]), qrels[qid], k) for qid in shared_query_ids(qrels, run)}


def score_run(qrels: Qrels, run: Run, k_values: Iterable[int] = K_VALUES) -> dict[str, float]:
    """Mean metrics in mteb's key shape (`ndcg_at_10`, `mrr_at_10`, `hit_rate_at_10`, …)."""
    ks = sorted({int(k) for k in k_values})
    qids = shared_query_ids(qrels, run)
    if not qids:
        return {f"{fam}_at_{k}": 0.0 for fam in METRIC_FAMILIES for k in ks}
    trec_orders = {qid: rank_order(run[qid]) for qid in qids}
    exact_orders = {qid: mrr_order(run[qid]) for qid in qids}
    scores: dict[str, float] = {}
    for fam, fn in _FUNCS.items():
        orders = exact_orders if fam == "mrr" else trec_orders
        for k in ks:
            total = sum(fn(orders[qid], qrels[qid], k) for qid in qids)
            scores[f"{fam}_at_{k}"] = total / len(qids)
    scores["accuracy"] = scores.get("recall_at_1", 0.0)  # mteb reports Recall@1 as `accuracy`
    return scores


def score_run_reference(qrels: Qrels, run: Run, k_values: Iterable[int] = K_VALUES) -> dict[str, float]:
    """The same metrics via pytrec_eval + mteb's MRR — the reference our implementation is pinned against."""
    import pytrec_eval  # noqa: PLC0415 — reference path only

    ks = sorted({int(k) for k in k_values})
    ks_str = ",".join(str(k) for k in ks)
    measures = {f"map_cut.{ks_str}", f"ndcg_cut.{ks_str}", f"recall.{ks_str}", f"P.{ks_str}", f"success.{ks_str}"}
    trec_qrels = {q: {d: int(v) for d, v in rels.items()} for q, rels in qrels.items()}
    trec_run = {q: {d: float(s) for d, s in docs.items()} for q, docs in run.items()}
    raw = pytrec_eval.RelevanceEvaluator(trec_qrels, measures).evaluate(trec_run)
    if not raw:
        return {f"{fam}_at_{k}": 0.0 for fam in METRIC_FAMILIES for k in ks}

    out: dict[str, float] = {}
    n = len(raw)
    for k in ks:
        out[f"ndcg_at_{k}"] = sum(v[f"ndcg_cut_{k}"] for v in raw.values()) / n
        out[f"map_at_{k}"] = sum(v[f"map_cut_{k}"] for v in raw.values()) / n
        out[f"recall_at_{k}"] = sum(v[f"recall_{k}"] for v in raw.values()) / n
        out[f"precision_at_{k}"] = sum(v[f"P_{k}"] for v in raw.values()) / n
        out[f"hit_rate_at_{k}"] = sum(v[f"success_{k}"] for v in raw.values()) / n

    from mteb._evaluators.retrieval_metrics import mrr as mteb_mrr  # noqa: PLC0415

    mrr_scores = mteb_mrr(trec_qrels, trec_run, ks)
    for key, values in mrr_scores.items():
        out[f"mrr_at_{key.split('@')[1]}"] = sum(values) / len(values)
    return out


def main_score(scores: Mapping[str, float]) -> float:
    """AppsRetrieval's `main_score` is `ndcg_at_10` (docs/spec/01 V-06)."""
    return float(scores.get("ndcg_at_10", 0.0))


__all__ = [
    "K_VALUES",
    "METRIC_FAMILIES",
    "Qrels",
    "Run",
    "as_float32",
    "main_score",
    "map_at_k",
    "mrr_order",
    "ndcg_at_k",
    "per_query",
    "precision_at_k",
    "rank_order",
    "recall_at_k",
    "rr_at_k",
    "score_run",
    "score_run_reference",
    "shared_query_ids",
    "success_at_k",
]
