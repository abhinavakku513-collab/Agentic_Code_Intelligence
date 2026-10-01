#!/usr/bin/env python
"""Full-corpus dense diagnostics on the dev split from dumped matrices (`dump_vectors.py`).

Per encoder: NDCG@10, MRR@10, Recall@{10,50,100,200,500,1000}; marginal recall over gte (gold in the other
encoder's top-k but not gte's); and score-level ensembles gte + w·other. One relevant document per query, so
NDCG@10 = 1/log2(1+rank) for rank ≤ 10 exactly (checked against `acis.eval.metrics` once).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from acis.core.paths import acis_root
from acis.eval.dev_task import dev_qrels

LAB = acis_root() / "runs" / "lab"
KS = (1, 10, 50, 100, 200, 500, 1000)


def load(key: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    z = np.load(LAB / f"vec.{key}.npz")
    return z["queries"], z["docs"], [str(q) for q in z["qids"]], [str(d) for d in z["doc_ids"]]


def gold_ranks(scores: np.ndarray, gold: np.ndarray) -> np.ndarray:
    g = scores[np.arange(len(gold)), gold][:, None]
    return (scores > g).sum(axis=1) + 1


def summary(ranks: np.ndarray) -> dict[str, float]:
    ndcg = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    mrr = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    out = {"ndcg_at_10": 100 * ndcg.mean(), "mrr_at_10": 100 * mrr.mean()}
    out.update({f"R@{k}": 100 * float((ranks <= k).mean()) for k in KS})
    return out


def main() -> int:
    keys = sys.argv[1:] or ["gte-modernbert-base", "granite-embedding-english-r2", "granite-embedding-small-english-r2"]
    base_q, base_d, qids, doc_ids = load(keys[0])
    qrels = dev_qrels(qids)
    col = {d: i for i, d in enumerate(doc_ids)}
    gold = np.array([col[next(d for d, r in qrels[q].items() if r > 0)] for q in qids])
    scores = {}
    report = {}
    for key in keys:
        q, d, _, _ = load(key)
        s = q @ d.T
        scores[key] = s
        report[key] = summary(gold_ranks(s, gold))
    base = keys[0]
    rb = gold_ranks(scores[base], gold)
    for key in keys[1:]:
        ro = gold_ranks(scores[key], gold)
        report[key]["marginal_over_" + base] = {
            f"@{k}": int(((ro <= k) & (rb > k)).sum()) for k in (10, 100, 300, 500)
        }
        report[key]["base_only"] = {f"@{k}": int(((rb <= k) & (ro > k)).sum()) for k in (10, 100, 300, 500)}
        sb = scores[base]
        so = scores[key]
        # per-query z-scored fusion (within the query's own corpus scores, INV-3)
        zb = (sb - sb.mean(1, keepdims=True)) / sb.std(1, keepdims=True)
        zo = (so - so.mean(1, keepdims=True)) / so.std(1, keepdims=True)
        for w in (0.1, 0.2, 0.3, 0.5):
            report[key][f"zfuse_w{w}"] = summary(gold_ranks(zb + w * zo, gold))
    if len(keys) > 2:
        sb = scores[base]
        zs = [(scores[k] - scores[k].mean(1, keepdims=True)) / scores[k].std(1, keepdims=True) for k in keys]
        report["all3_z_0.2_0.2"] = summary(gold_ranks(zs[0] + 0.2 * zs[1] + 0.2 * zs[2], gold))
    print(json.dumps(report, indent=1, default=float))
    (LAB / "encoder_diag.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
