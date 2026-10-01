#!/usr/bin/env python
"""Failure buckets over every dev query (dev split only): retrieval failures vs ranking failures.

Inputs: the served pools (`runs/lab/base.npz`), dumped dense matrices (`dump_vectors.py`) and a feature-lab ranks
file (`feature_lab.py --save-ranks`). Per query it records the gold document's rank under each channel over the
full corpus (dense encoders) or within the channel's own top list (BM25, symbols), whether the gold reached the
candidate pool, and the final rank; then assigns exactly one bucket:

* `top1` — final rank 1;  `top10` — final rank 2–10 (a ranking loss inside the metric window);
* `ranking_failure` — gold in the pool, final rank > 10;
* `union_failure` — gold outside the pool although the dense channel ranks it ≤ 1000 (a deeper cut would find it);
* `missing_from_all` — gold outside the pool and beyond dense rank 1000 and every lexical list.

Side flags (not buckets): `near_duplicate_top1` (served #1 has gte cosine ≥ 0.97 to the gold),
`long_query` (query over 4,000 characters, i.e. head+tail truncation likely), `rescued` (dense rank > 10, final ≤ 10),
`demoted` (dense rank ≤ 10, final > 10).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from acis.appsdata import apps
from acis.core.paths import acis_root
from acis.eval.dev_task import dev_qrels

LAB = acis_root() / "runs" / "lab"


def full_rank(q: np.ndarray, d: np.ndarray, gold: np.ndarray) -> np.ndarray:
    s = q @ d.T
    g = s[np.arange(len(gold)), gold][:, None]
    return (s > g).sum(1) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", required=True, help="feature-lab ranks file in runs/lab")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--encoders", default="gte-modernbert-base,granite-embedding-small-english-r2")
    parser.add_argument("--out", default="runs/lab/failure_buckets")
    args = parser.parse_args(argv)
    z = np.load(LAB / "base.npz")
    X, off, pos = z["X"], z["offsets"], z["pos"]
    names = [str(n) for n in z["feature_names"]]
    qids = [str(q) for q in z["qids"]]
    doc_ids = [str(d) for d in z["doc_ids"]]
    qrels = dev_qrels(qids)
    col = {d: i for i, d in enumerate(doc_ids)}
    gold = np.array([col[next(d for d, r in qrels[q].items() if r > 0)] for q in qids])
    final = np.array(json.loads((LAB / args.ranks).read_text())["ranks"][args.variant])
    queries = apps.load_queries()

    enc_ranks = {}
    gte = np.load(LAB / "vec.gte-modernbert-base.npz")
    for key in args.encoders.split(","):
        v = np.load(LAB / f"vec.{key}.npz")
        enc_ranks[key] = full_rank(v["queries"], v["docs"], gold)
    i_lex, i_ch = names.index("rank_lex"), names.index("n_channels")
    rows = []
    for i, q in enumerate(qids):
        p = pos[off[i] : off[i + 1]]
        hit = np.flatnonzero(p == gold[i])
        in_pool = bool(hit.size)
        lex = float(X[off[i] + hit[0], i_lex]) if in_pool else float("nan")
        dense = int(enc_ranks["gte-modernbert-base"][i])
        f = int(final[i])
        if f == 1:
            bucket = "top1"
        elif f <= 10:
            bucket = "top10"
        elif in_pool:
            bucket = "ranking_failure"
        elif dense <= 1000:
            bucket = "union_failure"
        else:
            bucket = "missing_from_all"
        rows.append(
            {
                "query_id": q,
                "gold_doc": doc_ids[gold[i]],
                **{f"rank_{k}": int(r[i]) for k, r in enc_ranks.items()},
                "rank_bm25_top200": None if lex != lex else int(lex),
                "n_channels_gold": float(X[off[i] + hit[0], i_ch]) if in_pool else 0.0,
                "in_pool": in_pool,
                "pool_size": int(len(p)),
                "final_rank": f if f < 10**6 else None,
                "ndcg_at_10": float(1 / np.log2(f + 1)) if f <= 10 else 0.0,
                "mrr_at_10": float(1 / f) if f <= 10 else 0.0,
                "bucket": bucket,
                "rescued": dense > 10 and f <= 10,
                "demoted": dense <= 10 and f > 10,
                "long_query": len(queries[q]) > 4000,
            }
        )
    # near-duplicate confusion: when the gold is not first, is the doc above it almost the same program?
    d = gte["docs"]
    counts = Counter(r["bucket"] for r in rows)
    summary = {
        "variant": args.variant,
        "n_queries": len(rows),
        "buckets": dict(counts),
        "rescued": sum(r["rescued"] for r in rows),
        "demoted": sum(r["demoted"] for r in rows),
        "long_query_share_of_failures": float(
            np.mean([r["long_query"] for r in rows if r["bucket"] in ("ranking_failure", "union_failure", "missing_from_all")] or [0])
        ),
        "long_query_share_overall": float(np.mean([r["long_query"] for r in rows])),
        "pool_recall": float(np.mean([r["in_pool"] for r in rows])),
        "gold_found_by_other_encoder_only_at_100": {
            k: int(((enc_ranks[k] <= 100) & (enc_ranks["gte-modernbert-base"] > 100)).sum()) for k in enc_ranks
        },
    }
    # max doc-doc cosine between gold and the other docs (how many golds have a near twin in the corpus)
    sims = d[gold] @ d.T
    sims[np.arange(len(gold)), gold] = -1
    twin = sims.max(1)
    summary["gold_has_near_twin_ge_0.97"] = int((twin >= 0.97).sum())
    summary["near_twin_failures"] = int(sum(1 for r, t in zip(rows, twin, strict=True) if t >= 0.97 and r["bucket"] != "top1"))
    out = acis_root() / args.out
    out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
