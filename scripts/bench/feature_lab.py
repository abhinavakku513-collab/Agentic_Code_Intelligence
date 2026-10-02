#!/usr/bin/env python
"""Feature lab: extra dense signals as ranker features over the served candidate pools, out of fold (dev only).

Reads `runs/lab/base.npz` (the served pools: `ranker_lab.py build`) and dumped matrices (`dump_vectors.py`), adds
per-candidate features — each extra encoder's (or query view's) cosine, within-pool z-score and full-corpus rank —
and cross-fits LambdaRank exactly like `ranker_lab.evaluate`: five folds, group dropout, abstention on the original
groups, dense tail. Generic-routed queries keep the served weighted fusion unless `--rank-all`.

Every feature is computed from one query's own vector against the corpus (INV-3); none is a document prior.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from acis.core.paths import acis_root
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.splits import fold_members
from acis.rank import ltr
from acis.rank.candidates import GROUPS, MONOTONE

LAB = acis_root() / "runs" / "lab"
TOP_K = 100


def _load_vec(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """`tag` = file tag, or `queries_tag@docs_tag` to pair one view's queries with another file's documents."""
    qtag, _, dtag = tag.partition("@")
    zq = np.load(LAB / f"vec.{qtag}.npz")
    zd = np.load(LAB / f"vec.{dtag or qtag}.npz")
    return zq["queries"], zd["docs"]


def dense_features(q: np.ndarray, d: np.ndarray, pos: np.ndarray, off: np.ndarray) -> np.ndarray:
    """(rows, 3): cosine, within-pool z, full-corpus rank (1 = best) of every pooled candidate."""
    out = np.empty((len(pos), 3), dtype=np.float32)
    for i in range(len(off) - 1):
        s = d @ q[i]
        c = s[pos[off[i] : off[i + 1]]]
        srt = np.sort(s)
        rank = len(s) - np.searchsorted(srt, c, side="right") + 1
        sd = c.std() or 1.0
        out[off[i] : off[i + 1]] = np.stack([c, (c - c.mean()) / sd, rank], axis=1)
    return out


def prf_features(q: np.ndarray, d: np.ndarray, pos: np.ndarray, off: np.ndarray, m: int, beta: float) -> np.ndarray:
    """Rocchio in embedding space: q' = q + beta·mean(top-m docs); cosine and within-pool z of every candidate."""
    out = np.empty((len(pos), 2), dtype=np.float32)
    for i in range(len(off) - 1):
        s = d @ q[i]
        top = np.argpartition(-s, m)[:m]
        q2 = q[i] + beta * d[top].mean(axis=0)
        q2 /= np.linalg.norm(q2)
        c = d[pos[off[i] : off[i + 1]]] @ q2
        out[off[i] : off[i + 1]] = np.stack([c, (c - c.mean()) / (c.std() or 1.0)], axis=1)
    return out


def mix_features(X: np.ndarray, names: list[str], off: np.ndarray, gte_rank: np.ndarray) -> np.ndarray:
    """Agreement and residuals between the two dense encoders, within one query's own pool (INV-3).

    Columns: z_cos + z_cos2, z_cos − z_cos2, log(gte whole-corpus rank) − log(Qwen whole-corpus rank),
    log min(rank, rank2), and the candidate's rank inside the pool by z_cos + z_cos2.
    """
    zc, zq, rq = (X[:, names.index(n)] for n in ("z_cos", "z_cos2", "rank_dense2"))
    out = np.full((len(X), 5), np.nan, dtype=np.float32)
    out[:, 0] = zc + zq
    out[:, 1] = zc - zq
    out[:, 2] = np.log(gte_rank) - np.log(rq)
    out[:, 3] = np.log(np.minimum(gte_rank, rq))
    for i in range(len(off) - 1):
        block = out[off[i] : off[i + 1], 0]
        order = np.argsort(-np.nan_to_num(block, nan=-1e9), kind="stable")
        ranks = np.empty(len(block), dtype=np.float32)
        ranks[order] = np.arange(1, len(block) + 1)
        out[off[i] : off[i + 1], 4] = ranks
    return out


def context_features(d: np.ndarray, pos: np.ndarray, off: np.ndarray, cos2: np.ndarray, top: int = 5) -> np.ndarray:
    """Pool context from document–document similarity under one encoder (no query statistics across queries).

    Columns: cosine to the pool's top-1 by the query cosine, mean cosine to the pool's top-`top`, and the number of
    near-duplicates (document cosine ≥ 0.97) of this candidate inside the pool.
    """
    out = np.empty((len(pos), 3), dtype=np.float32)
    for i in range(len(off) - 1):
        p = pos[off[i] : off[i + 1]]
        c = np.nan_to_num(cos2[off[i] : off[i + 1]], nan=-1.0)
        best = np.argsort(-c, kind="stable")[:top]
        vec = d[p]
        sims = vec @ vec[best].T
        near = (vec @ vec.T >= 0.97).sum(axis=1) - 1
        out[off[i] : off[i + 1]] = np.stack([sims[:, 0], sims.mean(axis=1), near], axis=1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", default="base")
    parser.add_argument("--variants", required=True, help='JSON {"name": {"add": [groups], "drop": [cols], ...}}')
    parser.add_argument("--rank-all", action="store_true", help="the ranker ranks every query (no fusion route)")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--save-ranks", default="", help="write per-query gold ranks of each variant here")
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="partition-neutral leakage check: keep only candidates that are some dev query's gold document",
    )
    args = parser.parse_args(argv)
    import lightgbm as lgb

    z = np.load(LAB / f"{args.dump}.npz", allow_pickle=False)
    X, y, off, pos, dense = z["X"], z["y"], z["offsets"], z["pos"], z["dense"]
    names = [str(n) for n in z["feature_names"]]
    qids = [str(q) for q in z["qids"]]
    doc_ids = [str(d) for d in z["doc_ids"]]
    routes = [str(r) for r in z["routes"]]
    qrels = dev_qrels(qids)
    col = {d: i for i, d in enumerate(doc_ids)}
    gold = np.array([col[next(d for d, r in qrels[q].items() if r > 0)] for q in qids])
    fold_of = {q: f for f, members in fold_members().items() for q in members}
    variants: dict[str, dict[str, Any]] = json.loads(args.variants)
    if args.neutral:
        # On dev, every gold is a dev-partition document while a third of the distractors are not, so a document
        # prior that tells the two partitions apart (length does: AUC 0.65) looks useful on dev and points the wrong
        # way on the held-out split. Restricting every pool and tail to dev-gold documents (from dev labels only)
        # removes that shortcut; a feature whose gain survives here is a relevance signal, not a partition one.
        keep_docs = np.zeros(len(doc_ids), dtype=bool)
        keep_docs[gold] = True
        mask = keep_docs[pos]
        sizes = np.add.reduceat(mask.astype(np.int64), off[:-1])
        X, y, pos = X[mask], y[mask], pos[mask]
        off = np.concatenate([[0], np.cumsum(sizes)])
        dense = np.where(keep_docs[np.maximum(dense, 0)] & (dense >= 0), dense, -1)
        print(f"neutral: kept {mask.mean():.1%} of candidates", flush=True)

    # Extra feature blocks, computed once. Group name -> (column names, matrix).
    extra: dict[str, tuple[list[str], np.ndarray]] = {}
    needed = {g for v in variants.values() for g in v.get("add", [])}
    for g in sorted(needed):
        t = time.perf_counter()
        kind, _, spec = g.partition(":")
        if kind == "enc":  # enc:<tag>
            q, d = _load_vec(spec)
            extra[g] = ([f"{spec}.cos", f"{spec}.z", f"{spec}.rank"], dense_features(q, d, pos, off))
        elif kind == "mix":  # mix: gte/Qwen agreement and residuals (needs the gte dump for the gte whole rank)
            q, d = _load_vec("gte-modernbert-base")
            gte_rank = dense_features(q, d, pos, off)[:, 2]
            extra[g] = (
                ["mix.zsum", "mix.zdiff", "mix.lograt", "mix.logmin", "mix.poolrank"],
                mix_features(X, names, off, gte_rank),
            )
        elif kind == "ctx":  # ctx:<tag>: pool context under that encoder's document vectors
            _, d = _load_vec(spec)
            extra[g] = (
                [f"ctx.{spec}.top1", f"ctx.{spec}.top5", f"ctx.{spec}.near"],
                context_features(d, pos, off, X[:, names.index("cos2")]),
            )
        elif kind == "prf":  # prf:<tag>:m:beta
            tag, m, beta = spec.split(":")
            q, d = _load_vec(tag)
            extra[g] = ([f"{tag}.prf{m}.cos", f"{tag}.prf{m}.z"], prf_features(q, d, pos, off, int(m), float(beta)))
        else:
            raise SystemExit(f"unknown group {g}")
        print(f"built {g} in {time.perf_counter() - t:.0f}s", flush=True)

    i_cos, i_bm, i_cos2 = names.index("cos"), names.index("bm25"), names.index("cos2")
    base_groups = {g: [names.index(n) for n in cols] for g, cols in GROUPS.items()}
    results: list[dict[str, Any]] = []
    all_ranks: dict[str, list[int]] = {}
    for vname, spec in variants.items():
        t = time.perf_counter()
        add = list(spec.get("add", []))
        drop = set(spec.get("drop", ()))
        keep = [i for i, n in enumerate(names) if n not in drop]
        blocks = [X[:, keep]] + [extra[g][1] for g in add]
        full = np.hstack(blocks)
        fnames = [names[i] for i in keep] + [n for g in add for n in extra[g][0]]
        # dropout groups: original groups (restricted to kept columns) + one group per added block
        groups: list[list[int]] = []
        for cols in base_groups.values():
            idx = [keep.index(c) for c in cols if c in keep]
            if idx:
                groups.append(idx)
        start = len(keep)
        for g in add:
            k = len(extra[g][0])
            groups.append(list(range(start, start + k)))
            start += k
        rounds = int(spec.get("rounds", ltr.DEFAULT_ROUNDS))
        dropout = float(spec.get("dropout", ltr.DROPOUT_RATE))
        params = {
            **ltr.PARAMS,
            "seed": 0,
            "monotone_constraints": [MONOTONE.get(n, 1 if n.endswith(".cos") else 0) for n in fnames],
            **spec.get("params", {}),
        }
        rng = np.random.default_rng(0)
        train_x = full.copy()
        for i in range(len(qids)):
            if dropout > 0 and rng.random() < dropout:
                gcols = groups[int(rng.integers(0, len(groups)))]
                train_x[off[i] : off[i + 1], gcols] = np.nan
        rank_all = bool(spec.get("rank_all", args.rank_all))
        order_of: dict[int, np.ndarray] = {}
        for fold in sorted(set(fold_of.values())):
            tr = np.array([fold_of[q] != fold for q in qids])
            rows = np.repeat(tr, np.diff(off))
            ds = lgb.Dataset(
                train_x[rows], label=y[rows], group=np.diff(off)[tr], feature_name=fnames, params={"verbosity": -1}
            )
            booster = lgb.train(params, ds, num_boost_round=rounds)
            for i, q in enumerate(qids):
                if fold_of[q] != fold:
                    continue
                cand = pos[off[i] : off[i + 1]]
                orig = X[off[i] : off[i + 1]]
                if routes[i] != "statement_like" and not rank_all:
                    c, b = orig[:, i_cos], orig[:, i_bm]

                    def minmax(v: np.ndarray) -> np.ndarray:
                        lo, hi = np.nanmin(v), np.nanmax(v)
                        return np.nan_to_num((v - lo) / (hi - lo)) if hi > lo else np.zeros_like(v)

                    mm = minmax(c)
                    w = float(spec.get("fusion_aux_weight", 0.0))
                    if w and np.isfinite(orig[:, i_cos2]).any():  # the served fusion's second-encoder mix
                        mm = (1 - w) * mm + w * minmax(orig[:, i_cos2])
                    bmax = np.nanmax(b) if np.isfinite(b).any() else np.nan
                    bn = np.nan_to_num(b / bmax) if bmax and bmax > 0 else np.zeros_like(c)
                    alpha = float(spec.get("fusion_alpha", 0.9))
                    s = alpha * mm + (1 - alpha) * bn
                else:
                    s = booster.predict(full[off[i] : off[i + 1]])
                order_of[i] = cand[np.argsort(-s, kind="stable")]
        ranks = []
        for i in range(len(qids)):
            head = list(order_of[i])
            seen = set(head)
            order = head + [int(p) for p in dense[i] if p >= 0 and p not in seen]
            r = next((k for k, p in enumerate(order[:TOP_K], 1) if p == gold[i]), 10**6)
            ranks.append(r)
        r = np.array(ranks)
        ndcg = np.where(r <= 10, 1 / np.log2(r + 1), 0.0)
        mrr = np.where(r <= 10, 1 / r, 0.0)
        rec = {
            "variant": vname,
            "spec": spec,
            "ndcg_at_10": 100 * ndcg.mean(),
            "mrr_at_10": 100 * mrr.mean(),
            "hit_at_1": 100 * float((r <= 1).mean()),
            "recall_at_10": 100 * float((r <= 10).mean()),
            "recall_at_100": 100 * float((r <= 100).mean()),
            "seconds": round(time.perf_counter() - t, 1),
            "n_features": len(fnames),
        }
        all_ranks[vname] = ranks
        per = {q: float(v) for q, v in zip(qids, ndcg, strict=True)}
        rec["_per"] = per
        results.append(rec)
        base = next((x for x in results if x["variant"] == args.baseline), results[0])
        boot = paired_bootstrap(per, base["_per"])
        rec["vs_baseline"] = boot.as_dict()
        print(
            f"{vname:<34} NDCG@10 {rec['ndcg_at_10']:.2f} MRR@10 {rec['mrr_at_10']:.2f} H@1 {rec['hit_at_1']:.2f} "
            f"R@10 {rec['recall_at_10']:.2f} R@100 {rec['recall_at_100']:.2f}  Δ {boot.delta:+.2f} "
            f"[{boot.ci_low:+.2f}, {boot.ci_high:+.2f}] ({rec['seconds']}s)",
            flush=True,
        )
    out = LAB / "feature_lab.results.json"
    old = json.loads(out.read_text("utf-8")) if out.is_file() else []
    out.write_text(json.dumps(old + [{k: v for k, v in r.items() if k != "_per"} for r in results], indent=1))
    if args.save_ranks:
        (LAB / args.save_ranks).write_text(json.dumps({"qids": qids, "ranks": all_ranks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
