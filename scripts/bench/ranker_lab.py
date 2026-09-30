#!/usr/bin/env python
"""Ranker lab: dump the served candidate pools once, then compare ranker variants out of fold (dev split only).

`build` runs every dev query through the engine's own `candidate_pool` and `pool_features` (the functions serving
calls) and saves the pools, feature matrices, labels and the dense order to `runs/lab/<name>.npz`. `eval` then
cross-fits LambdaRank variants on that dump — hyper-parameters, feature subsets, extra features computed from the
stored candidates — so a question about the ranker costs seconds instead of a full pool rebuild.

Every variant is scored exactly like the served pipeline: each fold's queries are ranked by a model trained on the
other four folds, the ranker may abstain (dense order), and the dense tail fills the list to 100. Winners are then
confirmed end to end by `eval_pipeline.py`, which is the number that gets recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import K_VALUES, per_query, score_run
from acis.eval.splits import fold_members
from acis.rank import ltr
from acis.rank.candidates import FEATURE_NAMES, GROUPS, MONOTONE, group_availability, mask_group
from acis.rank.compose import rank_derived_scores

TOP_K = 100
LAB = acis_root() / "runs" / "lab"


def build(name: str, *, config_path: str, overrides: dict[str, Any], artifact: str) -> Path:
    config = load_frozen_config(config_path).with_overrides(**overrides)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    data = engine.snapshot_data(engine.build_snapshot(apps.load_corpus(), source=f"lab:{name}"))
    data.warm_features()
    ids = list(apps.dev_query_ids())
    qrels = dev_qrels(ids)
    queries = apps.load_queries()
    records = [json.loads(x) for x in (acis_root() / artifact).read_text("utf-8").splitlines() if x]
    routes = {r["query_id"]: r["route"]["route"] for r in records}
    blocks, labels, positions, offsets, dense, pool_hit, qlen = [], [], [], [0], [], [], []
    started = time.perf_counter()
    for n, qid in enumerate(ids, 1):
        query, _ = engine.normalise_query(queries[qid])
        head, pool = engine.candidate_pool(data, query, route=routes[qid], want=TOP_K)
        gold = {d for d, r in qrels[qid].items() if r > 0}
        blocks.append(engine.pool_features(data, query, pool))
        labels.append(np.array([int(c.doc_id in gold) for c in pool], dtype=np.int8))
        positions.append(np.array([data.position(c.doc_id) for c in pool], dtype=np.int32))
        offsets.append(offsets[-1] + len(pool))
        dense.append(np.array([data.position(d) for d, _ in head[:TOP_K]], dtype=np.int32))
        pool_hit.append(any(c.doc_id in gold for c in pool))
        qlen.append(len(query))
        if n % 500 == 0:
            print(f"{name}: {n}/{len(ids)} pools ({time.perf_counter() - started:.0f}s)", flush=True)
    LAB.mkdir(parents=True, exist_ok=True)
    target = LAB / f"{name}.npz"
    np.savez_compressed(
        target,
        X=np.vstack(blocks).astype(np.float32),
        y=np.concatenate(labels),
        pos=np.concatenate(positions),
        offsets=np.array(offsets, dtype=np.int64),
        dense=np.stack([np.pad(d, (0, TOP_K - len(d)), constant_values=-1) for d in dense]),
        qids=np.array(ids),
        routes=np.array([routes[q] for q in ids]),
        pool_hit=np.array(pool_hit),
        feature_names=np.array(FEATURE_NAMES),
        doc_ids=np.array(data.doc_ids),
        config_hash=np.array(config.config_hash),
        overrides=np.array(json.dumps(overrides)),
    )
    print(f"wrote {target} ({time.perf_counter() - started:.0f}s)", flush=True)
    return target


# -- variants ---------------------------------------------------------------------------------------------------
def _params(overrides: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {
        **ltr.PARAMS,
        "seed": 0,
        "monotone_constraints": [MONOTONE.get(n, 0) for n in names],
        **overrides,
    }


def evaluate(dump: Path, variants: dict[str, dict[str, Any]], *, baseline: str | None) -> list[dict[str, Any]]:
    import lightgbm as lgb

    z = np.load(dump, allow_pickle=False)
    X, y, off = z["X"], z["y"], z["offsets"]
    names = [str(n) for n in z["feature_names"]]
    qids = [str(q) for q in z["qids"]]
    doc_ids = [str(d) for d in z["doc_ids"]]
    pos, dense = z["pos"], z["dense"]
    routes = [str(r) for r in z["routes"]]
    fold_of = {q: f for f, members in fold_members().items() for q in members}
    qrels = dev_qrels(qids)
    group_names = list(GROUPS)
    results = []
    for vname, spec in variants.items():
        t = time.perf_counter()
        drop = set(spec.get("drop", ()))
        cols = [i for i, n in enumerate(names) if n not in drop]
        use = [names[i] for i in cols]
        rounds = int(spec.get("rounds", ltr.DEFAULT_ROUNDS))
        dropout = float(spec.get("dropout", ltr.DROPOUT_RATE))
        rho = float(spec.get("rho", ltr.DEFAULT_RHO))
        only = spec.get("routes")  # rank only these routes with the ranker (others keep dense order)
        rng = np.random.default_rng(0)
        mats = []
        for i in range(len(qids)):
            m = X[off[i] : off[i + 1]]
            if dropout > 0 and rng.random() < dropout:
                m = mask_group(m, group_names[int(rng.integers(0, len(group_names)))])
            mats.append(m[:, cols])
        orders: dict[str, list[str]] = {}
        abstained = 0
        for fold in sorted(set(fold_of.values())):
            train_i = [i for i, q in enumerate(qids) if fold_of[q] != fold]
            ds = lgb.Dataset(
                np.vstack([mats[i] for i in train_i]),
                label=np.concatenate([y[off[i] : off[i + 1]] for i in train_i]),
                group=[int(off[i + 1] - off[i]) for i in train_i],
                feature_name=use,
                params={"verbosity": -1},
            )
            booster = lgb.train(_params(spec.get("params", {}), use), ds, num_boost_round=rounds)
            for i, q in enumerate(qids):
                if fold_of[q] != fold:
                    continue
                full = X[off[i] : off[i + 1]]
                cand = pos[off[i] : off[i + 1]]
                avail = group_availability(full)
                fired = sum(1 for g in GROUPS if avail.get(g, 0.0) > 0.0) / len(GROUPS)
                head: list[int]
                if (only and routes[i] not in only) or fired < rho:
                    head, abstained = [], abstained + (fired < rho)
                else:
                    s = booster.predict(full[:, cols])
                    head = [int(cand[j]) for j in np.argsort(-s, kind="stable")]
                seen = set(head)
                order = head + [int(p) for p in dense[i] if p >= 0 and p not in seen]
                orders[q] = [doc_ids[p] for p in order[:TOP_K]]
        run = {q: rank_derived_scores(o, TOP_K) for q, o in orders.items()}
        m = score_run(qrels, run, K_VALUES)
        ranks = []
        for q in qids:
            gold = {d for d, r in qrels[q].items() if r > 0}
            ranks.append(next((k for k, d in enumerate(orders[q], 1) if d in gold), 10**6))
        r = np.array(ranks)
        rec = {
            "variant": vname,
            "spec": spec,
            "ndcg_at_10": m["ndcg_at_10"],
            "mrr_at_10": m["mrr_at_10"],
            "recall_at_10": m.get("recall_at_10"),
            "recall_at_100": m["recall_at_100"],
            **{f"hit_at_{k}": float(np.mean(r <= k)) for k in (1, 3, 5, 10)},
            "abstained": abstained,
            "per_query_ndcg": per_query(qrels, run, "ndcg", 10),
            "seconds": round(time.perf_counter() - t, 1),
        }
        results.append(rec)
        base = next((x for x in results if x["variant"] == baseline), results[0])
        boot = paired_bootstrap(rec["per_query_ndcg"], base["per_query_ndcg"])
        rec["vs_baseline"] = boot.as_dict()
        print(
            f"{vname:<28} NDCG@10 {100 * rec['ndcg_at_10']:.2f} MRR@10 {100 * rec['mrr_at_10']:.2f} "
            f"H@1 {100 * rec['hit_at_1']:.2f} H@10 {100 * rec['hit_at_10']:.2f} R@100 {100 * rec['recall_at_100']:.2f} "
            f" Δ {boot.delta:+.2f} [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}] ({rec['seconds']}s)",
            flush=True,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--config", default="configs/dev.yaml")
    b.add_argument("--overrides", default="{}")
    b.add_argument("--artifact", default="runs/eval/p0-a72f7aa27c09/per_query.jsonl")
    e = sub.add_parser("eval")
    e.add_argument("name")
    e.add_argument("--variants", required=True, help='JSON {"name": {"params": {...}, "rounds": n, "drop": [...]}}')
    e.add_argument("--baseline", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "build":
        build(args.name, config_path=args.config, overrides=json.loads(args.overrides), artifact=args.artifact)
        return 0
    results = evaluate(LAB / f"{args.name}.npz", json.loads(args.variants), baseline=args.baseline)
    out = LAB / f"{args.name}.results.json"
    old = json.loads(out.read_text("utf-8")) if out.is_file() else []
    out.write_text(
        json.dumps(old + [{k: v for k, v in r.items() if k != "per_query_ndcg"} for r in results], indent=1), "utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
