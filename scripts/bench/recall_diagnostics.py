#!/usr/bin/env python
"""Candidate recall per retrieval channel on the dev split — diagnostics, not the P0 metric.

For each channel (dense encoder(s), BM25, exact symbols) and for the served union: Recall@{10,50,100,200,500,1000}
of the dev-labelled relevant unit, plus each channel's **marginal** contribution to the union (how many relevant
units only that channel brought in). Uses the engine's own channels and `candidate_pool`. One `dev` ledger row.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.base import exact_search
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ledger
from acis.eval.dev_task import dev_qrels
from acis.lexical.symbols import query_symbols
from acis.prep.normalize import lexical_view

KS = (10, 50, 100, 200, 500, 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)
    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    data = engine.snapshot_data(engine.build_snapshot(apps.load_corpus(), source="recall"))
    ids = list(apps.dev_query_ids())
    qrels = dev_qrels(ids)
    queries = apps.load_queries()

    ranks: dict[str, list[int | None]] = {"dense": [], "bm25": [], "symbols": [], "dense2": []}
    in_pool: list[bool] = []
    only: dict[str, int] = {"dense": 0, "bm25": 0, "symbols": 0, "dense2": 0}
    pool_sizes: list[int] = []
    with_symbols = 0
    for qid in ids:
        query, _ = engine.normalise_query(queries[qid])
        gold = {d for d, r in qrels[qid].items() if r > 0}
        route = engine.route(query)
        vector = engine._query_vector(data.snapshot.snapshot_id, query, route=route)
        scores = exact_search(vector.reshape(1, -1), data.vectors)[0]
        order = np.argsort(-scores, kind="stable")[: max(KS)]
        dense_ids = [data.doc_ids[i] for i in order]
        bm25_ids = [d for d, _ in data.lexical.search_one(lexical_view(query), max(KS))]
        symbols = query_symbols(query)
        with_symbols += bool(symbols)
        sym_ids = [d for d, _ in data.symbols.search(symbols, max(KS))] if symbols else []
        lists = {"dense": dense_ids, "bm25": bm25_ids, "symbols": sym_ids}
        if data.aux_vectors is not None and engine.aux_encoder is not None:
            v2 = engine._aux_query_vector(data.snapshot.snapshot_id, query, route=route)
            s2 = exact_search(v2.reshape(1, -1), data.aux_vectors)[0]
            lists["dense2"] = [data.doc_ids[i] for i in np.argsort(-s2, kind="stable")[: max(KS)]]
        for name, items in lists.items():
            ranks[name].append(next((i for i, d in enumerate(items, 1) if d in gold), None))
        _, pool = engine.candidate_pool(data, query, route=route, want=100)
        members = {c.doc_id for c in pool}
        pool_sizes.append(len(pool))
        hit = bool(members & gold)
        in_pool.append(hit)
        if hit:
            attrs = {"dense": "dense_rank", "bm25": "lexical_rank", "dense2": "aux_rank", "symbols": "symbol_rank"}
            contributors = {n for n, a in attrs.items() for c in pool if c.doc_id in gold and getattr(c, a)}
            if len(contributors) == 1:
                only[next(iter(contributors))] += 1

    report: dict[str, object] = {
        "n_queries": len(ids),
        "queries_with_symbols": with_symbols,
        "union_cap": int(config.get("retrieve.union_cap")),
        "mean_pool_size": float(np.mean(pool_sizes)),
        "channels": {},
    }
    for name, rs in ranks.items():
        if not rs:
            continue
        report["channels"][name] = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in rs])) for k in KS}
    report["union_recall"] = float(np.mean(in_pool))
    report["only_this_channel_found_the_relevant_unit"] = only
    print(json.dumps(report, indent=1))
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="dev")
            .with_metrics(
                {
                    "union_recall": report["union_recall"],
                    **{f"{n}_{k}": v for n, c in report["channels"].items() for k, v in c.items()},
                }
            )
            .with_fields(rung="recall-diagnostics", config=args.config, config_hash=config.config_hash, **report)
            .build()
        )
        print("ledger:", ledger.append(row).run_id)
    (acis_root() / "runs" / "recall_diagnostics.json").write_text(json.dumps(report, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
