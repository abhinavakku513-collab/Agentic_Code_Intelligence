#!/usr/bin/env python
"""Stage-by-stage traces of individual queries through the served pipeline (diagnostics, not a benchmark).

For each query: route, category, symbols; the rank of each relevant unit in every channel's own top list (dense,
BM25, exact symbols), whether it entered the candidate pool, and its final rank. "Relevant" is the dev label for
APPS dev queries, and — for the Dijkstra queries reported from the page — the units that define `dijkstra`, a
diagnostic set by construction (no label, nothing tuned on it). Written to `runs/traces/traces.json`.
"""

from __future__ import annotations

import json

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.core.types import SearchRequest
from acis.embed.base import exact_search
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval.dev_task import dev_qrels
from acis.lexical.symbols import query_symbols
from acis.prep.normalize import lexical_view

DEEP = 1000


def trace(engine: AcisEngine, data, query_text: str, relevant: set[str]) -> dict:
    response = engine.search(SearchRequest(query=query_text, top_k=100, explain=True))
    query, _ = engine.normalise_query(query_text)
    route = response.route
    vector = engine._query_vector(data.snapshot.snapshot_id, query, route=route)
    scores = exact_search(vector.reshape(1, -1), data.vectors)[0]
    dense = [data.doc_ids[i] for i in np.argsort(-scores, kind="stable")[:DEEP]]
    bm25 = [d for d, _ in data.lexical.search_one(lexical_view(query), DEEP)]
    symbols = query_symbols(query)
    sym = [d for d, _ in data.symbols.search(symbols, DEEP)] if symbols else []
    _, pool = engine.candidate_pool(data, query, route=route, want=100)
    pool_ids = {c.doc_id for c in pool}
    final = [h.unit.key for h in response.results]

    def rank(items: list[str], doc: str) -> int | None:
        return items.index(doc) + 1 if doc in items else None

    return {
        "query_head": query_text[:160],
        "route": route,
        "category": response.explanation.get("category"),
        "ordered_by": response.explanation.get("ordered_by"),
        "symbols": list(symbols),
        "confidence": response.confidence,
        "no_strong_match": response.no_strong_match,
        "pool_size": len(pool),
        "relevant": {
            d: {
                "cosine": round(float(scores[data.position(d)]), 4),
                "dense_rank": rank(dense, d),
                "bm25_rank": rank(bm25, d),
                "symbol_rank": rank(sym, d),
                "in_pool": d in pool_ids,
                "final_rank": rank(final, d),
            }
            for d in sorted(relevant)
        },
        "final_top3": final[:3],
        "top_cosines": [round(float(x), 4) for x in np.sort(scores)[::-1][:10]],
    }


def main() -> int:
    config = load_frozen_config("configs/dev.yaml")
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    engine.build_snapshot(apps.load_corpus(), source="serve:p0")
    data = next(iter(engine._snapshots.values()))
    dijkstra = {d for d in data.doc_ids if "def dijkstra(" in data.text_of(d)}
    cases = [
        ("implement Dijkstra shortest path using a priority queue", dijkstra),
        ("find the shortest path in a weighted graph", dijkstra),
        ("dijkstra", dijkstra),
    ]
    artifact = max((acis_root() / "runs" / "eval").glob("p0-*/per_query.jsonl"), key=lambda p: p.stat().st_mtime)
    records = [json.loads(line) for line in artifact.read_text("utf-8").splitlines() if line]
    misses = [r for r in records if r["rank_full"] is None][:2] + [r for r in records if (r["rank_full"] or 0) > 10][:1]
    queries = apps.load_queries()
    qrels = dev_qrels([r["query_id"] for r in misses])
    for r in misses:
        cases.append((queries[r["query_id"]], {d for d, v in qrels[r["query_id"]].items() if v > 0}))
    out = [trace(engine, data, q, rel) for q, rel in cases]
    target = acis_root() / "runs" / "traces" / "traces.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1), "utf-8")
    for t in out:
        best = min((v["final_rank"] or 999 for v in t["relevant"].values()), default=None)
        print(
            f"{t['query_head'][:70]!r}: route={t['route']} category={t['category']} pool={t['pool_size']} "
            f"best relevant final rank={best if best != 999 else 'not in top 100'}"
        )
        for d, v in list(t["relevant"].items())[:6]:
            print(f"    {d}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
