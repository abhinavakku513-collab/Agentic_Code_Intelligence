#!/usr/bin/env python
"""Dense-only encoder comparison on a fixed dev sample over a shared sub-corpus (spec 02 §3; owner request).

Embedding the full 8,765-document corpus with a 0.6B model takes hours on this CPU, so every model is measured on
the *same* reduced problem: `--queries` dev queries (fixed seed), ranked over a sub-corpus made of their gold
documents plus `--distractors` random others. That is easier than the full task, so the absolute numbers are not
comparable with full-corpus rows (G-M) — only the models are comparable with each other, here.

Each model is used as its card says (instruction, pooling, truncation), through the engine's own document and
query preparation, and scored with our metric code. One `dev` ledger row per model, with its measured throughput.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.demo_index import prepared_documents
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.core import pooled_query_vector, query_encoder_texts
from acis.eval import ladder
from acis.eval.metrics import score_run


def measure(key: str, sample: list[str], docs: dict[str, str], qrels: Any, config_path: str) -> dict[str, Any]:
    config = load_frozen_config(config_path).with_overrides(**{"model.encoder": key, "run.channel": "dense"})
    encoder = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=encoder)
    doc_ids = sorted(docs)
    prepared = prepared_documents(config, [docs[d] for d in doc_ids])
    started = time.perf_counter()
    d_vecs = np.asarray(encoder.encode(prepared), dtype=np.float32)
    doc_seconds = time.perf_counter() - started

    queries = apps.load_queries()
    started = time.perf_counter()
    q_vecs = []
    for qid in sample:
        texts = query_encoder_texts(config, engine.normalise_query(queries[qid])[0])
        # APPS dev queries are problem statements: the statement route, and its instruction where the card has one.
        vectors = np.asarray(encoder.encode(list(texts), is_query=True, route="statement_like"), dtype=np.float32)
        q_vecs.append(pooled_query_vector(vectors))
    query_seconds = time.perf_counter() - started

    scores = np.stack(q_vecs) @ d_vecs.T
    run = {}
    for i, qid in enumerate(sample):
        top = np.argsort(-scores[i], kind="stable")[:100]
        run[qid] = {doc_ids[j]: float(scores[i, j]) for j in top}
    metrics = score_run(qrels, run)
    return {
        "model": key,
        "name": encoder.name,
        "params": getattr(encoder, "n_parameters", None),
        "ndcg_at_10": metrics["ndcg_at_10"],
        "mrr_at_10": metrics["mrr_at_10"],
        "recall_at_100": metrics["recall_at_100"],
        "metrics": metrics,
        "docs_per_second": round(len(doc_ids) / max(doc_seconds, 1e-9), 3),
        "queries_per_second": round(len(sample) / max(query_seconds, 1e-9), 3),
        "note": "cached vectors make throughput look faster than a cold encode" if doc_seconds < 1 else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="gte-modernbert-base,qwen3-embedding-0.6b")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--distractors", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    qrels_all = apps.load_qrels()
    sample = sorted(rng.sample(sorted(qrels_all), args.queries))
    qrels = {q: dict(qrels_all[q]) for q in sample}
    corpus = {d.doc_id: d.mteb_text for d in apps.load_documents()}
    gold = {d for q in sample for d in qrels[q]}
    others = sorted(set(corpus) - gold)
    chosen = gold | set(rng.sample(others, args.distractors))
    docs = {d: corpus[d] for d in chosen}
    print(
        f"{len(sample)} queries over {len(docs)} documents ({len(gold)} gold + {args.distractors} distractors)",
        flush=True,
    )

    results = []
    for key in [k for k in args.models.split(",") if k]:
        print(f"{key}: measuring …", flush=True)
        r = measure(key, sample, docs, qrels, args.config)
        results.append(r)
        print(
            f"{key}: NDCG@10 {100 * r['ndcg_at_10']:.2f}  MRR@10 {100 * r['mrr_at_10']:.2f}  "
            f"R@100 {100 * r['recall_at_100']:.2f}  {r['docs_per_second']} docs/s  "
            f"{r['queries_per_second']} queries/s",
            flush=True,
        )
        if not args.no_ledger:
            result = ladder.LadderResult(
                rung=f"encoder-compare:{key}",
                run={},
                metrics=r["metrics"],
                seconds=0.0,
                n_queries=len(sample),
                n_docs=len(docs),
                notes=f"dense only; {len(sample)} dev queries over a {len(docs)}-document sub-corpus (gold + "
                f"{args.distractors} distractors, seed {args.seed}); comparable across models, not with G-M rows",
            )
            r["ledger_run_id"] = ladder.record(
                result,
                kind="dev",
                extra={
                    "model": key,
                    "docs_per_second": r["docs_per_second"],
                    "queries_per_second": r["queries_per_second"],
                },
            )
            print(f"{key}: recorded {r['ledger_run_id']}", flush=True)
    out = acis_root() / "runs" / "encoder_compare.json"
    out.write_text(
        json.dumps(
            {
                "queries": len(sample),
                "documents": len(docs),
                "seed": args.seed,
                "results": [{k: v for k, v in r.items() if k != "metrics"} for r in results],
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
