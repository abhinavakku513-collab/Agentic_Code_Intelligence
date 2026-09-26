#!/usr/bin/env python
"""Phase 2 acceptance: determinism across runs and thread counts, and cache hit ≡ cold (docs/spec/07 Phase 2).

Three checks on a fixed sample, with the real encoder:

* **run** — the same inputs encoded twice, cache off, same thread count: vectors must be identical;
* **threads** — the same inputs at a different thread count: cosine ≥ 0.9999 and identical top-10 rankings over
  the whole corpus (INV-6 promises equal rankings for equal threads; across thread counts the promise is the
  rankings, not bitwise vectors, since BLAS reduction order may differ);
* **cache** — vectors served by the persistent cache against the same inputs recomputed cold: cosine ≥ 0.9999
  (docs/spec/02 §3).

Dev split only. The result is written to `runs/determinism.json` and, unless `--no-ledger`, one bench row.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.numeric import apply_threads
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.eval import ledger
from acis.prep.normalize import d1
from acis.prep.truncate import head_tail

MIN_COSINE = 0.9999


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0.0] = 1.0
    return num / den


def _top10(queries: np.ndarray, docs: np.ndarray) -> list[list[int]]:
    scores = queries @ docs.T
    # Stable: equal scores ordered by index, so the comparison is about scores, not about tie handling.
    return [list(np.lexsort((np.arange(docs.shape[0]), -row))[:10]) for row in scores]


def measure(config_path: str, *, n: int, seed: int, threads: tuple[int, int]) -> dict[str, Any]:
    config = load_frozen_config(config_path)
    rng = random.Random(seed)
    queries = apps.load_queries()
    corpus = apps.load_corpus()
    q_ids = rng.sample(sorted(apps.dev_query_ids()), n)
    q_texts = [queries[q] for q in q_ids]
    # Exactly the engine's document preparation (build_snapshot + _embed_documents), so the cache keys match.
    prep = config.section("prep").get("doc", {})
    docs_all = [
        head_tail(
            d1(s.text),
            max_tokens=int(prep.get("max_tokens", 1024)),
            head=int(prep.get("head", 768)),
            tail=int(prep.get("tail", 256)),
        ).text
        for s in corpus
    ]
    d_idx = rng.sample(range(len(docs_all)), n)
    d_texts = [docs_all[i] for i in d_idx]

    cold = build_encoder(config, cache=False)
    warm = build_encoder(config, cache=True)

    started = time.perf_counter()
    apply_threads(threads[0])
    q_a = np.asarray(cold.encode(q_texts, is_query=True), dtype=np.float32)
    q_a2 = np.asarray(cold.encode(q_texts, is_query=True), dtype=np.float32)
    d_cold = np.asarray(cold.encode(d_texts), dtype=np.float32)
    apply_threads(threads[1])
    q_b = np.asarray(cold.encode(q_texts, is_query=True), dtype=np.float32)
    apply_threads(threads[0])

    # The whole corpus from the persistent cache (every vector already there; a miss is computed and counted).
    before = getattr(warm, "forward_calls", 0)
    corpus_matrix = np.asarray(warm.encode(docs_all), dtype=np.float32)
    corpus_misses = int(getattr(warm, "forward_calls", 0) - before)
    d_cached = corpus_matrix[d_idx]

    run_identical = bool(np.array_equal(q_a, q_a2))
    thread_cos = _cosines(q_a, q_b)
    cache_cos = _cosines(d_cached, d_cold)
    top_a, top_b = _top10(q_a, corpus_matrix), _top10(q_b, corpus_matrix)
    rank_identical = sum(1 for x, y in zip(top_a, top_b, strict=True) if x == y)

    return {
        "encoder": cold.name,
        "numeric_profile": str(config.get("model.numeric_profile", "cpu-fp32")),
        "config": config_path,
        "n_queries": n,
        "n_docs_sampled": n,
        "seed": seed,
        "threads": list(threads),
        "run_vectors_identical": run_identical,
        "run_max_abs_diff": float(np.max(np.abs(q_a - q_a2))),
        "thread_min_cosine": float(thread_cos.min()),
        "thread_top10_identical": rank_identical,
        "cache_min_cosine": float(cache_cos.min()),
        "cache_corpus_forward_calls": corpus_misses,
        "min_cosine_required": MIN_COSINE,
        "passes": bool(
            run_identical and thread_cos.min() >= MIN_COSINE and rank_identical == n and cache_cos.min() >= MIN_COSINE
        ),
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="determinism and cache parity with the real encoder")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", default="8,2", help="two thread counts, e.g. 8,2")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--out", default="runs/determinism.json")
    args = parser.parse_args(argv)

    t = tuple(int(x) for x in args.threads.split(","))
    report = measure(args.config, n=args.n, seed=args.seed, threads=(t[0], t[1]))
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="bench")
            .with_metrics(
                {
                    "thread_min_cosine": report["thread_min_cosine"],
                    "cache_min_cosine": report["cache_min_cosine"],
                    "thread_top10_identical_share": report["thread_top10_identical"] / report["n_queries"],
                }
            )
            .with_fields(rung="determinism", dataset="dev", decision_set="n/a", **report)
            .build()
        )
        report["ledger_run_id"] = ledger.append(row).run_id
    out = Path(args.out)
    out = out if out.is_absolute() else acis_root() / out
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
