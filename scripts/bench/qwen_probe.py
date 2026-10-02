#!/usr/bin/env python
"""Sub-corpus probe of a costly encoder against gte (dev split only; for sizing, never a gate).

Uses only corpus documents already in the encoder's vector cache (no document is embedded), picks `--queries`
dev queries whose gold document is among them (fixed seed), embeds those queries, and ranks each over the cached
sub-corpus with gte, with the probe encoder, and with per-query z-fusion of the two. Absolute numbers are easier
than full-corpus ones; the comparison between encoders on the same sub-problem is the point.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.demo_index import prepared_documents
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.core import pooled_query_vector, query_encoder_texts
from acis.eval.dev_task import dev_qrels

LAB = acis_root() / "runs" / "lab"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", default="qwen3-embedding-0.6b")
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--route", default="statement_like")
    args = parser.parse_args(argv)
    config = load_frozen_config("configs/dev.yaml").with_overrides(**{"model.encoder": args.key})
    enc = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=enc)
    corpus = apps.load_corpus()
    prepared = prepared_documents(config, [s.text for s in corpus])
    keys = [
        enc._cache_key(enc._render(t, is_query=False, route="generic"), is_query=False, route="generic")
        for t in prepared
    ]
    cached = [i for i, k in enumerate(keys) if enc.cache.path_for(k).is_file()]
    print(f"cached docs: {len(cached)}", flush=True)
    sub_docs = np.asarray(enc.encode([prepared[i] for i in cached], is_query=False), dtype=np.float32)
    ids = list(apps.dev_query_ids())
    qrels = dev_qrels(ids)
    handle = [str(s.handle) for s in corpus]
    sub_index = {handle[i]: j for j, i in enumerate(cached)}
    eligible = [q for q in ids if next(d for d, r in qrels[q].items() if r > 0) in sub_index]
    random.Random(0).shuffle(eligible)
    sample = sorted(eligible[: args.queries])
    queries = apps.load_queries()
    started = time.perf_counter()
    qv = []
    for n, q in enumerate(sample, 1):
        texts = query_encoder_texts(config, engine.normalise_query(queries[q])[0])
        qv.append(pooled_query_vector(np.asarray(enc.encode(list(texts), is_query=True, route=args.route))))
        if n % 50 == 0:
            print(f"queries {n}/{len(sample)} {time.perf_counter() - started:.0f}s", flush=True)
    qv = np.stack(qv)
    gz = np.load(LAB / "vec.gte-modernbert-base.npz")
    gq = gz["queries"][[ids.index(q) for q in sample]]
    gd = gz["docs"][cached]
    gold = np.array([sub_index[next(d for d, r in qrels[q].items() if r > 0)] for q in sample])

    def summary(s: np.ndarray) -> dict[str, float]:
        g = s[np.arange(len(gold)), gold][:, None]
        r = (s > g).sum(1) + 1
        return {
            "ndcg_at_10": 100 * float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()),
            "mrr_at_10": 100 * float(np.where(r <= 10, 1 / r, 0).mean()),
            "R@1": 100 * float((r <= 1).mean()),
            "R@10": 100 * float((r <= 10).mean()),
            "R@100": 100 * float((r <= 100).mean()),
        }

    def zs(s: np.ndarray) -> np.ndarray:
        return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)

    sg, sp = gq @ gd.T, qv @ sub_docs.T
    report = {
        "n_docs": len(cached),
        "n_queries": len(sample),
        "seconds_per_query": (time.perf_counter() - started) / len(sample),
        "gte": summary(sg),
        args.key: summary(sp),
    }
    for w in (0.3, 0.5, 1.0, 2.0):
        report[f"z_gte+{w}x"] = summary(zs(sg) + w * zs(sp))
    print(json.dumps(report, indent=1), flush=True)
    (LAB / f"probe.{args.key}.json").write_text(json.dumps(report, indent=1))
    np.savez(LAB / f"probe.{args.key}.npz", queries=qv, qids=np.array(sample), cached=np.array(cached))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
