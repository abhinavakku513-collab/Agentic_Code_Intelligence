#!/usr/bin/env python
"""Re-key cached vectors from the pre-split prep hash (whole `prep` section) to the per-side prep hashes.

The G-M bake-off cached vectors when the prep identity covered the whole `prep` section; the factory now keys each
side on its own settings. The *settings* are the same, so the vectors are the same: this copies each corpus and
dev-query vector from its old key to its new key (never overwriting) instead of re-embedding hours of text. A 1 %
sample is recomputed first and must match at cosine ≥ 0.9999 (`acis.embed.cache.audit_sample`'s threshold).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.hashing import hash_obj, short
from acis.embed.demo_index import prepared_documents
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.core import query_encoder_texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key")
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args(argv)
    config = load_frozen_config(args.config).with_overrides(**{"model.encoder": args.key})
    new = build_encoder(config)
    whole = short(hash_obj(config.as_dict()["prep"]), 12)
    old = replace(new, prep_hash=whole, query_prep_hash=whole)
    engine = AcisEngine.from_config(config, encoder=new)
    corpus = apps.load_corpus()
    items = [(t, False) for t in prepared_documents(config, [s.text for s in corpus])]
    queries = apps.load_queries()
    for qid in apps.dev_query_ids():
        items += [(t, True) for t in query_encoder_texts(config, engine.normalise_query(queries[qid])[0])]

    def keys(text: str, is_query: bool) -> tuple[str, str]:
        rendered = new._render(text, is_query=is_query, route="generic")
        return (
            old._cache_key(rendered, is_query=is_query, route="generic"),
            new._cache_key(rendered, is_query=is_query, route="generic"),
        )

    sample = random.Random(0).sample(items, max(1, len(items) // 200))
    worst = 1.0
    for text, is_query in sample:
        k_old, _ = keys(text, is_query)
        cached = new.cache.get(k_old)
        if cached is None:
            continue
        fresh = new._forward([new._render(text, is_query=is_query, route="generic")])[0]
        cos = float(np.dot(cached, fresh) / (np.linalg.norm(cached) * np.linalg.norm(fresh)))
        worst = min(worst, cos)
    print(f"audit: {len(sample)} sampled, worst cosine {worst:.6f}", flush=True)
    if worst < 0.9999:
        raise SystemExit("audit failed: cached vectors do not match a fresh encode")
    copied = present = absent = 0
    for text, is_query in items:
        k_old, k_new = keys(text, is_query)
        if new.cache.path_for(k_new).is_file():
            present += 1
            continue
        vector = new.cache.get(k_old)
        if vector is None:
            absent += 1
            continue
        new.cache.put(k_new, vector)
        copied += 1
    print(f"copied {copied}, already present {present}, absent {absent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
