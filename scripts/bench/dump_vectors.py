#!/usr/bin/env python
"""Dump one encoder's corpus and dev-query matrices to `runs/lab/vec.<key>.npz` (dev split only).

Vectors come through the encoder runtime and its content-addressed cache, so nothing is recomputed that was
already embedded under the same identity. `--legacy-prep` keys on the pre-split prep hash (the whole `prep`
section), which is how the G-M bake-off cached the granite vectors.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.hashing import hash_obj, short
from acis.core.paths import acis_root
from acis.embed.demo_index import prepared_documents
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.core import pooled_query_vector, query_encoder_texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--legacy-prep", action="store_true")
    parser.add_argument("--route", default="statement_like")
    parser.add_argument("--queries-only", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="config override key=value (YAML scalar)")
    parser.add_argument("--tag", default="", help="suffix for the output file")
    args = parser.parse_args(argv)
    import yaml  # noqa: PLC0415

    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    config = load_frozen_config(args.config).with_overrides(**{"model.encoder": args.key, **overrides})
    encoder = build_encoder(config)
    if args.legacy_prep:
        whole = short(hash_obj(config.as_dict()["prep"]), 12)
        encoder = replace(encoder, prep_hash=whole, query_prep_hash=whole)
    engine = AcisEngine.from_config(config, encoder=encoder)
    started = time.perf_counter()
    corpus = apps.load_corpus()
    out = acis_root() / "runs" / "lab" / f"vec.{args.key}{args.tag}.npz"
    docs = None
    if not args.queries_only:
        prepared = prepared_documents(config, [s.text for s in corpus])
        docs = np.asarray(encoder.encode(prepared, is_query=False), dtype=np.float32)
        print(f"docs {docs.shape} {time.perf_counter() - started:.0f}s", flush=True)
    queries = apps.load_queries()
    ids = list(apps.dev_query_ids())
    rows = []
    for n, qid in enumerate(ids, 1):
        texts = query_encoder_texts(config, engine.normalise_query(queries[qid])[0])
        rows.append(pooled_query_vector(np.asarray(encoder.encode(list(texts), is_query=True, route=args.route))))
        if n % 500 == 0:
            print(f"queries {n}/{len(ids)} {time.perf_counter() - started:.0f}s", flush=True)
    np.savez(
        out,
        docs=docs if docs is not None else np.zeros((0,)),
        queries=np.stack(rows).astype(np.float32),
        qids=np.array(ids),
        doc_ids=np.array([str(s.handle) for s in corpus]),
        cache=np.array(str(encoder.cache.stats if encoder.cache else {})),
    )
    print(f"wrote {out} cache={encoder.cache.stats if encoder.cache else None}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
