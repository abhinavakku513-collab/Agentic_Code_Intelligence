#!/usr/bin/env python
"""Embed the P0 corpus and the dev queries with the second dense encoder on the CPU (reference profile).

Resumable by construction: every vector lands in the content-addressed cache as soon as its batch finishes, so a
killed run (or a reboot) resumes where it stopped, and `scripts/train/gpu_embed.py` skips whatever is here. The
corpus goes first (every experiment needs it), then each dev query under the route the router gives it (the
second encoder is route-sensitive: its instruction differs per route, INV-15).
"""

from __future__ import annotations

import argparse
import time

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.embed.demo_index import prepared_documents
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.core import query_encoder_texts

CHUNK = 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-embedding-0.6b")
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args(argv)

    base = load_frozen_config(args.config)
    router = AcisEngine.from_config(base.with_overrides(**{"model.aux_encoder": ""}), encoder=build_encoder(base))
    config = base.with_overrides(**{"model.encoder": args.model})
    encoder = build_encoder(config)

    docs = prepared_documents(config, [d.mteb_text for d in apps.load_documents()])
    started = time.perf_counter()
    for i in range(0, len(docs), CHUNK):
        encoder.encode(docs[i : i + CHUNK], is_query=False)
        print(f"corpus {min(i + CHUNK, len(docs))}/{len(docs)} ({time.perf_counter() - started:.0f}s)", flush=True)

    queries = apps.load_queries()
    ids = list(apps.dev_query_ids())
    started = time.perf_counter()
    for n, qid in enumerate(ids, start=1):
        normalised, _ = router.normalise_query(queries[qid])
        route = router.route(normalised)
        encoder.encode(list(query_encoder_texts(config, normalised)), is_query=True, route=route)
        if n % 50 == 0:
            print(f"queries {n}/{len(ids)} ({time.perf_counter() - started:.0f}s)", flush=True)
    print("done", encoder.stats(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
