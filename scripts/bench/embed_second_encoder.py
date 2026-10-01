#!/usr/bin/env python
"""Embed the P0 corpus and the dev queries with the second dense encoder on the CPU (reference profile).

Resumable by construction: every vector lands in the content-addressed cache as soon as its batch finishes, so a
killed run (or a reboot) resumes where it stopped, and `scripts/train/gpu_embed.py` skips whatever is here. The
corpus goes first (every experiment needs it), then each dev query under the route the router gives it (the
second encoder is route-sensitive: its instruction differs per route, INV-15).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.demo_index import prepared_documents
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.engine.core import query_encoder_texts

CHUNK = 64


def write_progress(model: str, name: str, **fields: object) -> None:
    """`runs/embed/<model>.progress.json`: what this job has actually encoded so far. The service reads it to say
    whether the second encoder is serving, still indexing, or absent — a count, never an estimate."""
    target = acis_root() / "runs" / "embed" / f"{model}.progress.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"model": model, "card": name, "updated": time.time(), **fields}, indent=1), "utf-8")
    os.replace(tmp, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-embedding-0.6b")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument(
        "--token-budget", type=int, default=4096, help="tokens per forward batch: bounds activation memory on CPU"
    )
    args = parser.parse_args(argv)

    base = load_frozen_config(args.config)
    config = base.with_overrides(**{"model.encoder": args.model})
    encoder = build_encoder(config)
    encoder.token_budget = args.token_budget  # batching only; vectors are keyed without it

    docs = prepared_documents(config, [d.mteb_text for d in apps.load_documents()])
    started = time.perf_counter()
    for i in range(0, len(docs), CHUNK):
        encoder.encode(docs[i : i + CHUNK], is_query=False)
        write_progress(
            args.model, encoder.name, stage="corpus", corpus_done=min(i + CHUNK, len(docs)), corpus_total=len(docs)
        )
        print(f"corpus {min(i + CHUNK, len(docs))}/{len(docs)} ({time.perf_counter() - started:.0f}s)", flush=True)

    # The router (the primary encoder) is loaded only now: during the long corpus pass it would hold ~1.5 GB for
    # nothing, and memory on the reference host is what limits running anything else alongside this job.
    router = AcisEngine.from_config(base.with_overrides(**{"model.aux_encoder": ""}), encoder=build_encoder(base))
    queries = apps.load_queries()
    ids = list(apps.dev_query_ids())
    started = time.perf_counter()
    # Every dev query under the statement route first (the route APPS statements take, and the one the ranker is
    # trained on), then the generic route for the queries the router sends there: an experiment can start on the
    # first pass while the second runs, and an out-of-fold re-route finds its vector already cached.
    plan = [(qid, "statement_like") for qid in ids]
    for qid in ids:
        normalised, _ = router.normalise_query(queries[qid])
        if router.route(normalised) != "statement_like":
            plan.append((qid, str(router.route(normalised))))
    for n, (qid, route) in enumerate(plan, start=1):
        normalised, _ = router.normalise_query(queries[qid])
        encoder.encode(list(query_encoder_texts(config, normalised)), is_query=True, route=route)
        if n % 50 == 0 or n == len(plan):
            write_progress(
                args.model,
                encoder.name,
                stage="queries",
                corpus_done=len(docs),
                corpus_total=len(docs),
                queries_done=n,
                queries_total=len(plan),
            )
            print(f"queries {n}/{len(plan)} ({time.perf_counter() - started:.0f}s)", flush=True)
    print("done", encoder.stats(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
