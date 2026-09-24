#!/usr/bin/env python
"""Build the TRAIN-query bank and calibrate τ (routing v1.1, docs/spec/10 §4).

τ is not a number anyone should pick by feel. It is set so that **≥ 95 % of held-out dev queries** take the
`statement_like` path — the queries the ranker was trained on must reach the ranker — and the calibration is
done out-of-bank: each query's OOD score is computed against a bank that excludes it, because a query is always
its own nearest neighbour and including it would put every score at 1.0.

The bank itself costs nothing: the dev query vectors are already in the content-addressed cache from the
bake-off, so this reads them back rather than embedding anything.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine.routing import DEFAULT_K, QueryBank

OUT = "artifacts/route"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the TRAIN-query bank and calibrate tau")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--coverage", type=float, default=0.95, help="share of dev queries that must route as statements"
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_frozen_config(args.config)
    encoder = build_encoder(config)
    ids = list(apps.dev_query_ids())
    if args.limit:
        ids = ids[: args.limit]
    queries = apps.load_queries()

    started = time.perf_counter()
    from acis.engine import AcisEngine

    engine = AcisEngine.from_config(config, encoder=encoder)
    prepared = [engine.normalise_query(queries[qid])[0] for qid in ids]
    vectors = np.asarray(encoder.encode(prepared, is_query=True), dtype=np.float32)
    print(f"bank: {vectors.shape[0]} queries in {time.perf_counter() - started:.0f}s", flush=True)

    bank = QueryBank.from_vectors(vectors, k=args.k, source="dev", n=len(ids))
    matrix = bank.matrix

    # Out-of-bank scores: a query is its own nearest neighbour, so its own row is excluded from its own score.
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    take = min(args.k, sims.shape[0] - 1)
    nearest = np.partition(sims, -take, axis=1)[:, -take:]
    scores = nearest.mean(axis=1)

    tau = float(np.quantile(scores, 1.0 - args.coverage))
    out = acis_root() / OUT
    out.mkdir(parents=True, exist_ok=True)
    bank.save(out / "dev_query_bank.npy")
    report = {
        "model": str(config.get("model.encoder")),
        "config_hash": config.config_hash,
        "n_queries": len(ids),
        "k": args.k,
        "coverage_target": args.coverage,
        "tau": round(tau, 6),
        "score_quantiles": {
            str(q): round(float(np.quantile(scores, q)), 6) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "coverage_at_tau": round(float((scores >= tau).mean()), 6),
    }
    (out / "route_calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwritten: {out / 'dev_query_bank.npy'} and route_calibration.json")
    print(f"set `route: {{bank: {OUT}/dev_query_bank.npy, tau: {tau:.4f}, k: {args.k}}}` in the config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
