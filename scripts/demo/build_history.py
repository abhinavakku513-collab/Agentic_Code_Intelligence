#!/usr/bin/env python
"""Build `apps-history`: a real versioned repository for the demo's P1 and Bonus acts (docs/spec/09 §6).

Real APPS solutions evolve across five versions through Apps-Evolve's edits (reformat, comment, rename, reorder,
constant, branch, move, add, delete), ingested through the ordinary P1 path with the configured encoder. The page
then offers it in its repository list: pin a version, compare versions, or tick "group by lineage" to see each
unit once with its history. Idempotent — an existing `apps-history` is left alone unless `--rebuild`.
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.errors import NotFound
from acis.core.types import SourceSpec
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import appsevolve as ae
from acis.store import catalog
from acis.store.snapshots import drop_repository

REPO = "apps-history"
OPERATORS = (*ae.EDIT_OPERATORS, "rekey", "add", "delete")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--units", type=int, default=400)
    parser.add_argument("--versions", type=int, default=5)
    parser.add_argument("--edits", type=int, default=40, help="edits per version")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    with catalog.open_catalog() as db:
        exists = catalog.get_repo_exists(db, REPO)
    if exists and not args.rebuild:
        print(f"{REPO}: already built (use --rebuild to regenerate)")
        return 0
    if exists:
        with contextlib.suppress(NotFound):
            drop_repository(REPO)

    rng = random.Random(args.seed)
    docs = [d for d in apps.load_documents() if 200 <= len(d.text) <= 3000]
    chosen = rng.sample(docs, args.units + args.units // 5)
    seeds = [(f"solution_{i:04d}.py", d.text) for i, d in enumerate(chosen[: args.units])]
    spares = [d.text for d in chosen[args.units :]]
    history = ae.generate(
        seeds,
        n_versions=args.versions,
        seed=args.seed,
        edits_per_version=args.edits,
        operators=OPERATORS,
        spares=spares,
    )

    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    started = time.perf_counter()
    engine.ingest(SourceSpec(kind="memory", location=REPO, options=history.as_source_options()), repo_id=REPO)
    index = engine.lineage_index(REPO)
    print(
        f"{REPO}: {len(history.labels)} versions, {args.units} units at v1, {index.size} lineages, "
        f"built in {time.perf_counter() - started:.0f}s with {engine.encoder.name if engine.encoder else 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
