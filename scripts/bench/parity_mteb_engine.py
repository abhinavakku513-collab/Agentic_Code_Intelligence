#!/usr/bin/env python
"""Path parity: the MTEB adapter (Mode A) and the engine rank DEV queries identically (dev split only).

Stage `mteb` runs `mteb.evaluate` with `PrePostPipelineEncoder` (Mode A) on the dev task restricted to a fixed
sample of DEV queries and keeps the predictions MTEB saved. Stage `engine` builds a fresh `AcisEngine` from the same
configuration in a separate process and ranks the same queries with `_rank_one` — the function behind the API, the UI
and the CLI. Stage `compare` checks that the two ranked lists are identical, query by query. The scores differ by
design (Mode A writes rank-derived scores), so the comparison is on the order of the returned documents.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from acis.core.paths import acis_root

OUT = acis_root() / "runs" / "parity"
TOP_K = 100


def sample(n: int) -> list[str]:
    from acis.appsdata import apps

    ids = sorted(apps.dev_query_ids())
    random.Random(0).shuffle(ids)
    return sorted(ids[:n])


def stage_mteb(config: str, ids: list[str]) -> None:
    os.environ["ACIS_CONFIG"] = config
    os.environ["ACIS_MODE"] = "A"
    import mteb

    from acis.eval.dev_task import make_dev_task
    from acis.eval.official import _load_predictions
    from acis.mteb_adapter import PrePostPipelineEncoder

    task = make_dev_task(ids)
    model = PrePostPipelineEncoder()
    folder = OUT / "mteb_predictions"
    mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
        cache=None,
        overwrite_strategy="always",
        prediction_folder=str(folder),
        show_progress_bar=False,
    )
    run = _load_predictions(folder, task)
    order = {q: [d for d, _ in sorted(r.items(), key=lambda kv: -kv[1])][:TOP_K] for q, r in run.items()}
    (OUT / "mteb_orders.json").write_text(json.dumps(order))
    print(f"mteb: {len(order)} queries", flush=True)


def stage_engine(config: str, ids: list[str]) -> None:
    from acis.appsdata import apps
    from acis.core.config import load_frozen_config
    from acis.embed.factory import build_encoder
    from acis.engine import AcisEngine

    cfg = load_frozen_config(config)
    engine = AcisEngine.from_config(cfg, encoder=build_encoder(cfg))
    data = engine.snapshot_data(engine.build_snapshot(apps.load_corpus(), source="parity"))
    queries = apps.load_queries()
    order = {q: [d for d, _ in engine._rank_one(data, queries[q], top_k=TOP_K, strict=False)] for q in ids}
    (OUT / "engine_orders.json").write_text(json.dumps(order))
    print(f"engine: {len(order)} queries", flush=True)


def stage_compare() -> int:
    a = json.loads((OUT / "mteb_orders.json").read_text())
    b = json.loads((OUT / "engine_orders.json").read_text())
    common = sorted(set(a) & set(b))
    same = [q for q in common if a[q] == b[q]]
    top10 = [q for q in common if a[q][:10] == b[q][:10]]
    report = {
        "queries": len(common),
        "identical_top100": len(same),
        "identical_top10": len(top10),
        "missing": sorted(set(a) ^ set(b))[:5],
    }
    (OUT / "parity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if len(same) == len(common) and not report["missing"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["mteb", "engine", "compare"])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sample(args.n)
    if args.stage == "mteb":
        stage_mteb(args.config, ids)
    elif args.stage == "engine":
        stage_engine(args.config, ids)
    else:
        return stage_compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
