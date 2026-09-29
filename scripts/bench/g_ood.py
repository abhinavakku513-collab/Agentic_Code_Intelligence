#!/usr/bin/env python
"""Gate G-OOD, perturbation suite: is Mode A more brittle than the frozen dense base? (docs/spec/10 §5)

For every perturbation family F in `tests/robustness/perturb.py`, on a fixed sample of dev queries:

    drop(S, F) = NDCG@10(S, original query) - NDCG@10(S, perturbed query), per query
    criterion:  mean[drop(system) - drop(base)] <= 1.0 pt   (mild families: format_noise 0.0, strip_headings and
                lower 0.3 pt), reported with its paired-bootstrap CI

`base` is the frozen dense ranking. `system` is Mode A exactly as served — routing v1.1, the ranker only where its
evidence fired, the dense order otherwise — and **out of fold**: five rankers are trained on the *unperturbed* dev
queries of the other folds, and each query, perturbed or not, is scored by the model that never saw it.

Dev split only. The report goes to `runs/g_ood/report.json`; `--record` appends one gate row per family.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ladder
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import per_query
from acis.eval.splits import fold_members
from acis.rank import ltr
from acis.rank.compose import rank_derived_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(acis_root() / "tests" / "robustness"))
import ltr_build  # noqa: E402
import perturb  # noqa: E402

FAMILIES = {
    "format_noise": 0.0,
    "strip_headings": 0.3,
    "lower": 0.3,
    "collapse_whitespace": 1.0,
    "drop_last_paragraph": 1.0,
    "sentence_dropout": 1.0,
    "typos": 1.0,
    "shuffle_paragraphs": 1.0,
    "truncate": 1.0,
}


def shipped(
    groups: list[ltr.TrainingGroup],
    models: dict[int, Any],
    fold_of: dict[str, int],
    routes: dict[str, str],
    dense_runs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Mode A as served, each query scored by the fold model that never saw it."""
    run: dict[str, dict[str, float]] = {}
    for g in groups:
        model = models[fold_of[g.query_id]]
        if routes.get(g.query_id) != "statement_like" or model.should_abstain(g.features):
            run[g.query_id] = dense_runs[g.query_id]
            continue
        order = np.argsort(-model.score(g.features), kind="stable")
        run[g.query_id] = rank_derived_scores([g.doc_ids[i] for i in order][:100], 100)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)

    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    snapshot = engine.build_snapshot(apps.load_corpus(), source="g-ood")
    ids = list(apps.dev_query_ids())
    fold_of = {q: f for f, members in fold_members().items() for q in members}
    kw = dict(dense_k=100, lexical_k=30, cap=100)

    started = time.perf_counter()
    groups, dense_runs, _, routes = ltr_build.build_groups(engine, snapshot, ids, **kw)
    models = {}
    for fold in sorted(set(fold_of.values())):
        models[fold] = ltr.train([g for g in groups if fold_of[g.query_id] != fold], rounds=args.rounds, seed=0)
    print(f"fold rankers trained on unperturbed queries in {time.perf_counter() - started:.0f}s", flush=True)

    sample = sorted(random.Random(args.seed).sample(ids, args.sample))
    in_sample = set(sample)
    qrels = dev_qrels(sample)
    orig_groups = [g for g in groups if g.query_id in in_sample]
    base_orig = per_query(qrels, {q: dense_runs[q] for q in sample}, "ndcg", 10)
    sys_orig = per_query(qrels, shipped(orig_groups, models, fold_of, routes, dense_runs), "ndcg", 10)

    queries = apps.load_queries()
    report: dict[str, Any] = {"sample": args.sample, "seed": args.seed, "families": {}}
    for family in [f for f in args.families.split(",") if f]:
        op = getattr(perturb, family)
        texts = {q: op(queries[q], seed=args.seed) for q in sample}
        t = time.perf_counter()
        p_groups, p_dense, _, p_routes = ltr_build.build_groups(engine, snapshot, sample, texts=texts, **kw)
        base_p = per_query(qrels, p_dense, "ndcg", 10)
        sys_p = per_query(qrels, shipped(p_groups, models, fold_of, p_routes, p_dense), "ndcg", 10)
        # extra drop of the system over the base, per query; positive = the system is more brittle
        extra = {q: (sys_orig[q] - sys_p[q]) - (base_orig[q] - base_p[q]) for q in sample}
        boot = paired_bootstrap(extra, dict.fromkeys(sample, 0.0))
        limit = FAMILIES.get(family, 1.0)
        entry = {
            "drop_base_pts": 100 * float(np.mean([base_orig[q] - base_p[q] for q in sample])),
            "drop_system_pts": 100 * float(np.mean([sys_orig[q] - sys_p[q] for q in sample])),
            "extra_drop_pts": boot.delta,
            "ci": [boot.ci_low, boot.ci_high],
            "limit_pts": limit,
            "passes": bool(boot.delta <= limit),
            "routed_to_ranker": sum(1 for q in sample if p_routes.get(q) == "statement_like"),
            "seconds": round(time.perf_counter() - t, 1),
        }
        report["families"][family] = entry
        print(
            f"{family:22s} base -{entry['drop_base_pts']:.2f}  system -{entry['drop_system_pts']:.2f}  "
            f"extra {boot.delta:+.2f} [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}] limit {limit}  "
            f"{'PASS' if entry['passes'] else 'FAIL'}",
            flush=True,
        )

    report["passes"] = all(e["passes"] for e in report["families"].values())
    out = acis_root() / "runs" / "g_ood"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"G-OOD perturbation suite: {'PASS' if report['passes'] else 'FAIL'}")

    if args.record:
        for family, e in report["families"].items():
            result = ladder.LadderResult(
                rung=f"G-OOD:{family}",
                run={},
                metrics={
                    "extra_drop_pts": e["extra_drop_pts"],
                    "drop_base_pts": e["drop_base_pts"],
                    "drop_system_pts": e["drop_system_pts"],
                },
                seconds=e["seconds"],
                n_queries=args.sample,
                n_docs=8765,
                notes="G-OOD perturbation family; Mode A as shipped (out of fold) vs frozen dense; dev split only",
            )
            extra = {k: v for k, v in e.items() if k != "seconds"}
            print(family, ladder.record(result, kind="dev", extra={"gate": "G-OOD", **extra}))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
