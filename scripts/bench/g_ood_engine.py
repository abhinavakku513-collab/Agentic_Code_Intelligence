#!/usr/bin/env python
"""Gate G-OOD perturbation suite through the **served** pipeline (docs/spec/10 §5).

`g_ood.py` measured Mode A through `ltr_build`'s re-implementation, which sends every non-statement query to the
plain dense order. The served pipeline no longer does that — the generic route is weighted fusion — so its rows
describe a pipeline nobody runs. This script measures the same families, sample, seed and limits through
`AcisEngine._rank_one` (the function behind `search()`), out of fold exactly as `eval_pipeline.py` does: fold
rankers trained on the other folds' **unperturbed** queries, fold routing banks, τ re-derived per bank.

    extra_drop(q) = [NDCG@10_sys(orig) − NDCG@10_sys(perturbed)] − [NDCG@10_base(orig) − NDCG@10_base(perturbed)]
    pass:  mean extra_drop ≤ limit  (format_noise 0.0, strip_headings and lower 0.3, others 1.0 pt), with its CI

`base` is the dense channel (the frozen encoder, Mode B's ranking). Dev split only.
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
from acis.engine.routing import QueryBank
from acis.eval import ladder
from acis.eval.bootstrap import paired_bootstrap
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import per_query
from acis.eval.splits import fold_members
from acis.rank import ltr
from acis.rank.compose import rank_derived_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(acis_root() / "tests" / "robustness"))
import perturb  # noqa: E402
from eval_pipeline import calibrated_tau  # noqa: E402
from g_ood import FAMILIES  # noqa: E402
from ltr_build import build_groups  # noqa: E402

TOP_K = 100


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
    snapshot = engine.build_snapshot(apps.load_corpus(), source="serve:p0")
    data = engine.snapshot_data(snapshot)
    data.warm_features()
    ids = list(apps.dev_query_ids())
    fold_of = {q: f for f, members in fold_members().items() for q in members}

    started = time.perf_counter()
    groups, _, _, _ = build_groups(
        engine,
        snapshot,
        ids,
        dense_k=int(config.get("retrieve.dense_k", 100)),
        lexical_k=int(config.get("retrieve.lexical_k", 30)),
        cap=int(config.get("retrieve.union_cap", 100)),
    )
    models = {
        fold: ltr.train([g for g in groups if fold_of[g.query_id] != fold], rounds=args.rounds, seed=0)
        for fold in sorted(set(fold_of.values()))
    }
    print(f"fold rankers trained on unperturbed queries in {time.perf_counter() - started:.0f}s", flush=True)

    full_bank = engine.query_bank
    coverage = float(
        json.loads((acis_root() / "artifacts/route/route_calibration.json").read_text("utf-8"))["coverage_target"]
    )
    sample = sorted(random.Random(args.seed).sample(ids, args.sample))
    qrels = dev_qrels(sample)
    queries = apps.load_queries()
    families = [f for f in args.families.split(",") if f]
    texts = {"original": {q: queries[q] for q in sample}}
    for family in families:
        op = getattr(perturb, family)
        texts[family] = {q: op(queries[q], seed=args.seed) for q in sample}

    runs: dict[str, dict[str, dict[str, dict[str, float]]]] = {v: {"system": {}, "base": {}} for v in texts}
    routed: dict[str, int] = dict.fromkeys(texts, 0)
    started = time.perf_counter()
    for fold, model in models.items():
        engine._ranker, engine._ranker_loaded = model, True
        if full_bank is not None:
            keep = np.array([fold_of.get(q) != fold for q in ids])
            engine._bank, engine._bank_loaded = QueryBank(matrix=full_bank.matrix[keep], k=full_bank.k), True
            tau = calibrated_tau(engine._bank.matrix, k=full_bank.k, coverage=coverage)
            engine.config = config.with_overrides(**{"route.tau": tau})
        for variant, by_query in texts.items():
            for qid in (q for q in sample if fold_of[q] == fold):
                text = by_query[qid]
                decision = engine.route_decision(engine.normalise_query(text)[0])
                routed[variant] += decision.route == "statement_like"
                system = engine._rank_one(data, text, top_k=TOP_K, strict=False, route=decision.route)
                base = engine._rank_one(data, text, top_k=TOP_K, strict=False, channel="dense")
                runs[variant]["system"][qid] = rank_derived_scores([d for d, _ in system], TOP_K)
                runs[variant]["base"][qid] = rank_derived_scores([d for d, _ in base], TOP_K)
        print(f"fold {fold} done ({time.perf_counter() - started:.0f}s)", flush=True)

    scores = {v: {s: per_query(qrels, runs[v][s], "ndcg", 10) for s in ("system", "base")} for v in texts}
    report: dict[str, Any] = {
        "sample": args.sample,
        "seed": args.seed,
        "config": args.config,
        "config_hash": config.config_hash,
        "protocol": "engine path, out of fold",
        "original": {s: 100 * float(np.mean(list(scores["original"][s].values()))) for s in ("system", "base")},
        "families": {},
    }
    so, bo = scores["original"]["system"], scores["original"]["base"]
    for family in families:
        sp, bp = scores[family]["system"], scores[family]["base"]
        extra = {q: (so[q] - sp[q]) - (bo[q] - bp[q]) for q in sample}
        boot = paired_bootstrap(extra, dict.fromkeys(sample, 0.0))
        limit = FAMILIES.get(family, 1.0)
        entry = {
            "drop_base_pts": 100 * float(np.mean([bo[q] - bp[q] for q in sample])),
            "drop_system_pts": 100 * float(np.mean([so[q] - sp[q] for q in sample])),
            "ndcg_system_perturbed": 100 * float(np.mean(list(sp.values()))),
            "ndcg_base_perturbed": 100 * float(np.mean(list(bp.values()))),
            "extra_drop_pts": boot.delta,
            "ci": [boot.ci_low, boot.ci_high],
            "limit_pts": limit,
            "passes": bool(boot.delta <= limit),
            "routed_to_ranker": routed[family],
        }
        report["families"][family] = entry
        print(
            f"{family:22s} base -{entry['drop_base_pts']:.2f}  system -{entry['drop_system_pts']:.2f}  "
            f"(perturbed: system {entry['ndcg_system_perturbed']:.2f} vs base {entry['ndcg_base_perturbed']:.2f})  "
            f"extra {boot.delta:+.2f} [{boot.ci_low:+.2f}, {boot.ci_high:+.2f}] limit {limit}  "
            f"ranker {routed[family]}/{args.sample}  {'PASS' if entry['passes'] else 'FAIL'}",
            flush=True,
        )
    report["passes"] = all(e["passes"] for e in report["families"].values())
    out = acis_root() / "runs" / "g_ood"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report_engine.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"original: system {report['original']['system']:.2f} base {report['original']['base']:.2f}")
    print(f"G-OOD perturbation suite (engine path): {'PASS' if report['passes'] else 'FAIL'}")

    if args.record:
        for family, e in report["families"].items():
            result = ladder.LadderResult(
                rung=f"G-OOD-engine:{family}",
                run={},
                metrics={
                    "extra_drop_pts": e["extra_drop_pts"],
                    "drop_base_pts": e["drop_base_pts"],
                    "drop_system_pts": e["drop_system_pts"],
                },
                seconds=0.0,
                n_queries=args.sample,
                n_docs=snapshot.n_units,
                notes="G-OOD family through the served pipeline (engine path, OOF) vs the dense channel; dev only",
            )
            print(
                family,
                ladder.record(result, kind="dev", extra={"gate": "G-OOD", "config_hash": config.config_hash, **e}),
            )
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
