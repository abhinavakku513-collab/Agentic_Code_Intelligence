#!/usr/bin/env python
"""Track B2 acceptance: evolution-aware retrieval against flat all-version search (docs/spec/04 §6, D12).

A history is generated with Apps-Evolve from real APPS dev solutions, so every lineage is known by construction,
and each seed's own dev statement is its query. Definitions are fixed here, before any number exists:

* **best revision** — the seed's original body: the only revision guaranteed to be an accepted solution, since
  `constant` and `branch` edits can change behaviour;
* **Evolution-NDCG@10** — gain 3 for the best revision of the query's lineage, 1 for any other body of that
  lineage, 0 otherwise; a body already ranked gains nothing again (a duplicate is not new information). The ideal
  list is the best revision followed by the lineage's other distinct bodies. A grouped answer is judged
  **expanded** — each group contributes its best revision, then its other members by score — because that is
  what it shows (`members[]`; only `expand=false` returns bare heads). A first definition judged a group by its
  head alone; a 20-query smoke run showed that makes the ideal list unattainable for any grouped answer (the
  other members' gains can never be earned), so it penalised grouping by construction. It was replaced before
  the benchmark was run;
* **Duplicate-Rate@10** — share of the top 10 answers whose true lineage already appeared above it; for grouped,
  the answers are its lineages (the first recorded run, bench-0ae8ddf9d4be, computed it on the expanded list and
  so counted a group's own members as duplicates — its grouped duplicate rate is wrong, its NDCG is not);
* **Best-Revision-Hit@1** — the top answer is the best revision of the query's lineage;
* **wrong-merge rate** — among revision pairs the cascade put in one lineage, the share whose true lineages
  differ (pairwise precision error); pairwise recall and F1 are reported with it.

Acceptance: grouped beats flat on Evolution-NDCG@10 with a paired-bootstrap CI lower bound > 0, wrong-merge
≤ 1 %, grouped Duplicate-Rate@10 ≈ 0. Dev split only; the benchmark repository is removed afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.paths import acis_home, acis_root
from acis.core.types import EvolveRequest, SourceSpec
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import appsevolve as ae
from acis.eval import ledger
from acis.eval.bootstrap import paired_bootstrap

REPO = "bonus-bench"
OPERATORS = (*ae.EDIT_OPERATORS, "rekey", "add", "delete")
K = 10


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:K]))


def evolution_ndcg(ranked: Sequence[tuple[str | None, str]], target: str, best: str, others: set[str]) -> float:
    """`ranked` is (true lineage, body hash) per position."""
    seen: set[str] = set()
    gains = []
    for lineage, body in ranked[:K]:
        gain = 0.0
        if lineage == target and body not in seen:
            gain = 3.0 if body == best else 1.0
        seen.add(body)
        gains.append(gain)
    ideal = _dcg([3.0] + [1.0] * len(others))
    return _dcg(gains) / ideal if ideal else 0.0


def duplicate_rate(ranked: Sequence[tuple[str | None, str]]) -> float:
    top = ranked[:K]
    seen: set[str | None] = set()
    dup = 0
    for lineage, _ in top:
        if lineage in seen:
            dup += 1
        seen.add(lineage)
    return dup / len(top) if top else 0.0


def pairwise(index: Any, truth: Mapping[tuple[str, str], str]) -> dict[str, float]:
    """Pairwise precision/recall of the predicted lineages against the constructed ones, over revisions."""
    predicted_pairs = wrong = 0
    for lineage in index.lineages:
        members = [(m.version, m.key) for m in lineage.members]
        for a, b in combinations(members, 2):
            predicted_pairs += 1
            if truth.get(a) != truth.get(b):
                wrong += 1
    true_groups: dict[str, list[tuple[str, str]]] = {}
    for member, lid in truth.items():
        true_groups.setdefault(lid, []).append(member)
    true_pairs = sum(len(v) * (len(v) - 1) // 2 for v in true_groups.values())
    correct = predicted_pairs - wrong
    precision = correct / predicted_pairs if predicted_pairs else 1.0
    recall = correct / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "wrong_merge_rate": wrong / predicted_pairs if predicted_pairs else 0.0,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "predicted_pairs": float(predicted_pairs),
        "true_pairs": float(true_pairs),
    }


def measure(config_path: str, *, n: int, versions: int, edits: int, seed: int) -> dict[str, Any]:
    config = load_frozen_config(config_path).with_overrides(**{"run.channel": "dense"})
    rng = random.Random(seed)
    qrels = apps.load_qrels()
    queries = apps.load_queries()
    docs = {d.doc_id: d.mteb_text for d in apps.load_documents()}
    qids = rng.sample(sorted(qrels), n + n // 5)
    chosen, spare_q = qids[:n], qids[n:]
    seed_units = [(f"u{i:04d}.py", docs[next(iter(qrels[q]))]) for i, q in enumerate(chosen)]
    spares = [docs[next(iter(qrels[q]))] for q in spare_q]
    evolution = ae.generate(
        seed_units, n_versions=versions, seed=seed, edits_per_version=edits, operators=OPERATORS, spares=spares
    )

    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    shutil.rmtree(acis_home() / "repos" / REPO, ignore_errors=True)
    started = time.perf_counter()
    engine.ingest(SourceSpec(kind="memory", location=REPO, options=evolution.as_source_options()), repo_id=REPO)
    ingest_s = time.perf_counter() - started
    index = engine.lineage_index(REPO)

    # Ground truth on the stored text (the engine normalises bodies before hashing, so hash what it stored).
    stored: dict[tuple[str, str], str] = {}
    for label in evolution.labels:
        data = engine.open_version(REPO, label)
        for key in data.doc_ids:
            stored[(label, key)] = data.hash_of[key]
    truth = {member: lid for member, lid in evolution.lineage_of.items() if member in stored}

    per_query: dict[str, dict[str, float]] = {"grouped": {}, "flat": {}}
    dup: dict[str, list[float]] = {"grouped": [], "flat": []}
    hit1: dict[str, list[float]] = {"grouped": [], "flat": []}
    started = time.perf_counter()
    for i, qid in enumerate(chosen):
        target = f"L{i:04d}"
        best = stored[("v1", f"u{i:04d}.py")]
        bodies = {h for member, h in stored.items() if truth.get(member) == target}
        others = bodies - {best}

        grouped = engine.retrieve_evolution(EvolveRequest(query=queries[qid], repo_id=REPO, top_k=K))
        g_ranked: list[tuple[str | None, str]] = []
        for g in grouped.groups:
            head = (g["best"]["version"], g["best"]["key"])
            g_ranked.append((truth.get(head), str(g["best"]["body_hash"])))
            rest = sorted(
                (m for m in g["members"] if (m["version"], m["key"]) != head), key=lambda m: -float(m["score"])
            )
            g_ranked.extend((truth.get((m["version"], m["key"])), stored[(m["version"], m["key"])]) for m in rest)
        flat = engine.retrieve_evolution(EvolveRequest(query=queries[qid], repo_id=REPO, top_k=K, flat=True))
        f_ranked = [(truth.get((str(h.unit.version_id), h.unit.key)), h.unit.body_hash) for h in flat.flat_results]
        # Duplicate rate is a property of the *answer list*: for grouped that is the lineages it returns (one
        # entry each), not the expansion used for NDCG — within a group, its own members are not duplicates.
        g_heads = [(truth.get((g["best"]["version"], g["best"]["key"])), "") for g in grouped.groups]
        for name, ranked, answers in (("grouped", g_ranked, g_heads), ("flat", f_ranked, f_ranked)):
            per_query[name][qid] = evolution_ndcg(ranked, target, best, others)
            dup[name].append(duplicate_rate(answers))
            hit1[name].append(1.0 if ranked and ranked[0] == (target, best) else 0.0)
    search_s = time.perf_counter() - started

    boot = paired_bootstrap(per_query["grouped"], per_query["flat"])
    merge = pairwise(index, truth)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    report = {
        "encoder": engine.encoder.name if engine.encoder else "none",
        "config": config_path,
        "n_queries": n,
        "versions": versions,
        "edits_per_version": edits,
        "operators": list(OPERATORS),
        "seed": seed,
        "lineages_predicted": index.size,
        "lineages_true": len(set(truth.values())),
        "evolution_ndcg10_grouped": mean(list(per_query["grouped"].values())),
        "evolution_ndcg10_flat": mean(list(per_query["flat"].values())),
        "delta_pts": boot.delta,
        "ci": [boot.ci_low, boot.ci_high],
        "duplicate_rate10_grouped": mean(dup["grouped"]),
        "duplicate_rate10_flat": mean(dup["flat"]),
        "best_revision_hit1_grouped": mean(hit1["grouped"]),
        "best_revision_hit1_flat": mean(hit1["flat"]),
        **merge,
        "ingest_seconds": round(ingest_s, 1),
        "search_seconds": round(search_s, 1),
    }
    report["passes"] = bool(
        boot.ci_low > 0 and merge["wrong_merge_rate"] <= 0.01 and report["duplicate_rate10_grouped"] <= 0.01
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bonus: grouped vs flat all-version retrieval on Apps-Evolve")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--versions", type=int, default=5)
    parser.add_argument("--edits", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--out", default="runs/bonus_evolution.json")
    args = parser.parse_args(argv)

    try:
        report = measure(args.config, n=args.n, versions=args.versions, edits=args.edits, seed=args.seed)
    finally:
        shutil.rmtree(acis_home() / "repos" / REPO, ignore_errors=True)
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="bench")
            .with_metrics(
                {
                    k: float(report[k])
                    for k in (
                        "evolution_ndcg10_grouped",
                        "evolution_ndcg10_flat",
                        "duplicate_rate10_grouped",
                        "duplicate_rate10_flat",
                        "best_revision_hit1_grouped",
                        "best_revision_hit1_flat",
                        "wrong_merge_rate",
                        "pairwise_f1",
                    )
                }
            )
            .with_fields(rung="bonus_evolution", dataset="dev", decision_set="n/a", **report)
            .build()
        )
        report["ledger_run_id"] = ledger.append(row).run_id
    out = Path(args.out)
    out = out if out.is_absolute() else acis_root() / out
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
