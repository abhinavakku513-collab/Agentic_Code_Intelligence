#!/usr/bin/env python
"""Arbitrary-query stress run over the P0 corpus, through `engine.search()` — the call the API makes (spec 10 §1).

Judges will type whatever they like. For each family below the run records whether the page would show a ranked,
non-empty list, which route and stage ordered it, the confidence the engine reported, the latency, and the top three
units with their first line, so a person can judge "sensible" by reading them. Nothing is tuned on these queries
(R-Q4): they are fixed here, run once per pipeline state, and reported.

Two families carry an objective check, both diagnostics rather than benchmarks:
* **statement / paraphrase** — a dev (TRAIN) problem, verbatim and paraphrased; the rank of its dev-labelled
  solution is reported.
* **dijkstra** — the best rank reached by any corpus unit that defines `def dijkstra(`: the case reported from the
  page, where such a unit sat at #6 behind unrelated programs.

Output: `runs/stress/<label>.json` and one `audit` ledger row (counts and latencies only).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.errors import AcisError
from acis.core.paths import acis_root
from acis.core.types import SearchRequest
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ledger
from acis.eval.dev_task import dev_qrels

STATEMENT_ID = "q215"  # a dev (TRAIN) problem: the zigzag string conversion


def families(statement: str) -> list[tuple[str, str]]:
    return [
        ("statement_verbatim", statement),
        ("statement_paraphrase_short", "write the string in a zigzag over n rows and then read it off row by row"),
        ("statement_paraphrase_casual", "hey how do i do that zig zag thing with a string and a number of rows?"),
        ("keyword_dijkstra", "dijkstra"),
        ("identifier_heapq", "heapq.heappush"),
        ("identifier_snake", "lru_cache"),
        ("nl_dijkstra", "find the shortest path in a weighted graph"),
        ("nl_dijkstra_named", "dijkstra shortest path with a priority queue"),
        ("vague_1", "make it faster"),
        ("vague_2", "something with arrays"),
        ("off_corpus_1", "configure a kubernetes ingress with TLS certificates"),
        ("off_corpus_2", "train a convolutional neural network on images with pytorch"),
        ("typos", "fnd the shortst path in a wieghted grpah"),
        (
            "unusual_phrasing",
            "Given: n nodes, m weighted edges. Want: minimal total weight from node 1 to node n. How?",
        ),
        ("mixed_nl_code", "for i in range(n): dp[i] = min(dp[i-1], dp[i-2]) + cost[i]  minimum cost climbing"),
        ("non_english", "trouver le plus court chemin dans un graphe"),
        ("empty", ""),
        ("whitespace_only", "   \n\t "),
        ("very_long_16k_plus", (statement + "\n\n") * (16_500 // max(1, len(statement)) + 2)),
    ]


def run(engine: AcisEngine, snapshot: Any, *, top_k: int = 10) -> list[dict[str, Any]]:
    data = engine.snapshot_data(snapshot)
    dijkstra_docs = {d for d in data.doc_ids if "def dijkstra(" in data.text_of(d)}
    qrels = dev_qrels([STATEMENT_ID])
    gold = {d for d, rel in qrels.get(STATEMENT_ID, {}).items() if rel > 0}
    statement = apps.load_queries()[STATEMENT_ID]
    out = []
    for family, text in families(statement):
        started = time.perf_counter()
        record: dict[str, Any] = {"family": family, "query_chars": len(text), "query_head": text[:120]}
        try:
            response = engine.search(SearchRequest(query=text, top_k=top_k, explain=True))
        except AcisError as exc:
            record.update(ok=False, error_type=type(exc).__name__, error=exc.message)
            out.append(record)
            continue
        keys = [h.unit.key for h in response.results]
        record.update(
            ok=bool(response.results),
            n_results=len(response.results),
            route=response.route,
            route_reason=response.explanation.get("route_decision", {}).get("reason"),
            ordered_by=response.explanation.get("ordered_by"),
            confidence=response.confidence,
            no_strong_match=response.no_strong_match,
            query_truncated=bool(response.interpreted_intent.get("query_truncated")),
            degradations=list(response.degradations),
            ms_total=round(response.timings_ms["total"], 1),
            ms_wall=round((time.perf_counter() - started) * 1000, 1),
            top3=[
                {
                    "key": h.unit.key,
                    "first_line": h.source.strip().split("\n")[0][:90],
                    "similarity": round(float(h.signals.get("similarity", float("nan"))), 4),
                }
                for h in response.results[:3]
            ],
        )
        if family.startswith("statement"):
            record["gold_rank"] = next((i for i, k in enumerate(keys, 1) if k in gold), None)
        if "dijkstra" in family or family in ("typos", "unusual_phrasing", "non_english"):
            record["best_dijkstra_rank"] = next((i for i, k in enumerate(keys, 1) if k in dijkstra_docs), None)
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--label", required=True, help="names the output, e.g. before-fusion / after-fusion")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)

    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    snapshot = engine.build_snapshot(apps.load_corpus(), source="serve:p0")
    engine.snapshot_data(snapshot).warm_features()
    records = run(engine, snapshot)

    for r in records:
        if not r.get("ok", False) and "error" in r:
            print(f"{r['family']:<28} typed error {r['error_type']}: {r['error']}")
            continue
        extra = ""
        if "gold_rank" in r:
            extra += f" gold@{r['gold_rank']}"
        if "best_dijkstra_rank" in r:
            extra += f" dijkstra@{r['best_dijkstra_rank']}"
        print(
            f"{r['family']:<28} n={r['n_results']:<3} {r['route']:<14} conf={r['confidence']:<6} "
            f"weak={str(r['no_strong_match']):<5} {r['ms_total']:>7.0f} ms{extra}  | {r['ordered_by']}"
        )
        for hit in r["top3"]:
            print(f"      {hit['key']:<8} {hit['similarity']:.3f}  {hit['first_line']}")

    out = acis_root() / "runs" / "stress" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"label": args.label, "config": args.config, "config_hash": config.config_hash, "records": records}
    if not args.no_ledger:
        answered = [r for r in records if "error" not in r]
        row = (
            ledger.LedgerRowBuilder(kind="audit")
            .with_metrics(
                {
                    "families": float(len(records)),
                    "non_empty_ranked": float(sum(1 for r in answered if r["ok"])),
                    "typed_errors": float(len(records) - len(answered)),
                    "weak_match_flagged": float(sum(1 for r in answered if r["no_strong_match"])),
                }
            )
            .with_fields(
                rung=f"stress:{args.label}",
                config=args.config,
                config_hash=config.config_hash,
                artifact=str(out.relative_to(acis_root())),
                records=records,
            )
            .build()
        )
        payload["ledger_run_id"] = ledger.append(row).run_id
        print(f"\nledger: {payload['ledger_run_id']}")
    out.write_text(json.dumps(payload, indent=1, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
