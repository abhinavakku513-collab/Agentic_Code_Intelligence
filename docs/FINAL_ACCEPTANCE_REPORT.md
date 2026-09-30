# ACIS — final acceptance report (implementation pass of 2026-09-29/30)

Every number cites its ledger row (`runs/ledger.jsonl`, hash-chained). "Not measured" means exactly that. No held-out
TEST label was read in this pass; TEST touches used: **0 of 6**.

## A. System summary
ACIS ranks the units of a code corpus for a natural-language query, offline and on the CPU. One engine serves P0 (the
APPS corpus), P1 (a pinned version of a repository) and the Bonus (all versions, grouped by lineage). No generation,
no LLM, no HyDE and no query-time network anywhere in the scored path.

## B. Final P0 architecture (as served — one function, `AcisEngine._rank_one`, for the page and the evaluation)
```
query → normalise (q1) → route (routing v1.1: statement-like | generic) → category (reported, never a gate)
  → gte-modernbert-base dense, exact matmul (top 300)
  + code-aware BM25 (top 200)  + exact symbols (top 50)            → union cut to 500 by RRF over all channels
  → every candidate scored by every channel (exact cosine, BM25, symbol hits/coverage/IDF, channel agreement)
  → statement-like: LightGBM LambdaRank (abstains on thin evidence → dense order)
    generic: weighted fusion α·minmax(cos) + (1−α)·bm25/max, α = 0.9; a query that is one identifier → its units first
  → dense tail → evidence re-read from the content store by hash (INV-1) → calibrated confidence (report only)
```

## C. Retrieval channels actually active
| Channel | P0 | P1/Bonus | Status |
|---|---|---|---|
| gte-modernbert-base dense | yes | yes | measured |
| BM25, code tokenizer | yes | yes | measured |
| Exact symbols (shape-based query symbols, IDF inverted index) | yes | yes | measured |
| LambdaRank (statement-like route) | yes | yes | measured |
| Weighted fusion (generic route) | yes | yes | measured |
| Second dense encoder, Qwen3-Embedding-0.6B (`model.aux_encoder`) | **off** | off | **implemented — not validated** (no corpus vectors: ~3 s/unit on this CPU; GPU path owner-gated) |
| Corpus-defined symbol weight in fusion (`symbol_beta`) | off (0.0) | off | measured: no held-out gain → kept off |
| SPLADE | — | — | **not integrated**: mainstream checkpoints are CC BY-NC-SA (outside D4); the Apache-2.0 option needs a new card type |

## D. Model versions
gte-modernbert-base @ `e7f32e3c00f91d699e8c43b53106206bcc72bb22` (Apache-2.0, per-file SHA-256 pinned). Qwen3-Embedding-0.6B
@ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` (Apache-2.0, pinned; not used in the measured pipeline). Ranker
`artifacts/ranker/p0-gte-pool500-symbols.txt` (SHA-256 `441c4186…`), 29 features, 400 rounds, trained on all 5,000 dev
pools of the served pipeline.

## E. Preprocessing
Query q1 normalisation; view V0, head 768 + tail 256 tokens (prep hash `3bc1956c196f`); documents d1, head 768 + tail 256
(prep hash `68ee54e268ef`). G1 (9 cells) keeps V0/1024; the rule's cheaper choice V0/512 awaits the owner.

## F. Candidate pool
dense 300 ∪ BM25 200 ∪ symbols 50 → cap 500 (mean 482 candidates). Pool recall **97.16 %** vs 92.94 % for the previous
100-candidate union `[ledger:dev-f320c5718b54]`, `[ledger:dev-f168f31bc434]`.

## G. Fusion / ranker configuration
`rank.generic.alpha` 0.9 (CosQA valid; re-checked on the 500 pool), `symbol_beta` 0.0; ranker `p0-gte-pool500-symbols`;
routing bank `artifacts/route/dev_query_bank.npy` (τ 0.6002, ρ 0.35); confidence calibration `[ledger:dev-50c946dc1c29]`.

## H–L. P0 dev result (the number the page shows)
| System | NDCG@10 | MRR@10 | Recall@100 | Queries | Ledger |
|---|---|---|---|---|---|
| **Served pipeline** | **74.13** | **70.93** | **94.86** | 5,000 | `[ledger:dev-299a3010be5a]` |
| Dense only (same code, same queries) | 71.03 | 67.46 | 93.78 | 5,000 | `[ledger:dev-5bd4f1c50c7c]` |

Δ NDCG@10 +3.09, 95 % CI [+2.62, +3.57]; Δ MRR@10 +3.47 (paired bootstrap, 10,000 resamples). Dataset: APPS (CoIR
AppsRetrieval) TRAIN split = dev, decision set `oof_5fold_5000` (each fold ranked by a ranker and routing bank that never
saw it, τ re-derived per fold). Code `9d38d5d98147` (clean tree), config `configs/dev.yaml` hash `1750bf7449fc…`.
Ranking runtime 391 s for both systems (575 s end to end with the training groups). Per-query artifact
`runs/eval/p0-a72f7aa27c09/per_query.jsonl` (SHA-256 `a72f7aa2…`), byte-identical across two independent runs.

## M. Historical comparison (same dev queries, same metric code)
| Pipeline | NDCG@10 | MRR@10 | Ledger |
|---|---|---|---|
| G-M bake-off, gte dense | 71.03 | 67.46 | `gate-055152620da6` |
| Served pipeline at the start of this pass (100 pool, dense-only generic route) | 72.72 | 69.39 | `dev-4b81d12e6619` |
| + generic fusion, per-fold τ | 72.80 | 69.50 | `dev-fd48ec6b73e1` |
| **+ 500 union, symbols, exact scores, refit ranker (final)** | **74.13** | **70.93** | `dev-299a3010be5a` |

## N. Candidate recall diagnostics `[ledger:dev-f320c5718b54]`
| Channel | R@10 | R@50 | R@100 | R@200 | R@500 | R@1000 |
|---|---|---|---|---|---|---|
| gte dense | 82.2 | 91.1 | 93.8 | 96.1 | 97.9 | 98.9 |
| BM25 | 43.9 | 54.0 | 58.5 | 63.4 | 70.1 | 75.9 |
| exact symbols | 23.7 | 28.0 | 30.4 | 32.3 | 34.1 | 35.6 |
| union (≤ 500) | — | — | — | — | 97.2 | — |

Units found by one channel only: dense 1,595, BM25 18, symbols 2. Qwen, SPLADE: not measured.

## O. Failure-query traces (`scripts/bench/trace_queries.py`, `runs/traces/traces.json`)
- `dijkstra` → symbol channel ranks all six `def dijkstra(` units first; final ranks 1–6.
- "implement Dijkstra shortest path using a priority queue" → all six in the pool, dense ranks 16–810 (cosine
  0.53–0.64 vs top 0.79); best final rank 6. Encoder representation, not recall, IDs or fusion wiring.
- "find the shortest path in a weighted graph" → best final rank 29; 3 of 6 outside the pool (dense recall).
- Two APPS dev misses: gold at dense rank 397 and 646, absent from BM25 → not in the pool (exact dense failure; there is
  no ANN, D5). One miss: gold in the pool at dense 120, final 66 → ranker failure.

## P. G-M evidence
`configs/gates/G-M.yaml`: gte selected at 71.03 (`gate-055152620da6`); granite-r2 56.70 (`gate-b698dad7af99`);
granite-small-r2 53.95 (`gate-f02f6ced3b1d`); Qwen3-0.6B excluded on its projected cold pass (26.1 h vs the 4 h SLO).

## Q. P1 evidence
Suites green (isolation, incremental ≡ scratch, kill -9 recovery, rollback). Fixed this pass: catalog snapshots keyed per
repository (identical content in two repositories showed zero units), rollback kept two ACTIVE rows. Real git history
(`scripts/demo/real_git_check.py`): edit → 1 unit embedded, 2 reused; `git mv` → 0 embedded, lineage `moved`;
unrelated file → 1 embedded; `git revert` → 0 embedded, body back to the first revision's hash. Update time
`[ledger:bench-9153e49050de]`.

## R. Bonus evidence
Lineage recovery exact; grouping removes duplicates; grouped and flat **tie** on Evolution-NDCG@10
`[ledger:bench-52d48f680740]` — no ranking gain is claimed. Reverts are now named in timelines.

## S. UI / demo
One process (`acis serve`) serves the page and the API. Search shows route, category, channels, candidate counts,
ordering stage, calibrated confidence, stage timings, and per hit cosine/dense/BM25 ranks and the stored code. The P0
evaluation tab renders the ledger row server-side (tested identical to the ledger file) and the per-query records from
the SHA-256-pinned artifact. Repository identity is checked end to end (tested). The synthetic commit is labelled a demo
mutator. Stale-server detection via `/healthz` `api_features`.

## T. Limitations
- Generic (non-statement) queries are ranked mainly by the encoder (α 0.9): paraphrased Dijkstra reaches #6, not #1.
- The router sends 411 APPS statements to fusion where the ranker would be +1.60 [+0.62, +2.65] better
  (`dev-299a3010be5a`, routing analysis) — below the +0.5 pt system-level gate, so thresholds are unchanged.
- The ranker is measured worse than fusion on human REG queries (CosQA −3.60 [−5.57, −1.75]), so it stays off them.
- G-OOD `format_noise` fails its strict identity limit (+0.12) `[ledger:dev-e8304c8b6364]`.
- Latency on this 8-core host misses parts of spec 02 §7 `[ledger:bench-8f34183886c2]`: short cold p95 202 ms, full
  statement cold p95 1.66 s, warm long p95 38 ms (features on 500 candidates).
- The official Mode A cold run with this pipeline has not been timed; Mode B (dense) remains the primary.

## U. Not measured
Qwen dense channel; SPLADE; cross-encoder (none added); official TEST scores; peak RSS of the served pipeline beyond the
bench row (1,516 MB during `make bench`); P1 update time with the new 500 union.

## V. Recommendation on the official TEST evaluation
The dev pipeline is verified, reproducible from a clean commit, and ledgered. Before spending a touch: (1) the owner
decides G1 and G-AB (Mode A vs B on the OOF evidence above); (2) run `python -m acis.eval.final --smoke` (sealed cache);
(3) time the cold Mode A pass — gte's cold pass is projected at 2.58 h (`gate-055152620da6`) and Mode A adds BM25,
symbols and the ranker (measured in seconds per 5,000 queries here). Cost: RC0 (Mode B) = 1 of the 6 touches; RC1 (A+B)
= 2 more. Not run in this pass, by design.
