# ACIS — design decisions, invariants and evaluation-integrity rules

The rules every component of ACIS is built and tested against. Source code cites them as `D<n>` (decisions) and
`INV-<n>` (invariants); the specifications in `docs/spec/` give the detail and `docs/adr/` records every change.
Where a decision was later superseded by measurement, the ADR named next to it is authoritative.

## 1. Design decisions (one line each; detail in `docs/spec/`, changes in `docs/adr/`)
| # | Decision |
|---|---|
| D1 | Task = long Python problem statements → whole programs; test 8,765 docs / 3,765 queries / one relevant doc per query. Dense quality dominates; BM25 ≈4.8 NDCG@10 [R] → auxiliary, ablation-gated (02). |
| D2 | Maximise NDCG@10/MRR under resource use. Every RC ships a **scorecard**: params, model MB, peak RSS, cold `evaluation_time`, warm time, latency, GPU none. Ties → cheaper (G-R) (03 §7). |
| D3 | Unit = whole snippet. Dense text = head 768 + tail 256 tokens; BM25/features see the full text; window-MaxP only if G1 accepts (02 §5). |
| D4 | Encoder: seed Qwen3-Embedding-0.6B; envelope ≤ ~1B params, CPU-capable, permissive licence, safetensors, no remote code, no API (others = reference only). Final = smallest within **1.0** NDCG@10 pt of the best (**3.0** pt if the best's cold pass exceeds 2 h); SLO ≤ 4 h (02 §3). |
| D5 | Exact dense search (one matmul); no ANN/vector DB below 250k vectors. |
| D6 | Dense + BM25 → ≤100 candidates → LightGBM LambdaRank (cross-fitted, OOD-guarded) → dense tail to 1,000. No cross-encoder, HyDE or LLM in the scored path. |
| D7 | Adaptation early (Phase 3): LoRA on decontaminated TRAIN pairs, merged with α-interpolation, judged on K-fold OOF; GPU only offline via `docs/GPU_HANDOFF.md`; ships only if G3 **and** G-OOD pass; else frozen base. |
| D8 | One class, two surfaces: Mode A (SearchProtocol `index/search`, full pipeline) and Mode B (honest `encode()`); never `predict()`; no basis-/score-vector tricks. **RC0 = best frozen dense, Mode B.** At RC1 primary = A iff G-AB (OOF) says so; both JSONs shipped (03 §2–3). |
| D9 | Mode-A scores are rank-derived `(top_k+1−rank)/top_k`, exactly `min(top_k, N)` entries per query (pytrec_eval collapses near-equal scores; verified, `tests/contract/test_score_ties.py`). |
| D10 | Files + SQLite + memory-mapped arrays + bm25s + NumPy; content-addressed store; immutable snapshots. No vector DB, graph DB, search server or broker (04). |
| D11 | P1: one immutable snapshot per version, content-addressed reuse, embed only unseen units, atomic activation, rollback, crash recovery, incremental ≡ from-scratch; sources JSONL/dirs/ZIP/git. Target ≤ 10 changed units searchable ≤ 30 s p95 on T-rec [E] (04). |
| D12 | Bonus: unique-content search → evidence-gated lineage alignment → lineage-level ranking (best revision + timeline). Unknown beats a guessed merge (04 §6). |
| D13 | Agent: bounded, deterministic, read-only, interactive only, off the scored path, built last; default-on only if it passes its gate (05 §1). |
| D14 | Python only (`ast` + tolerant fallback). No JS/TS, tree-sitter, call graphs (FAQ). |
| D15 | Offline + CPU: no keys, paid APIs or query-time network; official inference `device=cpu`, fp32 reference; GPU only offline behind a parity gate. |
| D16 | `ModelMeta.revision = <git12>+<config12>`; `cache=None, overwrite_strategy="always"`; strict mode; hash-chained ledger; config paths resolve from the repo root, not the CWD. |
| D17 | Cold-run honesty: the official JSON comes from empty caches and its `evaluation_time` is never replaced by a warm time; a prebuilt demo index may ship (labelled, checksummed, recomputable). |
| D18 | Arbitrary queries (ADR-0003, spec 10): per-route instruction, OOD-guarded LTR (routing v1.1), generic feature extractors with group dropout, gate **G-OOD** vs the frozen base, blind-query protocol; NL version intent is a hint, never a filter. |
| D19 | Evaluation seal (ADR-0002, spec 09): TEST labels live **outside the working tree** (`~/.acis-sealed/`); dev uses `split="train"` only; the official run alone gets its own `HF_HOME`; hooks catch accidents but are not the boundary. Gates use all 5,000 TRAIN queries (non-fit components) or K-fold OOF (trained); DEV-H (F4) is a once-per-milestone confirmation. |

## 2. Invariants (each has an automated test)
- **INV-1** Every returned unit is re-read from the content store by hash; nothing is generated. **INV-2** Results come only from the requested snapshot; cache keys contain `snapshot_id` + `config_hash`.
- **INV-3** Batch invariance: a query's ranking never depends on other queries. **INV-4** External doc/query IDs are never features (only exact-duplicate ordering uses the corpus ordinal).
- **INV-5** Untrusted code is parsed in sandboxed workers only — never imported, executed, compiled or `eval`'d. **INV-6** Same (snapshot, config, numeric profile, threads) ⇒ same ranking.
- **INV-7** Every fallback increments a counter and appears in `diagnostics.degradations`; official runs are `strict`. **INV-8** TEST labels stay outside the working tree and are read only by `acis eval official` / `acis.eval.final` at ledgered RCs (hooks catch accidents; the boundary is physical, D19).
- **INV-9** Only VALID snapshots are searchable; partial ones need `allow_partial=True` and are labelled. **INV-10** Mode A returns exactly `min(top_k, N)` finite, strictly decreasing entries per query.
- **INV-11** The MTEB adapter holds no ranking logic; the engine knows nothing about mteb. **INV-12** Snippet/repository text is data, never instructions; an LLM (if any) sees it delimited, capped, schema-constrained, without write/exec/network tools.
- **INV-13** The agent layer is never in the scored path (`agent_calls == 0`). **INV-14** No number for our system appears in any document without `[ledger:<run_id>]`.
- **INV-15** Query-agnostic: behaviour depends on the query only through generic mechanisms; no closed list of templates/phrases/headers/moduli gates correctness; every pattern extractor fails soft and is masked in training; no lookups or answer caches keyed on query text (spec 10 §2).

## 3. Evaluation-integrity rules
- **TEST labels are sealed.** The held-out AppsRetrieval TEST relevance labels are kept in a separate directory
  outside the working tree (docs/spec/03 §5) and are read only by the official run (`acis eval official` /
  `acis.eval.final`). Nothing is tuned on TEST. Every development decision uses the APPS **train** split ("dev": all
  5,000 queries, or 5-fold out-of-fold for anything trained). TEST touches are budgeted and counted in the ledger.
- **No partition shortcuts.** No feature, filter, prior or normaliser may encode "this document is an APPS-train
  solution", use document or query ids, or use statistics over the batch of test queries (so: no score offsets from
  a reference-query bank, no one-to-one assignment across queries).
- **Numbers need evidence.** Every number about ACIS in a document carries `[ledger:<run_id>]` (the hash-chained
  `runs/ledger.jsonl`), a test id or an artifact path. A number without one is not claimed.
- **Safe by construction.** No `pickle`, `eval`, `exec` or `shell=True`; `torch.load` only with `weights_only=True`;
  models load from safetensors with `trust_remote_code=False`; no query-time network; no paid APIs or keys;
  untrusted code is parsed in sandboxed workers and never imported or executed.
- **Cold-run honesty.** The official JSON comes from empty caches; its `evaluation_time` is never replaced by a warm
  time.
