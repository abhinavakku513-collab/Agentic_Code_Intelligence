# 07 · Phases (v1.1), Definition of Done, final architecture audit

Authority: subordinate to `CLAUDE.md` §5–§6. Ordering rationale, gate power and the seal: `docs/spec/09` (ADR-0002); arbitrary-query work: `docs/spec/10` (ADR-0003); cut order and shippable floors: `docs/TRIAGE.md`. Every phase ends with a working subsystem, a ledger entry, green CI on a clean checkout and `/phase-gate <id>` PASS (evidence recorded in `docs/STATUS.md`). "Not yet" items must not be built in that phase. Jobs > ~10 min run through `scripts/run_detached.sh`. Paths listed as "seeded" ship with the kit; other paths named in specs/skills are Phase 0/1 outputs and do not exist until built.

## Phase 0 — Foundations and trust gates
- **Objective**: a toolchain the hooks/skills can rely on, enforcement that is proven to run, facts verified before anything is built on them.
- **Build**: turn the seeded `pyproject.toml` / `Makefile` / `tests/conftest.py` / `configs/*` into a locked toolchain (`uv lock` with hashes, CI, pre-commit) using the ADR-0001 allowlist; `src/acis/{core,sec,obs}` minimal; `acis doctor` (hardware, tier, checksums); `acis fetch` (allow-listed patterns, checksummed, physical seal per spec 03 §5); `eval/guard.py`, hash-chained ledger, `splits.lock.json` builder; a `README.md` contract in every package dir; `scripts/demo/` and `scripts/train/` placeholders.
- **Gates**: **G0.0** `/canary` (hooks execute: sealed read blocked, sealed `cat` blocked, `eval(` write blocked). **G0.1** a toy SearchProtocol model runs through real `mteb.evaluate` on 2.21.0 and the compat versions (`tests/contract` green: dispatch direct, `encode` never called, `min(top_k,N)` hits, stale-cache hazard reproduced then defeated, datetime-safe writer, tie tests). **G0.2** data audit: counts vs [R-01] ±0.1 %, ID patterns, whether TRAIN solutions sit in the test corpus, alternate-solution availability, exact duplicates, marker inventory, token-length percentiles (sets truncation defaults), HF repo file layout. **G0.3** hardware profile: GEMM GFLOPS, tokens/s per shortlisted encoder (fp32/bf16, several batch/length) → projected cold hours. **G0.4** model radar (live leaderboards via the WebFetch allow-list), licence/remote-code/safetensors audit, pinned SHAs + SHA-256 manifest, parity vs sentence-transformers (cosine ≥ 0.9999), ADR-0001 licence table via `pip-licenses`. **G0.5** frozen zero-shot scores on all 5,000 TRAIN queries (TEST untouched). **G0.6** guard tests green (`tests/security`), TEST labels physically outside the tree, dev cache holds no TEST label file.
- **Not yet**: any retrieval logic beyond toys; the real adapter.

## Phase 1 — Harness and contracts (no TEST touch)
- **Build**: `eval/{metrics,splits,decontam,bootstrap,ledger}`; dev task (`AbsTaskRetrieval` subclass on train qrels); stock `bm25s` BM25 baseline + parity with `mteb/baseline-bm25s` (harness sanity — **not** a submission); `acis.robust_hook:search` over a fixture corpus (≥ 200 docs) so `tests/robustness` runs on the real engine; adapter v0 (Mode B) validated on dev and on a public non-APPS mteb code task; `verify-submission` v0; official script; `rank/compose.py`.
- **Acceptance**: metrics equal mteb/pytrec_eval to 1e-9 on random/oracle/BM25 runs (B0); BM25 parity ≥ 95 % top-10 identical (B1); split lock + sealed-data tests; tie and dispatch contract tests green on the matrix; robustness library tests green and engine tests run; **`AcisEngine` interface frozen** (Track B may start).
- **Not yet**: dense encoder runtime, LTR, TEST evaluation.

## Phase 2 — Dense engine, bake-off, RC0
- **Build**: `embed/` runtime (token-budget batching, CAS vector cache, numeric profiles), exact search, query-vector cache, dense channel, per-route G1 sweeps, G-M bake-off over the radar shortlist (non-permissive models measured as reference).
- **Acceptance**: B2 table with CIs on all 5,000 TRAIN queries; determinism across runs/thread counts; cache-hit ≡ cold (cosine ≥ 0.9999); **G-M** and **G1** decided; scorecard rows per candidate; **RC0 = best frozen dense, Mode B**: owner runs `make rc-official RC=RC0 MODE=B` (cold, strict), `verify-submission` PASS, README quick start written, 1 TEST touch ledgered (floor F1).
- **Not yet**: BM25 fusion, features, LoRA.

## Phase 3 — Adaptation (GPU hand-off)
- **Build**: decontaminated data builder, hard-negative miner with false-negative filter, `acis train export|import` + `scripts/train/train_lora.py --resume auto` (`docs/GPU_HANDOFF.md`), K-fold OOF vectors, α-interpolated merge, REG suite, `acis eval robustness` (G-OOD), D1 report on held-out folds; optional distillation.
- **Acceptance**: **G3** and **G-OOD** with full evidence; import parity gate (CPU fp32 vs GPU vectors cosine ≥ 0.999 on a 1 % sample); training reproducible from `TRAIN_MANIFEST.json`; merged safetensors + checksums; G-M re-run over {frozen, adapted}; GPU use disclosed in the manifest. No GPU ⇒ decision recorded, frozen base stays.
- **Not yet**: LTR, PRF.

## Phase 4 — Hybrid, features, PRF, routing on the final encoder
- **Build**: generic feature extractors with group dropout, OOF feature builder, LightGBM LambdaRank, PRF pass, routing v1.1 (OOD score + feature availability), weighted fusion tuned on REG data, isotonic confidence + `no_strong_match`, strict composition and fallback chain.
- **Acceptance**: B5–B7 with paired CIs on the OOF pool; union Recall@100 measured; **G2, G4, G5, G-AB, G6** decided; G-OOD re-passed; 5-seed LTR variance.

## Phase 5 — Freeze and RC1 (P0 complete)
- **Build**: frozen `configs/official.yaml` (hash in the ledger), G7 tests (batch invariance, ID permutation, determinism, `min(top_k,N)` monotone output, fallback chain), `acis eval official` for Modes A + B (owner-launched), manifest, `verify-submission`, `--cache-verify`, compat CI, resource scorecard, post-hoc clean-pool score (TEST touch ×1).
- **Acceptance**: `verify-submission` PASS; re-scored `run.trec` equals the JSON; both JSONs present; TEST touches match the ledger; every README number cites the RC1 ledger entry. **P0 complete (floor F2).**

## Track B — parallel from Phase 2 (starts when the Phase 1 gate freezes the interface)
- **B1 · Snapshot store and P1**: `store/`, `ingest/` (JSONL/dir/ZIP/git sources, sandbox workers), selectors, `index/update_version/compare_versions`, Apps-Evolve generator (P1 part). Acceptance: spec 04 §5 suite; `kill -9` recovery at every build step; incremental ≡ scratch; update times measured on ≥ 2 corpora against the target and published with hardware.
- **B2 · Bonus**: `lineage/` (cascade S0–S5, store, union-find), evolution-aware search, grouping/timeline, benchmarks. Acceptance: wrong-merge ≤ 1 %; Duplicate-Rate@10 ≈ 0 grouped; Evolution-NDCG@10 and Best-Revision-Hit@1 beat flat all-version search with CI lower bound > 0.
- **B3 · API/CLI/UI and demo**: FastAPI + pydantic, `acis` CLI, one static free-text search page (channel toggle, scorecard panel, `interpreted_intent` hint, timing bar; no CDN), metrics/logs/health, `acis report`, `scripts/demo/` runbook including `commit_stream.py` (spec 09 §6). Acceptance: API E2E tests; spec 02 §7 targets measured on the reference host; `make demo` runs offline end to end. No slides, no video.
- **B4 · Agent (optional, last)**: controller, tools, budgets, hard-query dev set. Acceptance: budgets provably never exceeded; injection corpus inert; Recovery@10 measured; default-on only if the benefit gate passes, else "evaluated, not adopted".

## Phase 6 — Closure
- **Build**: cut-list-aware security hardening (parse-only workers, limits, safe paths, scanners; S2/signing/fuzz-1h only if time), performance closure, memory-tier tests, `/evidence-pack` (claims N1–N8 → `docs/submission/`), owner hand-off files, README + release pack, blind-query protocol (run once), clean-machine network-off dry run, `/preflight`.
- **Acceptance**: every failure-matrix row has an automated test; clean `pip-audit`/`osv-scanner`/`gitleaks`; `/preflight` PASS on C1–C11 and C13; each innovation claim evidenced or deleted; Definition of Done complete.

## Definition of Done
Identical to `CLAUDE.md` §6 — each box needs a ledger id, test id or artifact path in `docs/STATUS.md`. Additionally: no TODO/FIXME in `src/`; every package has a `README.md` contract; `docs/spec` matches shipped behaviour (ADR for every deviation).

## Final architecture audit (red team — what could fail, and what already changed)
| Area | Risk | Correction / mitigation |
|---|---|---|
| Enforcement | hooks may not execute; regex hooks are bypassable | string-form hook commands + `/canary` (G0.0); physical TEST seal (D19); guard v1.1 tests with pinned `xfail(strict)` gaps; ask-gated official commands |
| Accuracy | small/frozen encoder may be weak; adaptation may be late or exploit train docs | adaptation moved to Phase 3; gates on all 5,000 / OOF (power); D1 on held-out folds; clean-pool report; no train-doc detector |
| Gate statistics | DEV-H ≈ 1,000 queries cannot see 0.5–1 pt gains | decision sets per spec 03 §5; DEV-H = once-per-milestone confirmation |
| Evaluation validity | stale mteb cache, tie collapse, datetime JSON crash, wrong dispatch, small-corpus fixtures | unique ModelMeta + `cache=None`; rank-derived scores; datetime-safe writer; no `predict()`; `min(top_k,N)`; contract tests on the matrix |
| Arbitrary queries | APPS-shaped instruction/routing/features/version regexes | INV-15, per-route instruction, routing v1.1, group dropout, G-OOD, blind-query protocol |
| Integration | organizer mteb version differs; sample subclasses `AbsEncoder` | compat matrix, keyword-only signatures + `**_`, Mode B always shipped, RC0 is Mode B |
| Resources / ops | multi-hour cold pass, GPU hand-off, session limits | tiers, G6, D4 rule, detached runner, `docs/GPU_HANDOFF.md`, honest cold time, prebuilt demo index |
| Security | hostile archives/git/snippets, prompt injection, supply chain | parse-only sandboxed workers, limits, hardened git, pinned/offline models, injection corpus, ADR-0001 allowlist |
| P1 / Bonus | partial updates, leaks, slow commits, over-merging | immutable snapshots, validate-then-activate, isolation property tests, update-time target, commit-stream demo, evidence-gated lineage |
| Over-scope | non-scored work crowding out scored work | `docs/TRIAGE.md` cut order; floors F0–F4; PPT/video owner-made |
| Unrealistic assumptions | cost estimates, third-party scores | all [E]/[R] items are re-measured in Phase 0/2 before use; sources in `docs/reference/third_party_scores.md` |
