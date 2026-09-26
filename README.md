# ACIS — Agentic Code Intelligence

**Natural-language query in, ranked code snippets out, with the exact source as evidence.** Local, offline,
CPU-only. No generation, no LLM in the scored path, no network at query time.

Built for Samsung PRISM GenAI Hackathon Theme 1: P0 (CoIR *AppsRetrieval* via MTEB), P1 (retrieval across code
versions with fast index rebuild) and the Bonus (retrieval across *all* versions).

---

## Quick start

```bash
make setup                       # uv sync — Python 3.12, hash-locked
make doctor                      # hardware profile -> runs/hardware.json
make fetch                       # the APPS dataset assets (the only network step)
make fetch-models MODELS=gte-modernbert-base   # the shipped encoder (G-M), pinned by commit + file hashes

uv run acis search "find the shortest path in a weighted graph"      # P0: free-text retrieval
make demo                        # the scripted runbook: P0, then P1 live, then the Bonus, then the UI
```

`make demo` ends by serving <http://127.0.0.1:8000/> — the same engine behind a single offline page: free-text
search, a channel toggle, version pinning and lineage grouping.

## Reproducing the official evaluation

```bash
make reproduce                   # cold official run from empty caches -> runs/<run_id>/
uv run acis eval verify-submission --run-dir runs/<run_id>
```

`configs/official.yaml` is the configuration: `gte-modernbert-base`, Mode B, strict, CPU only. The run is cold on
purpose — no embedding is read from a cache, so `evaluation_time` includes encoding the whole corpus. On an
8-core CPU expect about 2.6 hours (projected in `[ledger:gate-055152620da6]`); let it finish uninterrupted.

The official recipe is the one the guidelines specify: `PrePostPipelineEncoder(AbsEncoder)` →
`mteb.get_task("AppsRetrieval")` → `mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})`, written
through a datetime-safe JSON writer (the sample's bare `json.dump` crashes on the result's `date` field), with
`cache=None, overwrite_strategy="always"` so a stale result cache cannot be mistaken for a run.

Two surfaces ship, from one class: **Mode B** (an honest `encode()`, which mteb's own search wrapper drives) and
**Mode A** (the full `index`/`search` pipeline). `predict()` is never defined — that would send the object down
mteb's cross-encoder branch.

## What is in the box

| | |
|---|---|
| **Retrieval** | Exact dense search (one matmul, no ANN below 250k vectors), per-snapshot BM25, candidate union, parse-only features, a cross-fitted LightGBM LambdaRank ranker that **abstains** when its evidence has not fired |
| **P1 · versions** | Content-addressed store, immutable snapshots, validate-then-activate, atomic ref switch, instant rollback, crash recovery, incremental builds that embed only what changed |
| **Bonus · evolution** | An evidence-gated alignment cascade (identical → moved → renamed → inferred), lineages closed by union-find, one answer per lineage with its best revision and a timeline |
| **Surfaces** | `acis` CLI, FastAPI on loopback, a single static page with no CDN, `make demo` |
| **Evidence** | A hash-chained ledger: every number ACIS reports about itself carries a `run_id`, and a number without one does not appear in any document |

## Measured, with evidence

Every figure below is either a ledger row or a file in `runs/`. Numbers about ACIS never appear here without one.

| What | Result | Evidence |
|---|---|---|
| Metric parity with mteb/pytrec_eval | exact to 1e-9 on random, oracle and BM25 runs, including inside the float32 collapse zone | `tests/metamorphic/test_parity.py` |
| BM25 parity with `mteb/baseline-bm25s` | 100 % identical top-10 on all 5,000 dev queries | `[ledger:dev-8717922a9ca4]`, `[ledger:dev-ee395f07e048]` |
| Retrieval core latency | p95 16.4 ms at 8,765 units (stand-in encoder; see the note in `docs/PHASE2_REPORT.md`) | `[ledger:bench-b3586309bd8e]` |
| P1 update time | a 1- or 10-unit change searchable in ~2 s p95 where a full rebuild takes ~11 s, against a 30 s target | `[ledger:bench-9153e49050de]` |
| Encoder bake-off (G-M) | `gte-modernbert-base` (149M params) selected at NDCG@10 71.03 on all 5,000 dev queries; the two alternatives were 14 and 17 pt behind | `[ledger:gate-055152620da6]`, `configs/gates/G-M.yaml` |
| Determinism | bit-identical vectors across runs; identical top-10 rankings at 8 vs 2 threads; cached ≡ recomputed (cosine 1.0) | `[ledger:bench-d8ed869255a2]` |

## How it is kept honest

- **The held-out labels are physically outside the working tree** (`~/.acis-sealed/`). Dev work uses the TRAIN
  split only; one module may read the seal, and only inside the official run. Hooks catch accidents — they are
  not the boundary (D19, INV-8).
- **Gates decide, not opinions.** Every claimed improvement is a paired bootstrap over 10,000 resamples on all
  5,000 dev queries (or K-fold out-of-fold for anything trained), needing Δ ≥ +0.5 pt *and* a CI lower bound
  above zero. Ties go to the cheaper system.
- **Nothing degrades silently.** Every fallback increments a counter and appears in `degradations`; a strict run
  refuses all of them. A stand-in encoder says it cannot ship, and the strict path enforces that.
- **Untrusted code is parsed, never executed** (INV-5) — no import, no `exec`, no `eval`, no pickle. Archives
  stream, git is read with no checkout and hooks disabled, and every path goes through one `safe_path`.
- **Query-agnostic by construction** (INV-15): no closed list of templates, phrases, headers or moduli gates
  correctness; every pattern extractor fails soft and is masked during training.

## Layout

```
src/acis/         core · prep · embed · lexical · features · rank · engine · store · ingest · lineage · eval · api · cli
docs/spec/        the contracts each package is built against
docs/adr/         decisions that deviate from them, with the reasoning
runs/ledger.jsonl every number, hash-chained
tests/            unit · property · metamorphic · contract · integration · security · chaos · robustness
```

Start with `CLAUDE.md` (the constitution), then `docs/spec/`, then each package's `README.md` — every package
states its own contract and names the test that enforces it.

## Status

Phase 1 (harness) and Track B1 (P1) are built, with reports in `docs/`. In Phase 2 the encoder is chosen (G-M),
the query-preparation sweep (G1) is being measured, and the first release candidate (RC0) is next; Phase 4's
hybrid ranking and the Bonus are built and measured on the dev split. `docs/STATUS.md` is the single source of progress truth, and the PPT and demo video are the
owner's (`docs/submission/OWNER_HANDOFF.md`).
