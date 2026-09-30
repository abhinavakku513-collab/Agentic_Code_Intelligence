# ACIS — Agentic Code Intelligence

**Natural-language query in, ranked code snippets out, with the exact source as evidence.** Local, offline,
CPU-only. No generation, no LLM in the scored path, no network at query time.

Built for Samsung PRISM GenAI Hackathon Theme 1: P0 (CoIR *AppsRetrieval* via MTEB), P1 (retrieval across code
versions with fast index rebuild) and the Bonus (retrieval across *all* versions).

---

## Quick start (beginner-friendly)

**Prerequisites.** Linux, macOS or WSL2; Python 3.12; [`uv`](https://docs.astral.sh/uv/); about 10 GB free disk
and 8 GB RAM. No GPU is needed: serving and the official run are CPU-only.

```bash
# 1. environment (hash-locked; CPU-only torch)
make setup
make doctor                                        # hardware profile -> runs/hardware.json

# 2. data and model (the only steps that use the network)
make fetch                                         # APPS assets
make fetch-models MODELS=gte-modernbert-base       # the encoder, pinned by commit + per-file SHA-256

# 3. search from the terminal (P0)
uv run acis search "reverse a linked list" --top-k 5

# 4. the page: P0 search, P1 versions, Bonus evolution, recorded P0 evaluation
uv run python scripts/demo/build_history.py        # once: a 5-version APPS repository for P1/Bonus (minutes)
uv run acis serve                                  # then open http://127.0.0.1:8000/
```

`acis serve` is both the backend and the page (one process). It is ready when
`curl -s http://127.0.0.1:8000/readyz` shows `"p0_corpus": true` (about a minute: the model loads and the 8,765-unit
index is built from the vector cache). Restart it after upgrading the code: the page warns when the running process
is older than the page it serves.

**The P0 pipeline the page and the evaluation both run** (one function, `AcisEngine._rank_one`):

```
query -> normalise -> route (statement-like | generic)
      -> dense (gte-modernbert-base, exact) + BM25 (code tokens) + exact symbols      -> union of <= 500 candidates
      -> every candidate scored by both channels, symbol and agreement features
      -> statement-like: LightGBM LambdaRank (abstains on thin evidence)
         generic:        weighted fusion of dense + BM25 (alpha tuned on CosQA; the ranker measured worse there)
         one identifier: units containing it verbatim first
      -> dense tail -> evidence re-read from the content store by hash
```

**P0 dev evaluation** (never the held-out TEST labels): `uv run python scripts/bench/eval_pipeline.py`
(all 5,000 dev queries, out of fold, about 10 minutes with cached vectors). It appends a ledger row and writes a
per-query artifact; the page's *P0 evaluation* tab shows exactly that row.

**P1 and Bonus from the terminal:** `acis ingest` (JSONL, directory, ZIP or git), `acis versions`, `acis diff`,
`acis activate`, `acis rollback`, `acis search --repo … --version …`, `acis evolve`. A real-git check with hand-made
commits (edit, move, unrelated file, revert): `uv run python scripts/demo/real_git_check.py`.

**Troubleshooting.**
- *"no weights for …"*: run `make fetch-models MODELS=gte-modernbert-base`.
- *The page shows "server process is older than this page"*: stop and restart `acis serve`.
- *First query slow*: the first encode of a new text is the model (≈0.16 s short, ≈0.7 s for a full statement on
  this CPU, `[ledger:bench-8f34183886c2]`); repeats are served from the vector cache.
- *Low memory (WSL2)*: give WSL at least 8 GB; do not run several evaluations at once.

**Artifacts.** Data, CAS, caches and models live under `ACIS_HOME` (default `~/.acis/home`); evaluation artifacts
under `runs/` (git-ignored except `runs/ledger.jsonl`); shipped models in `artifacts/` (ranker, routing bank,
confidence calibration). REG benchmarks are kept apart in `~/.acis/reg`.

**GPU (optional, offline only).** `bash scripts/train/gpu_env_setup.sh` (owner-run) creates a separate CUDA tool
environment; `scripts/train/gpu_embed.py` then embeds with a second encoder and admits its vectors to the cache only
after an fp32 parity gate (cosine >= 0.999). Serving never needs it.

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
| **Retrieval** | Exact dense search (one matmul, no ANN below 250k vectors), code-aware BM25, an exact-symbol channel, a ≤ 500-candidate union, parse-only and symbol features, a cross-fitted LightGBM LambdaRank ranker that **abstains** when its evidence has not fired, weighted dense + BM25 fusion for non-statement queries, a calibrated confidence; an optional second dense encoder channel |
| **P1 · versions** | Content-addressed store, immutable snapshots, validate-then-activate, atomic ref switch, instant rollback, crash recovery, incremental builds that embed only what changed |
| **Bonus · evolution** | An evidence-gated alignment cascade (identical → moved → renamed → inferred), lineages closed by union-find, one answer per lineage with its best revision and a timeline (reverts named) |
| **Surfaces** | `acis` CLI, FastAPI on loopback, a single static page with no CDN, `make demo` |
| **Evidence** | A hash-chained ledger: every number ACIS reports about itself carries a `run_id`, and a number without one does not appear in any document |

## Measured, with evidence

Every figure below is a ledger row (`runs/ledger.jsonl`, hash-chained). Numbers about ACIS never appear here without one.

| What | Result | Evidence |
|---|---|---|
| **P0 dev, served pipeline** (5,000 queries, out of fold, engine path) | **NDCG@10 74.13 · MRR@10 70.93 · Recall@100 94.86** | `[ledger:dev-299a3010be5a]` |
| P0 dev, dense only (same queries, same code) | NDCG@10 71.03 · MRR@10 67.46 · Recall@100 93.78 | `[ledger:dev-5bd4f1c50c7c]` |
| Served vs dense | +3.09 pt NDCG@10, 95 % CI [+2.62, +3.57] (paired bootstrap) | `[ledger:dev-299a3010be5a]` |
| Candidate recall (relevant unit in the union of ≤ 500) | 97.16 % (dense alone @500: 97.90 %, @100: 93.78 %) | `[ledger:dev-f320c5718b54]` |
| Encoder bake-off (G-M) | `gte-modernbert-base` selected (71.03); granite-r2 56.70, granite-small-r2 53.95; Qwen3-0.6B excluded on its projected cold pass | `[ledger:gate-055152620da6]` |
| Query prep sweep (G1, 9 cells) | V0/1024 71.03 best; the rule prefers the cheaper V0/512 (70.69, within 0.5 pt) — owner decision pending | `[ledger:gate-f736bd6e3a31]`, `[ledger:gate-afd311693e96]` |
| Generic route on human-written queries (REG) | fusion vs dense: CodeSearchNet-Py +0.81 [+0.30, +1.32]; CosQA −0.36 [−1.58, +0.84] | `[ledger:dev-63c41669d7e2]`, `[ledger:dev-ba33136ce9dc]` |
| G-OOD perturbations (served vs dense brittleness) | 8 of 9 families pass; `format_noise` fails its strict 0.0 limit (+0.12, CI [−0.30, +0.56]) | `[ledger:dev-e8304c8b6364]` … `[ledger:dev-19ccf584c73b]` |
| Confidence calibration (held out) | statements rated *high*: top result relevant 89 % | `[ledger:dev-50c946dc1c29]` |
| Latency, quiet CPU (8 cores) | short query cold p50 159 ms / p95 202 ms; warm p95 15.4 ms; full statement cold p50 657 ms | `[ledger:bench-8f34183886c2]` |
| Metric parity with mteb/pytrec_eval | exact to 1e-9 | `tests/metamorphic/test_parity.py` |
| P1 update time | 1- or 10-unit change searchable in ~2 s p95 | `[ledger:bench-9153e49050de]` |
| Bonus | lineage recovery exact, grouping removes duplicates; grouped and flat **tie** on Evolution-NDCG@10 | `[ledger:bench-52d48f680740]` |

## Verify it yourself

Every check below runs locally and offline after `make setup`, `make fetch` and
`make fetch-models MODELS=gte-modernbert-base`. Where this section gives a number, it cites the ledger row that
produced it. Everything else is described by the *shape* of the output, because timings depend on your machine.

**1. The integrity tests stay green** (metric parity, score ties, INV-1 evidence, the seal guard):

```bash
uv run pytest -q tests/metamorphic tests/contract tests/security
```

Expect all passed plus exactly 3 `xfailed`. Those are the documented guard-hook gaps (`tests/security/test_guard_hook.py`):
a regex hook cannot see shell indirection, which is why the held-out labels live outside the tree.

**2. Free-text search with the real encoder (P0)**

```bash
uv run acis search "reverse a linked list" --top-k 5
```

Expect `encoder : Alibaba-NLP/gte-modernbert-base`, a snapshot of 8,765 units, an `order` line naming what ranked
the list, per-stage `timings`, and five rows with a cosine similarity, the dense rank, the BM25 rank (or `-` when
BM25 did not retrieve it), the key, the content hash and the first line of the code. A short question like this takes
the generic route (`order: weighted fusion of dense + BM25 …`); a full problem statement goes through the learned
ranker; a query that is one identifier (`dijkstra`, `heapq.heappush`) lists the units containing it first.

**3. The page (P0, P1, Bonus)**

```bash
uv run python scripts/demo/build_history.py      # once: the 5-version APPS repository (minutes, real encoder)
uv run acis serve                                 # ready when /readyz says "p0_corpus": true
for p in / /app.css /app.js /favicon.ico; do curl -s -o /dev/null -w "$p %{http_code}\n" http://127.0.0.1:8000$p; done
```

Expect `200` for all four. Open <http://127.0.0.1:8000/>:

- **Search**: type any query. Each hit shows its cosine similarity, dense and BM25 ranks and its code; the
  summary line shows what ordered the list, the route and query category, the channels that ran and how many
  candidates each gave, the calibrated confidence and the engine's stage timings; the footer shows this session's
  p50/p95 latency.
- **Versions**: `apps-history` pinned to `v2` returns only v2's units, and the page checks the answering snapshot
  belongs to the selected repository. *Synthetic commit* (labelled a demo mutator, not real ingestion) edits units
  and reports `vN searchable in X s · N unit(s) embedded, M reused`. *Roll back* moves the active version back.
- **P0 evaluation**: the recorded dev evaluation, rendered from its ledger row (NDCG@10, MRR@10, query count,
  encoder and commit, config hash, split, run ids), with a per-query browser read from the SHA-256-pinned artifact.
- **Evolution**: *Grouped by lineage* shows each unit once with a timeline across versions, its best revision
  and that revision's code. *Flat · all versions* shows every matching revision and the flat duplicate rate
  that grouping removes.

The same checks as API calls:

```bash
curl -s -X POST localhost:8000/v1/search -H 'content-type: application/json' \
  -d '{"query":"reverse a linked list","top_k":3,"explain":true}'        # results[].signals, explanation.ordered_by
curl -s -X POST localhost:8000/v1/repos/apps-history/commit -H 'content-type: application/json' -d '{"edits":3}'
curl -s -X POST localhost:8000/v1/evolve -H 'content-type: application/json' \
  -d '{"query":"sort an array","repo_id":"apps-history","top_k":5}'      # groups[].timeline, groups[].best.source
```

**4. The accuracy numbers (dev split only, never the held-out labels)**

```bash
uv run python scripts/bench/eval_pipeline.py        # the served pipeline, out of fold, all 5,000 dev queries
uv run python scripts/bench/recall_diagnostics.py   # candidate recall per channel
```

The first reproduces `[ledger:dev-299a3010be5a]` (NDCG@10 74.13, MRR@10 70.93) against dense
`[ledger:dev-5bd4f1c50c7c]` (71.03, 67.46) through `AcisEngine._rank_one`, the function the page calls. With vectors
cached it takes minutes; on a clean machine the corpus and queries are embedded first, which takes hours on a CPU.

**Not verified yet, and not claimed:**

- A second dense encoder (Qwen3-Embedding-0.6B) is implemented as a channel (`model.aux_encoder`) but **not
  measured**: embedding the corpus on this CPU takes ~3 s per unit; the GPU path (`scripts/train/gpu_env_setup.sh`,
  owner-run) is ready but not yet run.
- SPLADE is **not integrated** (mainstream checkpoints are CC BY-NC-SA, outside the permissive-licence envelope).
- G-OOD: `format_noise` still fails its strict identity limit (`[ledger:dev-e8304c8b6364]`); the other 8 families pass.
- The official primary stays Mode B until the owner runs G-AB and the official cold run; `configs/official.yaml`
  now carries the same pipeline for Mode A.
- The Bonus ranking claim: grouped and flat tie on Evolution-NDCG@10 (`[ledger:bench-52d48f680740]`).
- Latency targets of spec 02 §7 are partly missed on this 8-core dev host (see the table above).

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

`docs/STATUS.md` is the single source of progress truth; `docs/FINAL_ACCEPTANCE_REPORT.md` is the evidence report for
this implementation pass. The PPT and demo video are the owner's (`docs/submission/OWNER_HANDOFF.md`).
