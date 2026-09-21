# 06 · Contracts, testing, performance, observability

Authority: subordinate to `CLAUDE.md`. Modules and classes are derived directly from this file; names below are the public surface.

## 0. Package map and hardware tiers — one engine, thin surfaces (`src/acis/`)
| Package | Responsibility (offline unless noted) | P0 | P1 | B |
|---|---|---|---|---|
| `core` | types (`Snippet`,`Unit`,`Hit`), config + hash, ids/hashing, errors, numeric profiles | ✓ | ✓ | ✓ |
| `data` | APPS loaders (train only), fold split, decontamination (MinHash) | ✓ | | |
| `prep` | query/doc normalisation, segmentation, views, truncation (deterministic) | ✓ | ✓ | ✓ |
| `embed` | encoder runtime, token-budget batching, content-addressed vector cache, ONNX/bf16 profiles | ✓ | ✓ | ✓ |
| `lexical` | code-aware tokenisers, per-snapshot bm25s | ✓ | ✓ | ✓ |
| `features` | `ast`-derived + query↔code bridge features (parse-only) | ✓ | ✓ | ✓ |
| `rank` | candidates, PRF, LightGBM LTR, confidence, score composition | ✓ | ✓ | ✓ |
| `engine` | `AcisEngine`: search pipeline, routing, verification, degradation ladder | ✓ | ✓ | ✓ |
| `store` | CAS, snapshot store (SQLite + mmap), journal, recovery, GC | | ✓ | ✓ |
| `ingest` | safe loaders (JSONL/dir/ZIP/git), sandbox workers, unit extraction | | ✓ | ✓ |
| `lineage` | alignment cascade, lineage store, evolution-aware search | | | ✓ |
| `agent` | bounded investigation controller + read-only tools (online, optional) | | | |
| `api` / `cli` | FastAPI (loopback) and `acis` CLI over the same engine | | ✓ | ✓ |
| `eval` | metrics parity, splits+guard, ledger, bootstrap, ladder, `final`, `verify` | ✓ | ✓ | ✓ |
| `mteb_adapter` | thin PrePostPipelineEncoder (Mode A + B), ModelMeta identity, artifacts | ✓ | | |
| `obs` / `sec` | logs, metrics, health / safe paths, limits, sandbox | ✓ | ✓ | ✓ |
| `robust_hook` | `search(query, top_k) -> list[str]` over a fixed fixture corpus, the plug-in point of `tests/robustness` (spec 10 §8) | ✓ | | |

Hardware tiers (measured by `acis doctor` → `runs/hardware.json`): **T-min** 4 physical cores / 8 GB (interactive with bf16/int8 or ≤150M encoder; official pass may take many hours) · **T-rec** 8+ cores / 16 GB (reference host for scorecards) · **T-gpu** optional, offline only. Degradation tiers: full → bf16/int8 + smaller token budget → cached vectors + BM25 with the query encoder disabled (always explicit).

## 1. Public Python interface (`acis.engine`) — **frozen at the end of Phase 1** (Track B builds against it; changes need an ADR)
```python
class AcisEngine(Protocol):
    def ingest(self, source: SourceSpec, *, repo_id: str, limits: Limits | None = None) -> JobHandle: ...      # async build job; idempotent per (source digest, refs)
    def index(self, repo_id: str, *, refs: RefSpec | None = None, mode: BuildMode = "eager_heads") -> BuildReport: ...
    def search(self, req: SearchRequest) -> SearchResponse: ...                                                # snapshot resolved once, pinned for the request
    def search_version(self, repo_id: str, version: str, query: str, **kw) -> SearchResponse: ...              # = search with a pinned selector
    def update_version(self, repo_id: str, delta: SourceSpec, *, expected_active: str | None = None) -> BuildReport: ...
    def compare_versions(self, repo_id: str, a: str, b: str, *, query: str | None = None) -> VersionComparison: ...
    def retrieve_evolution(self, req: EvolveRequest) -> EvolveResponse: ...                                    # lineage-grouped; flat=True disables grouping
    def diagnostics(self, repo_id: str | None = None) -> Diagnostics: ...                                      # config hash, model fingerprint, profile, hardware, tier, degradations
    def evaluate(self, spec: EvalSpec) -> EvalReport: ...                                                      # CLI-only; never exposed on the network API
    # batch surface used by the adapter (no mteb types):
    def build_snapshot(self, docs: Sequence[Snippet], *, source: str) -> Snapshot: ...
    def search_batch(self, snap: Snapshot, ids: Sequence[str], texts: Sequence[str], *, top_k: int,
                     restrict_to: Mapping[str, Sequence[str]] | None = None, strict: bool = False) -> dict[str, list[tuple[str, float]]]: ...
```
**Invariants**: pure w.r.t. inputs + snapshot + config (INV-6); `search_batch` is batch-invariant (INV-3); every `Hit` carries `(repo_id, version_id, snapshot_id, unit_id, body_hash)` and its `source` equals the CAS bytes (INV-1). **Errors** (subclass `AcisError`; HTTP): `InvalidInput`(400) · `NotFound`(404) · `IndexRequired`(409) · `SnapshotInvalid`(409) · `VersionConflict`(409) · `ResourceLimit`(413/429) · `NotReady`(503) · `StrictViolation`(500, official runs) · `SealedDataAccess`(never returned; raises in tests). **Core types**: `Snippet(handle, text)`, `Unit`, `Hit(rank, score, unit, source, signals, also_at)`, `SearchRequest{repo_id, version, query, top_k≤1000, mode, filters, investigate, allow_partial, explain, diagnostics}`, `SearchResponse{snapshot{id,version,complete,missing_channels}, results[], confidence, no_strong_match, route, interpreted_intent, timings_ms, degradations[], investigation}`.

## 2. REST (FastAPI + pydantic v2; JSON; `application/problem+json` errors; loopback by default; body ≤ 8 MB; query ≤ 16k chars)
`POST /v1/repos` · `POST /v1/repos/{id}/ingest` (202 + job) · `GET /v1/jobs/{id}` · `GET /v1/repos/{id}/versions` · `POST /v1/repos/{id}/versions/{v}/activate {expected_active}` · `POST /v1/repos/{id}/rollback` · `POST /v1/search` · `POST /v1/evolve` · `GET /v1/repos/{id}/diff?from=&to=` · `GET /v1/units/{unit_id}` · `GET /v1/healthz|readyz|metrics|diagnostics`. No evaluation endpoint.
```json
POST /v1/search  {"repo_id":"apps","version":"latest","query":"…","top_k":10,"mode":"auto","investigate":"auto","explain":true}
→ {"snapshot":{"id":"s_41ab…","version":"v3","complete":true,"missing_channels":[]},
   "route":"problem_statement","confidence":"high",
   "results":[{"rank":1,"score":0.999,"unit":{"unit_id":"u_9f2c…","key":"apps/1234","body_hash":"…","version":"v3"},
               "source":"<exact CAS bytes>","signals":{"dense_rank":1,"lexical_rank":4,"ltr":1.9},"also_at":[]}],
   "timings_ms":{"normalize":1,"encode":412,"dense":3,"lexical":2,"ltr":1,"verify":2,"total":421},"degradations":[]}
```
## 3. CLI (`acis`, same engine)
`doctor` · `fetch` · `ingest` · `index` · `search` · `versions` · `activate` · `rollback` · `diff` · `evolve` · `verify` · `report` · `serve` · `eval dev|ladder|gate|robustness|official|verify-submission|repro` · `train export|import` (GPU hand-off) · `report [--claims]` (ledger-backed evidence table for N1–N7). `make` targets in AGENTS.md wrap these.

## 4. Frozen config (`configs/official.yaml`, hashed → `config_hash`)
```yaml
run:      {mode: A, strict: true, device: cpu, threads: auto_physical, cold_official: true, test_touch_budget: 6}
model:    {name: <G-M>, base_commit: "<sha>", adapter: null, numeric_profile: cpu-fp32, pooling: model_card, normalize: true}
prep:     {query: {version: q1, task: T1, view: V0, max_tokens: 1024, head: 768, tail: 256}, doc: {version: d1, max_tokens: 1024, head: 768, tail: 256}}
lexical:  {tokenizer: acis-code-v1, k1: 1.2, b: 0.75}
retrieve: {dense_k: 100, lexical_k: 30, union_cap: 100, prf: {enabled: false, m: 3, alpha: 0.2}, top_k_out: 1000}
ranker:   {type: lightgbm_lambdarank, model_hash: "<sha>", fallbacks: [weighted_fusion, dense_only]}
compose:  {mode_a_scores: rank_derived}
gates:    {G_M: {run_id: "<ledger>", decision: "<model>"}, G1: {…}, G2: {…}, G3: {…}, G4: {…}, G5: {…}, G_AB: {…}, G6: {…}}
integrity: {split_lock_hash: "<sha>"}
```
Ranking-relevant keys change the hash; operational knobs (threads, ports) do not. Config and data paths resolve from the **repo root** (`ACIS_ROOT`, default: the directory containing `pyproject.toml`), never the CWD — a judge may run from anywhere; `ACIS_HOME` (outside git) holds data, CAS and caches.

## 5. Testing strategy (each item has an ID; CI stages: lint/type/security → unit/property → contract matrix → integration → chaos + perf nightly → release smoke)
| Layer | Content | Gate |
|---|---|---|
| Unit | normaliser, segmenter, tokenisers, feature extractors, hashing, `rank_derived_scores`, safe path/extractor, manifest | 100 % of critical modules; coverage ≥ 90 % on core |
| Property (Hypothesis) | INV-2 isolation over random histories, INV-3 batch invariance, INV-4 ID permutation, strictly decreasing scores survive float32, cache key covers every request field, hash stability under reformatting | no counter-example |
| Metamorphic / parity | P1 SearchProtocol path ≡ encoder path for the same embedder · P2 our BM25 ≈ `mteb/baseline-bm25s` · P3 metrics re-scored from `run.trec` = JSON (1e-9) · P4 query alone ≡ in batch · P5 corpus order / id relabelling invariance · P6 warm ≡ cold rankings · P7 adversarial ties/near-ties keep NDCG = MRR for a gold-at-rank-1 fixture | exact / stated tolerance |
| Contract (mteb matrix) | dispatch is direct in Mode A, `encode` never called, 1,000 hits, JSON writer, stale-cache hazard reproduced then defeated, Mode B via `SearchEncoderWrapper` | green on every matrix cell |
| Retrieval / ranking | ladder rungs reproduce ledger numbers within tolerance on dev; LTR skew test (train vs serve features identical); hard-query buckets (recall failure, ranking failure, confuser, truncation, parse, duplicates) | no unexplained drop |
| P0 evaluation | RC dry run on a 500-doc fixture through `mteb.evaluate`; `verify-submission` | PASS |
| P1 / Bonus | spec 04 §5 and §6 suites | thresholds in spec 04 |
| Security / adversarial | hostile archives, symlink/hook traps, path traversal, bombs, deep nesting, giant lines, binary-as-text, unicode tricks, prompt-injection and keyword-stuffing corpora, egress-blocked run, sandbox-kill test | zero tolerance |
| Failure injection / chaos | kill embedder, corrupt a vector file, truncate a manifest, delete `VALID`, disk full, `ulimit -v` memory caps, `kill -9` during activation | recovers to ACTIVE/PREV, counters correct |
| Fuzz (≥ 1 h/parser before release) | archive reader, `safe_path`, selector parser, JSONL reader, query normaliser | no crash/hang |
| Performance | stage latency histograms, cold/warm timing, memory ceilings, update-time curve, tiers under caps | spec 02 §7 targets or documented deviation |
| Robustness (INV-15, G-OOD) | `tests/robustness` label-free properties (engine plugged in via `acis.robust_hook:search`), `acis eval robustness` (perturbation families vs the frozen base), REG suite, blind-query protocol (≥ 50 queries, once) | G-OOD |
| Hooks / seal | `tests/security/test_guard_hook.py`, `test_guard_extra.py` (gap rows pinned `xfail(strict)`), `/canary` | G0.0, G0.6 |
| End to end | fresh container, network off: README → cold official run (or `--cache-verify`) → demo runbook | reproduces within tolerance |
Markers: `slow`, `gpu`, `network` are excluded from `make test-fast` (the Stop hook).

## 6. Performance protocol
Workloads: official cold pass (index + 3,765 queries), interactive long query, interactive short query, P1 update. Measure on a declared reference host with `acis doctor` output; report p50/p95/p99, warm vs cold, peak RSS, CPU-seconds, params, model MB, `evaluation_time`. Optimisation order: only what profiling shows (encode dominates): length-sorted token-budget batching → query-vector cache → bf16/int8 (G6) → shorter view V1 (G1) → smaller encoder (G-M) → optional KV reuse of the instruction prefix (valid because pooling is causal; needs an equivalence test). Parallel channels via thread pool so latency ≈ slowest branch.

## 7. Observability (local, privacy-preserving; nothing requires Prometheus/Grafana servers)
- **Logs** (structlog JSON lines): `ts, level, event, request_id, repo_id, snapshot_id, route, stage_ms{normalize,encode,dense,lexical,fusion,ltr,verify,agent}, candidates{dense,lexical,union}, confidence_band, degradations[], cache{qemb,result}, status` — no query text or code by default; opt-in redacted text logging.
- **Metrics** (`/metrics`, prometheus text via `prometheus_client`): `acis_request_seconds{route,stage}` histograms; counters from the failure matrix; `acis_cache_hits_total/misses_total{cache}`, `acis_verify_fail_total`, `acis_agent_escalations_total`, `acis_snapshot_builds_total{result}`; gauges: RSS, CPU, queue depth, active-snapshot age, units, segments, tier; histogram of confidence bands; build/update durations.
- **Per-request diagnostics** (`diagnostics=true`): stage timings, candidate counts, per-result channel ranks/scores ("why this result"), degradations, cache hits.
- **Health**: `/healthz` process · `/readyz` model checksum OK + active snapshot VALID + disk headroom · `/v1/diagnostics` config hash, model fingerprint, profile, hardware, tier.
- **`acis report`**: static HTML/JSON of p50/p95/p99, cache hit rate, freshness, failure counters — the source of every latency number in the README and the owner's slides (INV-14).
