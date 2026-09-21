# Agentic Code Intelligence (ACI) — Architecture & Implementation Blueprint

**Version 1.0 · 2026-09-20 · Samsung PRISM GenAI Hackathon · P0 + P1 + Bonus**
**Status:** authoritative specification, superseding the earlier "master prompt". Where the two conflict, this document wins.

---

## 0. How to read this document

### 0.1 Evidence tags

Every non-obvious claim carries a tag so you can see how much weight it can bear.

| Tag | Meaning |
|---|---|
| `[DOC]` | Stated in the official Samsung guideline slides (4 pages). |
| `[SRC]` | Read directly in the source code of mteb 2.0.5, 2.12.30 and 2.21.0. |
| `[EXEC]` | Executed in a sandbox against those three mteb versions (see 0.2 for the limits of that test). |
| `[STATS]` | Taken from the descriptive-statistics file shipped inside the mteb package. **Not** verified against the Hugging Face dataset. |
| `[MEM]` | From my memory (model names, licences, sizes). **Unverified.** Must be checked against the model card in Phase 0. |
| `[EST]` | Back-of-envelope estimate. Must be replaced by a measurement. |
| `[DEC]` | A decision this blueprint makes. |
| `[EXP]` | A component that is **not accepted** until it passes an experiment gate (Section 6). |
| `[RISK]` | Research-grade or fragile; optional, off by default. |

I cannot search the web or Hugging Face from here, so model names, licences and any paper/dataset references may be wrong or out of date. Treat every `[MEM]` item and every citation as something to double-check.

### 0.2 What was actually verified

- **Read:** all four guideline pages; the mteb source for the retrieval path in 2.0.5, 2.12.30, 2.21.0.
- **Executed** (real mteb code, real `AppsRetrieval` task metadata, 25-document synthetic corpus, `torch`/`sentence-transformers` **stubbed** because the sandbox had no disk for them):
  1. Dispatch behaviour for seven class shapes (Section 3.3) — identical on all three versions.
  2. The official call shape `mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})` with a `SearchProtocol`-enabled `AbsEncoder`: `encode()` was never called, `batch_size` reached `search()`, the result JSON was produced.
  3. The guideline sample's final step (`json.dump(task_result.to_dict())`) **fails** on 2.12.30 and 2.21.0 (`datetime` not serialisable).
  4. A score-gap pitfall in `pytrec_eval` (Section 3.6) that makes NDCG and MRR silently disagree.
  5. The adapter skeleton in Appendix A passes on all three versions.
- **Not verified:** the real dataset, any model card, real CPU throughput, the standard `SearchEncoderWrapper` path with real torch (read, not run), the judges' harness, network/HF availability at judging time.

### 0.3 Executive summary

1. ACI is a **retrieval engine**, not a RAG system and not a code-generation agent. Generation is out of scope `[DOC]`.
2. One MTEB-agnostic core library (`aci`) serves four surfaces: the MTEB adapter, a REST API, a CLI/UI, and read-only agent tools.
3. **The benchmark is just one snapshot.** P0 indexes the 8,765-document corpus as a single immutable snapshot; P1 is many snapshots sharing content-addressed caches; the Bonus is a union index over lineage nodes of those snapshots. One code path, three requirements.
4. **MTEB integration:** the class `PrePostPipelineEncoder(AbsEncoder)` *also* implements `index()` and `search()` (mteb's `SearchProtocol`). mteb then calls our search directly `[SRC][EXEC]`. `encode()`/`similarity()` remain valid as a dense-only fallback. The earlier "basis-vector / score-vector" workaround is **rejected**.
5. **P0 pipeline (default):** query/doc views → BM25 ∥ dense (1–2 small CPU-feasible embedders) → RRF → LightGBM LambdaMART reranker (accepted only if it clears its gate; plain RRF otherwise) → strictly-ordered scores. No LLM, no graph, no agent on the P0 path.
6. **The neural cross-encoder reranker in the old design is not viable on CPU** for 3,765 long queries `[EST]` (Section 5.9). It survives only as a gated experiment on low-margin queries.
7. **Accuracy is established, not asserted:** sealed test split, dev splits built from APPS-train, paired-bootstrap gates, a test-run ledger, parity tests that prove the adapter and BM25 are bug-free (Section 6).
8. **P1:** content-addressed blob store, immutable snapshots, per-blob derived-artifact caches, atomic pointer flip, rollback, and an *incremental == full-rebuild* invariant (Section 9).
9. **Bonus:** lineage identity + content-similarity alignment, dedup by content, grouped ranking, a diff ("delta") retrieval view. A full semantic evolution graph is `[RISK]` and optional (Section 10).
10. **Agent:** bounded, read-only, deterministic controller by default; an optional small local LLM only for decomposition. Never invoked on the P0 path. Enabled by default only if it beats the no-agent baseline on a held-out multi-hop set (Section 11).
11. Everything runs **offline, CPU-first, with pinned dependencies and model hashes, and with no paid API**. Voyage Code (hosted, paid) is removed.
12. Target language is **Python** (the dataset is Python). JavaScript/TypeScript parsing (Tree-sitter/Babel) is removed from the core.

---

### 0.4 Map: your 17 requested deliverables → sections

| # | Deliverable | Section |
|---|---|---|
| 1 | Final architecture | 4.1–4.2 |
| 2 | Component responsibilities | 4.3 |
| 3 | Data flow | 4.4 |
| 4 | Control flow | 4.5 |
| 5 | Storage / index design | 8.1–8.3, 9 |
| 6 | Model choices | 7 |
| 7 | API / interface contracts | 8.4–8.6, Appendix A |
| 8 | Evaluation architecture | 6 |
| 9 | P0 implementation architecture | 3, 5 |
| 10 | P1 implementation architecture | 9 |
| 11 | Bonus implementation architecture | 10 |
| 12 | Fallback / degradation architecture | 12 |
| 13 | Security architecture | 13 |
| 14 | Observability architecture | 14 |
| 15 | Deployment architecture | 15 |
| 16 | Testing strategy | 16 |
| 17 | Reproducibility strategy | 15.2–15.6, 6.2, 6.6 |
| — | Phase-by-phase plan · red-team · final checklist | 17 · 18 · 18.1 |

**Sections:** 1 Requirements · 2 Audit of the old design · 3 MTEB/P0 integration · 4 Final architecture · 5 Retrieval engine · 6 Accuracy framework · 7 Models and technology · 8 Storage and contracts · 9 P1 · 10 Bonus · 11 Agent · 12 Fallbacks and edge cases · 13 Security · 14 Observability · 15 Deployment and reproducibility · 16 Testing · 17 Implementation plan · 18 Red team · Appendices A–C.

---

## 1. Requirements: what Samsung actually requires vs. what we add

### 1.1 Official (source of truth)

| # | Requirement | Source |
|---|---|---|
| R1 | Given a library of code and a natural-language query, return a **ranking** of snippets by relevance. Retrieval improves when a single pass is not enough. | `[DOC]` |
| R2 | **P0:** retrieval accuracy, evaluated on the CoIR **AppsRetrieval test split** via the **MTEB** library; metrics **NDCG@10** and **MRR**. | `[DOC]` |
| R3 | **P1:** retrieval across code versions; index/cache rebuild or update in reasonable time. | `[DOC]` |
| R4 | **Bonus:** retrieval across *all* versions; similar snippets across versions make ranking hard. | `[DOC]` |
| R5 | Post-retrieval generation/explanation is out of scope. | `[DOC]` |
| R6 | Any retrieval/pre/post-processing method is allowed; **runtime, GPU requirement and model size are evaluated**; models are expected to be small enough for CPU with minimal GPU. | `[DOC]` + FAQ as relayed |
| R7 | The dataset is the **Python** APPS set; the JavaScript example in the slides is obsolete. | FAQ as relayed |
| R8 | Deliverables: MTEB result JSON, JSON inference results, PPT, GitHub repo with run instructions, release artifacts, demo video. | `[DOC]` |
| R9 | Hands-on: judges may run the code, ask dataset-like queries, and exercise P1 and the Bonus; the demo should show the system working and its speed. | `[DOC]` |
| R10 | Interface shape: `PrePostPipelineEncoder(AbsEncoder)`, zero-argument constructor, `mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})`, JSON from `task_result.to_dict()`. | `[DOC]` |

**Two inconsistencies inside the guidelines, both to be confirmed with the organizers:**

- One page says to submit **"a csv file with the responses"**, another says to submit the **MTEB JSON**. `[DEC]` Produce both (plus MTEB's predictions folder); the CSV is a free export of the same ranking.
- The sample writes JSON with `json.dump(...)` but never imports `json`, and on current mteb the call **raises** (Section 3.5). `[DEC]` Our repo ships a helper that works on all tested versions.

### 1.2 Constraints you set

Locally runnable · **no paid external API or API key** · local/open-source models · CPU-first, minimal GPU · offline-capable · pinned and reproducible · arbitrary Python code · priority order **retrieval accuracy → correctness → reproducibility → latency → resource efficiency → robustness → P1 → Bonus → demo polish**.

### 1.3 Everything else is ours (and must justify itself)

Repository upload UI, call graph, symbol index, agent, hierarchical retrieval, dashboards, React UI, evolution graph: **none of these is required by Samsung.** Each stays only if it measurably helps P0, P1 or the Bonus, or is explicitly demo-only and cannot touch the P0 path (Section 2).

### 1.4 What is *not* required, so we do not build it

Multimodal input · answer generation · a chat UI · a graph database · a vector-database server · a multi-agent swarm · JavaScript/TypeScript support · any hosted model.

---

## 2. Audit of the existing design

Verdicts: **KEEP**, **MODIFY**, **REPLACE**, **DEFER** (build later, behind a gate), **DROP**. Columns answer your eight questions in compressed form: *Required by Samsung? · helps P0/P1/Bonus/demo? · works with MTEB? · cost/risk · alternative.*

| Component in old design | Required? | Helps | MTEB-compatible? | Cost / risk | Verdict |
|---|---|---|---|---|---|
| BM25 lexical index | No | **P0** (literals, I/O strings, constants), P1 | Yes, inside `search()` | Cheap, CPU-trivial | **KEEP.** Code-aware tokenizer; own thin CSR implementation (Section 7) |
| Dense code embeddings | No (but effectively necessary) | **P0** | Yes | Dominant CPU cost `[EST]` | **KEEP, size-constrained.** Model chosen by bake-off, not by name |
| Voyage Code family | — | — | — | **Hosted paid API**, violates your constraint | **DROP** |
| RRF fusion | No | P0 baseline | Yes | Trivial | **KEEP as baseline;** superseded if LTR clears its gate |
| Hard-coded fusion weights | — | — | — | Arbitrary | **REPLACE** with tuned weights / learned ranker |
| Qwen3-Reranker-0.6B on top-100 | No | P0 upside | Yes | **Infeasible on CPU** for 3,765 long queries `[EST]` | **REPLACE** with LightGBM LambdaMART; cross-encoder becomes `[EXP]` on low-margin queries only |
| Symbol index | No | Repo mode only | n/a for APPS (docs are scripts, names are generic) | Low | **P0: DROP. Repo mode: KEEP** |
| AST / structural index | No | P0 only as *features*; repo mode chunking, symbols, graph | Yes | Low with stdlib `ast` | **MODIFY:** Python `ast`, not Tree-sitter/Babel |
| Call/reference graph | No | Repo mode, agent, demo | n/a for APPS | Medium (dynamic-language uncertainty) | **P0: DROP. Repo mode: KEEP** with confidence tiers |
| Graph DB / Qdrant / pgvector | No | None at this scale | — | Extra services, ops surface, zero benefit | **DROP.** Exact vector search + adjacency arrays + SQLite |
| Query classifier + routing | No | Repo mode | Trivial for APPS (all queries are long problem statements) | Misrouting can *lose* recall | **MODIFY:** deterministic rules; routing may only *add* retrievers or shift bounded weights, never remove the BM25+dense baseline |
| LLM query rewriting | No | Unproven | Adds latency + nondeterminism | High | **DROP from P0.** Deterministic segmentation instead |
| LLM agent (Qwen3-4B) | No | Demo, repo multi-hop | Off the P0 path | Latency, flakiness | **DEMOTE.** Deterministic controller default; LLM optional and gated |
| Hierarchical retrieval | No | Unproven at APPS scale | — | Complexity | **DEFER.** Only above a measured unit-count threshold |
| Evidence verification / no-invented-lines | No | Correctness | Yes | Low | **KEEP** (a core rule) |
| Confidence estimation | No | Routing, UI | Not used for ranking | Low | **KEEP** as a calibrated routing signal only |
| Version manager | Implicitly (P1) | **P1, Bonus** | n/a | High if naïve | **REDESIGN:** content-addressed, immutable snapshots (Section 9) |
| Evolution graph (SPLIT/MERGE/REPLACED) | Implicitly (Bonus) | Bonus | n/a | `[RISK]` fragile | **DEFER Tier 3;** ship lineage identity + similarity alignment |
| "Basis-vector / score-vector" MTEB trick | — | — | **Fragile — see 3.4** | Breaks on hands-on queries | **REJECT** |
| JS/TS parsers | No | None (dataset is Python) | — | Wasted effort | **DROP** |
| Prometheus/Grafana stack | No | Observability | — | Extra infra | **MODIFY:** `/metrics` + JSON logs + a built-in stats page; Grafana optional |
| Ingestion (GitHub/ZIP/folder/files) | No | Demo, P1 | — | Security surface | **KEEP, hardened,** secondary to P0 |
| React UI | No | Demo only | — | Time sink | **KEEP minimal;** one static bundle served by the API |
| Multimodal RAG | No | None | — | Pure cost | **DROP** (already decided) |

---

## 3. MTEB / P0 integration — resolved precisely

### 3.1 What the harness does (verified)

Facts below are `[SRC]` unless marked `[EXEC]`; they hold identically in 2.0.5, 2.12.30 and 2.21.0.

1. `mteb.evaluate(model, tasks, encode_kwargs=..., cache=..., overwrite_strategy=..., prediction_folder=...)` resolves the model via `_sanitize_model`, which reads `model.mteb_model_meta` for the **name and revision used as the cache key**.
2. For a retrieval task, `AbsTaskRetrieval._evaluate_subset` picks the search model with exactly this rule:

```
if   isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol): SearchEncoderWrapper(model)
elif isinstance(model, CrossEncoderProtocol):                                      SearchCrossEncoderWrapper(model)
elif isinstance(model, SearchProtocol):                                            model   # used directly
```

3. `SearchProtocol` needs `index(corpus, ...)` and `search(queries, ..., top_k, encode_kwargs, top_ranked=None, ...)` returning `{query_id: {doc_id: score}}`.
4. In the standard `SearchEncoderWrapper` path with no ANN backend: `index()` merely stores the corpus; `search()` then **encodes the queries first**, then encodes the corpus in chunks of 50,000, calls `model.similarity(query_embeddings, corpus_chunk_embeddings)`, and keeps the top-k. So an encoder can express *only* "vector in, vector out, similarity function" — never BM25, fusion or reranking.
5. Scoring: NDCG via `pytrec_eval` (`ndcg_cut`), MRR via mteb's own function. `k_values = (1, 3, 5, 10, 20, 100, 1000)`, `top_k = 1000`. Result keys: `ndcg_at_10`, `mrr_at_10`, … (151–152 keys). `main_score = ndcg_at_10`. `[EXEC]`
6. `AppsRetrieval`: dataset `CoIR-Retrieval/apps`, revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`, licence MIT, test split only.
7. mteb ships its own `SearchProtocol` implementers — a BM25 baseline and (2.21.0) a `HybridSearch` with RRF/DBSF/RSF fusion. Custom retrieval is the **intended** extension point, not a hack.

### 3.2 Dataset shape (`[STATS]`, verify in Phase 0)

| | Value |
|---|---|
| Queries (test) | 3,765; length 152–5,742 chars, mean ≈ 1,670 → **long natural-language problem statements** |
| Documents | 8,765; length 5–289,048 chars, mean ≈ 573 → short Python solutions with a huge outlier; 8,754 unique texts (11 duplicates) |
| Qrels | exactly **1 relevant doc per query**; 3,765 distinct relevant docs |

Consequences: (a) NDCG@10 and MRR@10 are both pure functions of the *rank of a single gold document*; (b) queries are far longer than documents, so query-side truncation and query embedding cost dominate; (c) 5,000 documents are never relevant (my hypothesis: train-split solutions — **unverified**, audited in Phase 0); (d) 11 duplicate texts create unbreakable ties.

### 3.3 Executed dispatch results (`[EXEC]`, identical on all three versions)

| Class shape | Encoder / Search / Cross-encoder protocol | Path mteb takes |
|---|---|---|
| `AbsEncoder` only | T / F / F | `SearchEncoderWrapper` (encode + similarity) |
| + `index` **and** `search` | T / **T** / F | **model used directly** ← what we want |
| + `index`, `search`, **`predict`** | T / T / **T** | **cross-encoder wrapper** ← trap: never define `predict` |
| + `index` only | T / F / F | encoder wrapper; **`index` silently ignored** |
| + `search` only | T / F / F | encoder wrapper; **`search` silently ignored** |
| `mteb_model_meta` as a `@property` | T / T / F | used directly (works) |
| `mteb_model_meta = None` | T / T / F | passes dispatch, but **crashes** when `prediction_folder` is set (`None.name`) and collapses the cache key to `no_model_name/available` |

### 3.4 Verdict on the "basis-vector / score-vector" workaround — **rejected**

The idea: encode documents as one-hot basis vectors and queries as full score vectors, so cosine similarity equals a hybrid score. Why it fails:

1. **Call order.** `SearchEncoderWrapper` encodes queries *before* documents `[SRC]`. At query time the encoder has not seen the corpus, so it would have to side-load the benchmark corpus itself, outside the harness's data path.
2. **Identity.** Documents arrive as text only; mapping text → basis index breaks on the **11 duplicate texts** and on any harness that reorders or chunks the corpus.
3. **Memory.** The vectors are N-dimensional: 8,765 × 8,765 float32 ≈ 307 MB for documents plus 3,765 × 8,765 ≈ 132 MB for queries — for no benefit.
4. **Hands-on failure.** A judge's *new* query or *new* corpus has no basis vectors. The system would work only on the exact benchmark.
5. **Version coupling.** It depends on internal call order that differs with `index_backend` and could change between mteb releases.

It is technically constructible but fragile and would look like gaming the harness. **Replaced by 3.5.**

### 3.5 The correct integration `[DEC]`

`PrePostPipelineEncoder(AbsEncoder)` implements **both** interfaces:

```
                     mteb.evaluate(PrePostPipelineEncoder(), [AppsRetrieval], encode_kwargs={"batch_size": 64})
                                            │
                        isinstance(model, SearchProtocol)? ── yes ──►  model.index(corpus)   → build/load ONE snapshot
                                            │                          model.search(queries) → full ACI pipeline
                                            no (harness forces the encoder path)
                                            ▼
                        SearchEncoderWrapper → model.encode(...) + model.similarity(...)   (dense-only fallback)
```

Rules (each was violated in at least one executed failure case):

1. Zero-argument constructor; heavy work (model loading) is lazy.
2. Define `index` **and** `search`; never `predict`.
3. Accept `**kwargs` and default `num_proc=None` — 2.0.5's `SearchProtocol` has no `num_proc`, newer versions pass it.
4. `mteb_model_meta` is a real `ModelMeta` whose `revision` is `git-sha + config-hash`. Use the helper in Appendix A: 2.0.5 has no `ModelMeta.create_empty()`.
5. Return **strictly decreasing** scores for every query (3.6), at least 2 results per query, all finite, covering every query id.
6. The adapter contains **no ranking logic** (≈150 lines). It converts mteb datasets to engine calls and back, so the API, CLI and UI exercise the identical engine.
7. `index()` builds/loads an ordinary ACI snapshot keyed by content hash — the benchmark is not special-cased.
8. The official entry script uses `cache=None` (or `overwrite_strategy="always"`). The default result cache silently returns stale results when the model name/revision are unchanged `[SRC]`.

### 3.6 The score-ordering trap (`[EXEC]`) and the fix

`pytrec_eval` (NDCG) treats score gaps below roughly 1e-6 (at score magnitude ≈ 5; ≈ 1e-7 at magnitude 0.5) as **ties** and breaks ties by **reverse document-id order**. mteb's own MRR compares Python floats and sees no tie. Result: with a naive `score = previous − 1e-9`, the adapter produced `mrr_at_10 = 1.000` but `ndcg_at_10 = 0.200` on the same run. Gap measurements at magnitude 5.0: 1e-9…1e-7 → tie (gold demoted), ≥1e-6 → correct.

`[DEC]` Emit scores squashed into (1, 2) with an **absolute gap ≥ 1e-5**. Float32 spacing there is ≈1.2e-7, drift over 1,000 results ≤ 0.01, and there is no underflow. Verified: NDCG@10 = MRR@10 = 1.0 with ties injected in the raw engine scores, on 2.0.5, 2.12.30, 2.21.0. Exact ties in the *source* ranking are resolved by our own deterministic key `(−score, doc_id)`, never by the harness.

### 3.7 Artifacts produced by the official run

| Artifact | How | Note |
|---|---|---|
| `AppsRetrieval_results.json` | `write_official_json(task_result, path)` (Appendix A) | Same structure as the sample; `datetime` handled |
| `predictions/AppsRetrieval_predictions.json` | `prediction_folder=` | Per-query rankings from mteb itself `[EXEC]` |
| `AppsRetrieval_ranking.csv` | export from predictions (`query_id, rank, doc_id, score`, top-100) | Covers the "csv" wording |
| `run_manifest.json` | our code | Hardware, thread counts, package versions, model hashes, config hash, git SHA, cold/warm flag, cache hit ratio, per-stage timings |

### 3.8 Judge-harness scenarios and behaviour

| Scenario | What happens | Outcome |
|---|---|---|
| S1 Judge runs our repo script | Full pipeline | Best |
| S2 Judge runs the official sample with our class imported | mteb detects `SearchProtocol` → full pipeline | Best — requires the zero-arg constructor and a working default profile |
| S3 Harness forces the encoder path | `encode`/`similarity` dense-only fallback | Valid, weaker; still a well-preprocessed dense retriever |
| S4 Different mteb version | Compat-tested on 2.0.5 / 2.12.30 / 2.21.0; lockfile pins one | Version-drift risk is covered by CI, not hope |
| S5 Judge's environment is offline | Models and dataset must already be cached; `HF_HUB_OFFLINE=1` | Documented in the README; artifacts in the GitHub release |

### 3.9 Integrity of the official number

The headline result must come from a **cold run** of the shipped code. Optional caches (document embeddings keyed by content hash + model revision) may accelerate reruns but must satisfy the **cache-parity test**: warm and cold runs produce identical rankings. The manifest records `cold|warm` and the cache hit ratio; cold-run wall-clock is reported next to the accuracy number.

---

## 4. Final architecture

### 4.1 Layered view

```
        ┌──────────────────────── ACCESS SURFACES (thin, no ranking logic) ────────────────────────┐
        │  MTEB adapter   │   REST API (FastAPI)   │   CLI   │   Static UI   │   Agent tools (RO)  │
        └────────┬────────┴───────────┬────────────┴────┬────┴───────┬───────┴──────────┬──────────┘
                 ▼                    ▼                 ▼            ▼                  ▼
        ┌────────────────────────────────────────────────────────────────────────────────────────┐
        │                              aci.engine  (pure library)                                │
        │  QueryProcessor → [BM25 ∥ Dense A ∥ Dense B ∥ (Symbol/Exact/Graph in repo mode)]        │
        │      → Candidate fusion (RRF) → LTR reranker → PostProcess → Verify → Confidence        │
        │      → (gate) → Bounded retrieval agent (repo mode only)                                │
        └───────────────┬─────────────────────────────────────────────┬──────────────────────────┘
                        ▼ read-only snapshot handle                     ▼
              ┌──────────────────────┐                        ┌────────────────────────┐
              │   Snapshot reader    │                        │  Model runtime (local) │
              │ (immutable, checksummed)                      │  embedder(s), LTR, opt. LLM │
              └──────────┬───────────┘                        └────────────────────────┘
                         ▼
   ┌───────────────────────────────── STORAGE (local disk, no server) ──────────────────────────────┐
   │ Blob CAS │ Per-blob artifact caches (tokens, AST facts, embeddings) │ Snapshots │ SQLite meta   │
   └───────────────────────────────────────────▲─────────────────────────────────────────────────────┘
                                               │ single writer, per-repo lock
      Ingest → Filter → Diff vs parent → Parse/Chunk (sandboxed) → Featurize → Embed (changed only)
            → Assemble snapshot → Validate (checksums + canary self-retrieval) → Atomic activate
```

`[DEC]` One Python process serves API + engine. Sandboxed worker processes do parsing; embedding uses intra-op threads. Indexing runs as a background job in a separate process. **No message broker, no microservices, no external database.**

### 4.2 Core abstractions

| Object | Definition |
|---|---|
| `CodeUnit` | The retrievable thing: `unit_id, repo_id, snapshot_id, path, qualname, kind, start_line, end_line, blob_sha256, chunk_sha256, language, flags`. For APPS, one document = one unit of `kind="snippet"` and `path = doc_id`. |
| `Snapshot` | Immutable, checksummed directory describing one version: unit table + BM25 CSR + vector matrices + symbol table + graph + manifest. `snapshot_id = sha256(canonical manifest)`. |
| `SnapshotHandle` | Read-only handle resolved **once** at request start and passed to every sub-retriever. No component may look up "the current version" on its own. |
| `Retriever` | `retrieve(query_views, handle, k) → list[Candidate]`; stateless w.r.t. version. |
| `Reranker` | `rerank(query_views, candidates, handle) → list[Scored]`. |
| `Candidate` | `unit_id, snapshot_id, per-retriever (rank, score), flags`. The fusion stage asserts every candidate's `snapshot_id` equals the handle's. |

**The benchmark is one snapshot.** P0 builds a snapshot from the MTEB corpus. P1 builds many. The Bonus builds a union view over them. There is one indexing code path.

### 4.3 Component responsibilities

| Component | Responsibility | Must not |
|---|---|---|
| Ingestion | Turn a source (git, folder, ZIP, JSONL, MTEB corpus) into a verified file/blob list | Execute anything; follow symlinks outside the root |
| Parser workers | `ast`-based unit/symbol/edge extraction under resource limits | Import or run repo code |
| Featurizer | Token counts, literal/I-O signature features, doc views | Use APPS-train data at document level |
| Embedder | Batched, length-sorted, cached embeddings | Download at runtime |
| Snapshot builder | Assemble, validate, atomically activate | Expose a partial snapshot |
| Query processor | Normalise, segment, extract literals; never destroy the original | Call an LLM on the P0 path |
| Retrievers | Produce ranked candidate lists | Cross version boundaries |
| Fusion + LTR | Combine signals; strict deterministic ordering | Depend on other queries in the batch |
| Post-processor | Dedup groups, restore exact source, attach evidence | Invent code, lines or versions |
| Confidence | Calibrated routing signal | Be presented as a probability unless calibrated |
| Agent | Read-only bounded investigation in repo mode | Run on the P0 path; call write/shell/network tools |
| Observability | Metrics, logs, traces, health | Log raw query text by default |

### 4.4 Query-time data flow and latency budget (proposals, to be measured)

| Stage | Work | Budget (8-core CPU, ≤100k units) |
|---|---|---|
| 1 Prepare | normalise, segment, literals, hash | ≤ 5 ms |
| 2 Embed query | one query, ~450 tokens, ≈110M-class model | 100–300 ms `[EST]` (dominant) |
| 3 Retrieve (parallel) | BM25 (CSR ops) ∥ dense (matmul) ∥ repo-mode retrievers | ≤ 30 ms |
| 4 Fuse | RRF over ≤ 300 candidates | ≤ 2 ms |
| 5 Rerank | LightGBM over ≤ 300 candidates | ≤ 10 ms |
| 6 Post-process + verify | read blobs, verify hashes, dedup groups | ≤ 20 ms |
| **Total p50 target** | | **≈ 150–400 ms `[EST]`**, p95 ≤ 1 s |

Query-embedding cost dominates; hence the query-embedding cache and ONNX/int8 study (Phase 3).

### 4.5 Control flow

```
search(req):
    handle = resolve_snapshot(req.repo, req.version)            # once; immutable
    q      = prepare(req.query)                                 # views, literals, class
    cands  = parallel(bm25(q,handle), dense_A(q,handle), dense_B?(q,handle), repo_retrievers?(q,handle))
    cands  = rrf(cands)                                         # baseline fusion; always available
    ranked = ltr(cands) if ltr_healthy else cands              # learned ranker, falls back to RRF
    ranked = post_process(ranked, handle)                       # dedup groups, verify, exact source
    conf   = calibrated_confidence(ranked)
    if gate_open(req, q.class, conf, budget):                   # repo mode only; never in the MTEB path
        ranked = bounded_agent(req, q, ranked, handle)          # may only add evidence; final score still from ranker
    return ranked, conf, timings, degradations
```

Routing rule `[DEC]`: routing may **add** retrievers or shift fusion weights within bounds. It may never remove the BM25 + dense baseline. This bounds the damage of a wrong classification.

### 4.6 Terminology (used consistently below)

| Term | Meaning here | In P0? |
|---|---|---|
| **Retrieval engine** | Indexes + first-stage retrievers + fusion producing a candidate ranking | Yes |
| **Reranking** | Re-scoring a small candidate set with richer features or a neural model | Yes (LightGBM); neural = `[EXP]` |
| **Agentic retrieval** | A bounded, read-only loop that issues *retrieval calls*, inspects evidence, refines, stops | **No** |
| **RAG** | Retrieval feeding a generator | **Not part of this product** |
| **Generation** | Producing prose/code | Out of scope `[DOC]` |

### 4.7 What each requirement actually needs

| Mechanism | P0 | P1 | Bonus | Demo/product |
|---|---|---|---|---|
| BM25 + dense + fusion + LTR | **Required** | Reused | Reused | ✓ |
| Content-addressed cache | Optional accelerator | **Required** | Required | ✓ |
| Snapshots + atomic activation | Used (one snapshot) | **Required** | Required | ✓ |
| Lineage / alignment / delta view | — | — | **Required** | ✓ |
| Symbol / graph / classifier | — | — | Helps | ✓ (repo mode) |
| Agent | — | — | — | ✓ (gated) |
| UI, dashboards | — | — | — | ✓ |

---

## 5. Retrieval engine (P0 first)

### 5.1 Ground rules

1. **Query-independence invariant.** The ranking for query *q* must not depend on which other queries are in the same batch. This forbids transductive tricks (dual-softmax or inverted-softmax over the query set, Hungarian assignment over queries) in the headline number — a judge's single query would score differently from the same query inside the 3,765-query batch. Tested by a metamorphic test (batch of 3,765 vs. one-at-a-time must give identical rankings).
2. **No dataset-specific priors.** No use of document-id patterns or order, no qrels-derived priors, no demotion of documents because they resemble APPS-train solutions (Section 6.2, rules R4–R5).
3. **Hypotheses, not facts.** Every idea below about *why* it should help on APPS is a hypothesis tested in Section 6.

### 5.2 Query processing (problem statement → views)

APPS statements are long: story, task, input format, output format, examples, notes. Section markers such as `-----Input-----` are common `[MEM]` — verify. `[DEC]` A deterministic segmenter splits on such markers and **falls back to the whole text** when none are found. The original text is always kept as a view (non-destructive).

| View | Content | Used by |
|---|---|---|
| `Q_full` | whole statement, head+tail truncated to the model's budget | dense |
| `Q_task` | text before the input/output sections (story + task) | dense `[EXP]` |
| `Q_io` | input + output specification | dense/LTR features `[EXP]` |
| `Q_lex` | tokenised statement with numbers normalised (`10^9+7`, `1e9+7`, `1000000007` → one token) and rare-term selection | BM25 |
| `Q_lit` | numeric constants, quoted strings, all-caps answer tokens (`YES`/`NO`), modulus | LTR features + BM25 boosts `[EXP]` |
| `Q_examples` | sample input/output | excluded from dense by default `[EXP]` |

Whether `Q_task`/`Q_io` beat `Q_full`, and which truncation policy wins (head-only / head+tail / segment-aware), are experiments E-QSEG and E-TRUNC. **A wrong query prompt format silently costs accuracy**: use each embedder's documented query/document prefixes `[MEM]`.

### 5.3 Document processing (code → views)

| View | Content | Used by |
|---|---|---|
| `D_raw` | normalised source (newline/tab/trailing-space normalisation), head+tail truncated for dense | dense |
| `D_lex` | identifier-split tokens (snake/camel/digit boundaries), lowercased; numeric literals normalised as in `Q_lex`; string-literal contents kept; Python keywords and punctuation dropped | BM25 |
| `D_sig` | I/O signature and idiom features: count of `input()`/`sys.stdin` reads, `map(int, …)`, `split()`, test-case loop shape, `print` count and literal outputs, recursion, `sys.setrecursionlimit`, use of `heapq`/`bisect`/`deque`/`Counter`/`itertools`/`math`, modulus constants, 2-D list initialisation | LTR features |
| `D_nl` | LLM-written natural-language description | `[EXP]` E-ENR, off by default |

**Parser policy `[DEC]`:** try `ast.parse` in a resource-limited subprocess; on failure (syntax error, Python-2 syntax, `RecursionError`, `MemoryError`, timeout) fall back to a tolerant tokenizer, then to regex tokens. A parse failure sets a flag and disables only AST-derived features; the document stays retrievable. The fraction of non-Python-3 documents in the corpus is unknown — measured in Phase 0.

### 5.4 Truncation and chunking

- Query lengths reach ≈1,500 tokens `[EST]`; a 512-token cap would cut input/output specifications. E-TRUNC compares caps of 512 and 1,024 and the three truncation policies.
- Documents average ≈190 tokens `[EST]`; the 289,048-character outlier and any other long tail exceed any sane cap. `[EXP]` E-LONG: split long documents at top-level `def`/`class` boundaries plus overlapping windows, embed the chunks, and score a document by max-pooled chunk similarity. BM25 always sees the full document.
- **The document, not the chunk, is the unit returned to MTEB** (qrels are per document id). Chunks are an internal scoring representation.
- **Repo mode** uses structure-aware units (module header, class header, function/method, module body). Each embedded text is prefixed with `path :: qualname :: signature` (a contextualised chunk) `[EXP]`.

### 5.5 First-stage retrievers

**BM25 (lexical).** `[DEC]` Own thin CSR implementation on numpy/scipy (Section 7), so that P1 can assemble a snapshot from cached per-blob term counts. Tuned parameters: `k1`, `b`, query-term cap `M` (long queries add noise), query-TF saturation, stopword/stemming choices. Must reproduce MTEB's built-in `bm25s` ranking when configured identically (parity test P2).

**Dense.** One primary embedder (`Dense A`); an optional second from a different family (`Dense B`) only if it clears the cost/benefit gate. Embeddings are L2-normalised; scored by exact matrix multiplication (Section 7.4). Length-sorted global batching (the standard MTEB wrapper batches in dataset order).

Candidate depth `K1` per retriever (initial guess 200, decided from the Recall@K curve, not by intuition).

### 5.6 Fusion

Baselines, in order: RRF (k = 60) → weighted RRF → per-query z-score linear fusion → CombSUM. The winner is whichever is best on the dev-validation split *and* stays robust across strata. Fusion is also the **fallback** when the LTR stage is unhealthy.

### 5.7 Learned ranker (LightGBM LambdaMART) `[EXP]`

Why: it directly optimises NDCG, costs milliseconds on CPU, is interpretable, and replaces hand-set weights. It is the main P0 upside that fits the CPU budget.

| Feature group | Examples |
|---|---|
| Retriever signals | dense cosine (A/B), BM25 score, per-list rank, RRF score, per-query z-score, gap to top-1 / top-10, number of lists containing the doc in their top-20 |
| Literal overlap | shared normalised numbers, shared string literals / answer tokens, modulus match, shared rare identifier-like tokens |
| I/O signature | query mentions test cases ↔ document has a test-case loop; expected vs. actual input-read count; presence of `sys.stdin`/`split`/`map(int…)` |
| Document/query shape | log lengths, parse-ok flag, comment ratio, unique-token ratio, duplicate-group size, query has examples |

Rules: **no document-id, text-hash or any feature conditioned on APPS-train documents.** Training uses dev-fit queries only, query-level cross-validation to estimate out-of-fold performance, early stopping on dev-val, small trees (tens of leaves, a few hundred trees), monotone constraints on the raw retriever scores. The model is stored as LightGBM text (no pickle). Feature code is shared between training and serving; a **skew test** recomputes features both ways.

Main risk: train/test distribution shift (dev corpus size and composition differ from the test corpus). Mitigations: per-query relative features, dev tasks built to mimic test statistics, holdout gap monitoring; fallback to RRF if the holdout gain vanishes.

### 5.8 Post-processing, duplicates, ordering

- **Exact duplicates** (11 in the benchmark corpus `[STATS]`): both ids are returned (either may be the gold), ordered by the deterministic key `(−score, doc_id)`; in UI mode they are collapsed into a group.
- **Ordering:** all ties are resolved by us; scores are emitted with the gap rule of Section 3.6.
- **Evidence (repo mode):** every returned unit is re-read from the blob store by `(blob_sha256, span)`; the hash is verified; the version membership is verified. Nothing is synthesised.

### 5.9 Compute budget — the reason the old reranker is replaced `[EST]`

Inputs: corpus 5,026,626 characters + queries 6,286,904 characters `[STATS]` ≈ **3 million tokens per embedding pass** at 3–4 characters/token. A transformer costs ≈ 2 × (non-embedding parameters) FLOPs per token. Effective CPU throughput of 100–500 GFLOPS is an assumption for an 8-core x86 machine including padding waste. **Replace with Phase-0 measurements.**

| Model class (non-embedding params) | FLOPs per pass | Time @100 GFLOPS | Time @500 GFLOPS |
|---|---|---|---|
| ≈25M (MiniLM/small class) | 1.5e14 | ≈25 min | ≈5 min |
| ≈110M (base class) | 6.6e14 | ≈1.8 h | ≈22 min |
| ≈450M (0.6B class) | 2.7e15 | ≈7.5 h | ≈1.5 h |
| **Cross-encoder 0.6B, top-20 for 3,765 queries** (≈75k pairs × ≈650 tokens ≈ 49M tokens) | 4.4e16 | **≈5 days** | **≈24 h** |

Conclusions `[DEC]`:
1. A 0.6B cross-encoder over top-K for the whole test set is **not** a CPU-feasible default. The bi-encoder embedding pass alone is hours at 0.6B.
2. Query embedding cost is the same order as corpus embedding cost: queries are ≈ 2.9× longer than documents on average (1,670 vs. 573 characters) but ≈ 2.3× fewer (3,765 vs. 8,765), so total query text is ≈ 1.25× the corpus text `[STATS]`.
3. **Profiles** (final contents from the bake-off): `lite` — one ≤≈150M embedder + BM25 + LTR; `standard` — adds a second embedder if the cold-run SLO holds; `max` — larger model, requires accelerator or a warm cache, never used for the headline number unless it also meets the SLO.
4. `[DEC]` Internal SLO: cold full-test run ≤ **60 minutes on 8 cores / 16 GB**. This is *our* proposal; Samsung stated no time limit — ask them. Levers, in order: length-sorted batching, ONNX Runtime/int8 (gate: ≤ 0.3-point NDCG@10 loss), shorter query views, smaller embedder.

### 5.10 Confidence (calibrated routing signal)

Features: top-1/top-2 margin, agreement of top-1 across retrievers, per-query score z-scores, softmax entropy over the top-10. Calibration: isotonic regression on dev-val → P(top-1 correct). Reported: reliability curve and ECE. Uses: gate for the optional cross-encoder/agent; UI shows high/medium/low, not a raw probability. Never used to reorder results.

### 5.11 Experiments that must earn their place `[EXP]`

| ID | Idea | Why it might help | Why it might not | Gate |
|---|---|---|---|---|
| E-QSEG | Query segmentation (`Q_task`, `Q_io`) | Story text is noise for code | Loses cues; markers absent | Section 6.5 |
| E-TRUNC | Length caps / head-tail policy | I/O spec sits at the end | Cost ↑ | Section 6.5 |
| E-LONG | Chunk + max-pool long docs | Avoids truncation | Tail is tiny | Section 6.5 |
| E-CE | Small/0.6B cross-encoder on **low-margin queries only** | Fixes close calls | Cost per pair; may not beat LTR | ΔNDCG@10 ≥ threshold **and** cost ≤ budget |
| E-ENR | Offline LLM-written NL description per document (index-time; content-hash cached; degrades gracefully if absent) | Bridges NL↔code vocabulary gap | Generation cost; reproducibility; not needed for new corpora at runtime | Large gain required; must run cold-reproducibly |
| E-FT | Contrastive fine-tuning of the embedder | Largest possible lift | **Contamination risk**: if the 5,000 non-relevant test documents are APPS-train solutions, train data overlaps the corpus | Only after the contamination audit; zero-shot and tuned numbers reported separately |
| E-HUB | Document-side hubness correction | Reduces "hub" documents | Any statistic derived from APPS-train queries would demote known train-solutions → forbidden (R4) | Only with task-independent query samples |
| E-QUANT | ONNX/int8 embedder | 1.5–3× CPU speed-up `[MEM]` | Accuracy loss | ≤ 0.3-pt NDCG@10 |
| E-SECOND | Second embedder (`Dense B`) | Complementary errors | +100% embedding cost | Cost-normalised gain rule |

### 5.12 Repo-mode additions (not on the P0 path)

- **Query classes** (deterministic rules first): `EXACT`, `SYMBOL`, `SEMANTIC`, `CALLER`, `CALLEE`, `REFERENCE`, `DATA_FLOW`, `VERSION`, `EVOLUTION`, `MULTI_HOP`. A small classifier or LLM may resolve ambiguity only behind the routing rule in 4.5.
- **Extra retrievers:** exact/literal (inverted index over identifiers and strings), symbol (qualified-name → units), graph (adjacency with confidence tiers `DIRECT/RESOLVED/INFERRED/POSSIBLE/UNRESOLVED`; never present inferred edges as certain).
- **Hierarchical narrowing** (module → file → symbol) is `[EXP]`, adopted only above a unit-count threshold where it measurably reduces latency without lowering NDCG.

---

## 6. How accuracy is established (evidence framework)

Nothing in this document promises an NDCG or MRR. A component enters the system only when it passes the gates below.

### 6.1 Splits

| Split | Source | Use |
|---|---|---|
| **TEST** | Official AppsRetrieval test (3,765 queries) | **Sealed.** Only at pre-declared milestones (M1 baselines, M2 fusion+LTR, M3 freeze). Proposed budget ≤ 10 runs before freeze; every run is ledgered. |
| **DEV-CV** | ≈80 % of APPS **train** problems (seeded split by problem) | All tuning, LTR training with 5-fold cross-validation, out-of-fold (OOF) evaluation, hard-query analysis. |
| **DEV-HOLD** | remaining ≈20 % of APPS-train problems | Touched once per milestone as a second opinion. |
| **DEV-SHORT** | first 1–2 sentences of DEV-HOLD statements | Non-regression guard only (hands-on queries may be shorter than APPS statements). Never used for tuning. |
| **REPO-QA** | 2–3 permissively licensed Python repos with history: docstring→function queries, call-graph questions, commit-message queries | Repo mode, P1, Bonus, agent gate. |

Construction of DEV tasks: queries = train problem statements; corpus = **one** solution per train problem (deterministic seeded pick), so each query has one gold and distractors are solutions to *other* problems — mirroring the test structure. The dev corpus (≈5,000 documents) is smaller than the test corpus (8,765), so absolute dev numbers will be optimistic; **decisions use paired deltas only.** Corpus-size sensitivity is checked by subsampling. The APPS-train source, its licence and the distribution of solutions per problem must be confirmed on the dev machine (Phase 0).

### 6.2 Integrity rules

| # | Rule |
|---|---|
| R1 | TEST is sealed: no inspection of per-query TEST failures; hard-query analysis is done on DEV only. |
| R2 | Every evaluation is appended to an experiment ledger (Section 6.9); TEST runs are pre-declared. |
| R3 | All splits and randomness are seeded; seeds are in the manifest. |
| R4 | **No priors from document identity or the answer key:** no use of document-id patterns/order, qrels-derived statistics, or any signal of the form "this test-corpus document resembles an APPS-train solution, so demote it". |
| R5 | **No transductive batch normalisation** in the headline number (query-independence invariant, 5.1). |
| R6 | **Contamination audit before any training** (LTR or fine-tuning): compare normalised hashes of training queries/documents against the test queries/corpus; report overlaps; exclude overlaps from training. |
| R7 | Zero-shot and tuned results are reported separately, with a plain statement of what was trained on what. |
| R8 | No external API; models pinned by revision hash and file checksums. |
| R9 | The headline result is a **cold run** of the shipped code; cache parity is tested. |
| R10 | Rejected components stay in the ledger with their numbers. |

### 6.3 Baseline ladder

Each rung answers one question. All report NDCG@10, MRR@10, Recall@{10, 50, 100, 200, 1000}, p50/p95 latency, peak RAM and cold-run time.

| Rung | System | Question it answers |
|---|---|---|
| B0 | mteb's built-in `bm25s` model through `mteb.evaluate` | Independent floor and harness sanity check |
| B1 | Our BM25 with matched tokenisation | Must equal B0 rank-for-rank (parity P2) |
| B2 | Small general embedder through the **unmodified** `SearchEncoderWrapper` path (needs real torch) | What a typical submission scores |
| B3 | Same model through our `SearchProtocol` path | Must equal B2 (parity P1): proves the adapter is bug-free |
| B4 | B3 + our query/document views, prefixes, truncation | Value of preprocessing |
| B5 | Tuned code-aware BM25 alone | Lexical ceiling |
| B6 | BM25 + dense, RRF | Value of hybrid |
| B7 | + second embedder | Value of diversity vs. cost |
| B8 | Tuned fusion | Value of tuning |
| B9 | + LightGBM ranker | Value of learning-to-rank |
| B10 | + gated cross-encoder (E-CE) | Value of neural rerank vs. cost |
| B11 | + enrichment / fine-tuning (only after R6) | Value of the riskiest levers |
| B12 | `lite` / `standard` / `max` profiles | Pareto frontier for the SLO |

### 6.4 Statistics

- Save per-query NDCG@10 and RR@10 vectors for every run.
- **Paired bootstrap** over queries (10,000 resamples, percentile 95 % CI); Holm correction across ≤ 10 comparisons per milestone.
- Noise intuition `[EST]`: with one gold document the per-query NDCG@10 has a standard deviation around 0.4, so an unpaired mean over 3,765 queries has SE ≈ 0.65 points. Differences below ≈ 0.5 point on TEST are inside noise; paired comparisons are tighter. This is why decisions use 5-fold OOF results over all of DEV-CV (≈4,000 queries) rather than a small validation slice.

### 6.5 Acceptance gates (proposed defaults; adjust once, before running)

| Gate | Requirement |
|---|---|
| **G1 accuracy** | On DEV-CV OOF: lower bound of the paired-bootstrap 95 % CI for ΔNDCG@10 > 0 **and** point Δ ≥ 0.5 pt. Components that are pure fallbacks/robustness need only non-inferiority (loss ≤ 0.2 pt). |
| **G2 holdout** | Same direction on DEV-HOLD. |
| **G3 strata** | No pre-declared stratum with ≥ 200 queries regresses by more than 1.0 pt: query-length terciles, document-length terciles, lexical-overlap terciles, parse-failure vs. parsed. |
| **G4 cost** | Accept if Δ ≥ 0.5 pt per +10 % cold-run time; hard cap = the SLO (5.9); latency budgets of 4.4 hold. |
| **G5 failure injection** | With the component killed/slowed, the fallback path engages and results stay valid. |
| **G6 simplicity** | If two options are statistically indistinguishable, keep the simpler and cheaper. |
| **G7 test confirmation** | Direction confirmed on TEST at the milestone. A contradiction triggers investigation of distribution shift — **never tuning on TEST**. |

### 6.6 Validating the harness itself (before trusting any number)

| Test | Assertion |
|---|---|
| **P1** dense parity | Our `SearchProtocol` path and the standard `SearchEncoderWrapper` path give identical rankings for the same embedder (float tolerance) |
| **P2** BM25 parity | Our BM25 (matched tokenisation) equals mteb's `bm25s` ranking |
| **P3** metric parity | NDCG@10 / MRR@10 recomputed independently from the predictions file equal the JSON to 1e-6 |
| **P4** query independence | Ranking of query *q* is identical alone and inside a 3,765-query batch |
| **P5** order/label invariance | Shuffling corpus order or relabelling document ids changes nothing except deterministic tie-breaks |
| **P6** cache parity | Warm-cache and cold runs give identical rankings |
| **P7** tie/gap test | Adversarial exact ties and sub-1e-6 gaps in raw scores still yield NDCG@10 = MRR@10 for a gold-at-rank-1 fixture (the pitfall of 3.6) |

### 6.7 Hard-query, negative and candidate analysis (DEV-CV OOF only)

| Bucket | Definition | Points to |
|---|---|---|
| F1 recall failure | gold ∉ union of first-stage lists | tokenisation, truncation, embedder, candidate depth |
| F2 ranking failure | gold in candidates, rank > 10 | fusion, LTR features |
| F3 confuser | top-ranked doc solves a sibling problem (same algorithm family) | hard-negative features, I/O-signature features |
| F4 truncation | gold cue lies in the truncated part | caps, head+tail, E-LONG |
| F5 parse/tokenisation | parse-fail flag, Python-2 syntax, literal-normalisation misses | parser fallback, tokenizer |
| F6 duplicates/ties | identical texts | tie policy |
| F7 hubs | documents appearing in many top-10 lists | E-HUB (under R4) |

Procedure: bucket automatically, review a stratified sample (≈50 per bucket), propose one change, re-run the ladder. Two diagnostics decide where to invest:

- **Recall@K curves** per retriever and for the union, plus a complementarity matrix (fraction of golds found by exactly one retriever) — decides whether a second embedder is worth its cost.
- **Oracle rerank upper bound:** NDCG@10 if the gold were moved to rank 1 whenever it is within top-K, for K ∈ {10, 20, 50, 100, 200}. If the gap between K = 10 and K = 200 is small, reranking cannot help much and effort belongs in candidate generation.

### 6.8 Latency/accuracy trade-off

Report the Pareto frontier of `(cold full-test time, p50/p95 single-query latency, peak RAM, model MB)` against NDCG@10 for each profile. The submitted profile is the highest-accuracy point that satisfies the SLO on the reference CPU. GPU, if present, may accelerate embedding but the official run is forced to CPU (`--device cpu`) to honour "minimal GPU".

### 6.9 Experiment ledger (append-only JSONL)

`run_id, timestamp, git_sha, config_hash, dataset (dev-cv|dev-hold|dev-short|test|repo-qa), split_seed, model_revisions, ndcg10, mrr10, recall@k, per_query_metrics_path, timings, hardware, decision (accept|reject|pending), notes`. TEST rows carry a `milestone` field; a check in CI fails if a TEST run is not preceded by a declaration.

### 6.10 Evidence required per component

| Component | Accepted when |
|---|---|
| BM25 | P2 parity passes; tuned parameters beat defaults on G1 |
| Dense embedder | Wins the bake-off Pareto under the SLO; licence, remote-code and offline checks pass |
| Views/truncation | G1–G3 vs. B3 |
| Fusion | G1 vs. best single retriever |
| LightGBM ranker | G1–G3 vs. best fusion; skew test passes; OOF vs. HOLD gap small |
| Cross-encoder | G1 **and** G4 on the low-margin subset |
| Enrichment / fine-tuning | R6 audit clean; large G1 gain; reproducible cold |
| Agent | Beats the no-agent baseline on REPO-QA multi-hop within latency budget (Section 11) |
| Incremental indexing | *Incremental == full-rebuild* invariant holds on the versioned benchmark (Section 9.6) |

---

## 7. Model and technology selection

### 7.1 Criteria

Retrieval quality on DEV · CPU cost (tokens/s on the reference machine) · RAM · model size · offline availability · licence · reproducibility (revision pin, no remote code) · integration complexity · reliability. **No model is chosen by name.** Facts marked `[MEM]` must be re-read from the model card in Phase 0.

### 7.2 Embedder shortlist (bake-off input, not a decision)

| Candidate | Size `[MEM]` | Licence `[MEM]` | Remote code `[MEM]` | Role / notes |
|---|---|---|---|---|
| Code-specific ≈137M (CodeRankEmbed, nomic) | ≈137M | permissive? | **yes** (custom architecture) | Strong small code embedder candidate; documented query prefix; supply-chain risk → vendor and pin the code, `trust_remote_code=False` policy |
| gte-modernbert-base (Alibaba-NLP) | ≈149M | Apache-2.0? | no (native ModernBERT) | General + code claims; 8k context |
| Qwen3-Embedding-0.6B | ≈0.6B | Apache-2.0? | no (native) | Instruction-aware; likely strongest of the shortlist but ≈4–5× the cost → `max` profile |
| jina-embeddings-v2-base-code | ≈160M | Apache-2.0? | yes | Older, code-specific |
| IBM granite-embedding r2 (small/base) | ≈47M / ≈149M | Apache-2.0? | no | Small; code coverage claimed |
| codet5p-110m-embedding | ≈110M | permissive? | yes | Older CoIR-era baseline |
| bge-m3 | ≈568M | MIT? | no | Old candidate; heavy for long inputs on CPU → baseline only |
| bge-small / MiniLM class | 22–33M | permissive | no | Floor baselines; not code-aware |
| jina-code-embeddings, SFR-Embedding-Code | 0.4–1.5B | **non-commercial** | — | **Excluded** unless the organizers explicitly allow |
| Voyage Code | hosted | **paid API** | — | **Excluded** (violates the no-API constraint) |
| Static embeddings (Model2Vec-style) | ≈16M | permissive | no | CPU-trivial diversity signal *if* a code-tuned checkpoint exists — verify |

**Selection procedure (Phase 3):** for every candidate measure DEV-CV NDCG@10, MRR@10, Recall@200, tokens/s and peak RAM on the reference CPU, projected cold-run time, licence, remote-code audit and offline availability; choose from the measured Pareto frontier under the SLO. Use each model's *documented* query/document prompt format. Sort texts globally by length before batching. Optionally test ONNX Runtime int8 (gate: ≤ 0.3-pt loss).

### 7.3 Other models

| Role | Candidates `[MEM]` | Status |
|---|---|---|
| Neural reranker | Qwen3-Reranker-0.6B; bge-reranker-v2-m3; small xsmall-class rerankers | `[EXP]` low-margin queries only |
| Agent/decomposition LLM (repo mode, optional) | Qwen3-4B-Instruct-2507; Qwen3-1.7B; Qwen2.5-Coder-1.5B-Instruct via `llama.cpp` GGUF Q4 | Off the P0 path; behind the agent gate |
| Ranker | LightGBM LambdaMART | Primary reranker |

### 7.4 Infrastructure choices

| Need | Choice `[DEC]` | Rejected | Reason |
|---|---|---|---|
| Lexical | Own CSR BM25 (numpy/scipy); `bm25s` used as the parity reference in tests | rank_bm25, Lucene/Pyserini (JVM), Tantivy, OpenSearch | Deterministic; assembles snapshots from cached per-blob term counts (P1); no server |
| Vector search | **Exact** matrix multiplication on float16-stored, L2-normalised vectors | Qdrant, pgvector, Milvus | Exact search over ≤≈1M vectors is millisecond-scale `[EST]`; ANN adds recall loss and incremental-update complexity; servers add ops surface. HNSW (FAISS/hnswlib) only above a *measured* size threshold |
| Metadata | SQLite (WAL) | Postgres | Embedded, transactional, zero-ops |
| Arrays / tables | `.npy` (memory-mapped) + Parquet + checksum manifest | pickle | Pickle loading executes code |
| Ranker format | LightGBM text model | pickled models | Same reason |
| API | FastAPI + uvicorn + pydantic v2 | Flask, gRPC | Schema-first validation |
| Python analysis | stdlib `ast` + own scope/import resolver | Tree-sitter, jedi | Zero dependency, never executes code; jedi may import modules; Tree-sitter only if multi-language is ever required |
| Git | hardened `git` CLI (Section 13) | libgit2 bindings | Speed, ubiquity |
| Embedding runtime | PyTorch CPU; ONNX Runtime int8 `[EXP]` | — | Measured |
| Telemetry | structlog, prometheus_client, optional OpenTelemetry | hosted APM | Local-only |
| Packaging | `uv` lockfile with hashes; Docker with digest-pinned base | floating dependencies | Reproducibility |
| Python | 3.11 (confirm against the pinned mteb's `Requires-Python`) | — | Widest wheel support |

---

## 8. Storage, index design and interface contracts

### 8.1 On-disk layout (`$ACI_HOME`)

```
config/                         profiles (lite|standard|max), pinned model table, thresholds (TOML)
blobs/sha256/ab/cdef…           content-addressed source blobs (immutable)
artifacts/                      per-blob derived artifacts; key = (blob sha, pipeline_version[, model_revision])
    tokens/<pipeline_v>/…       term-count vectors, identifier tokens, literals
    ast/<pipeline_v>/…          units, symbols, raw edges, parse status
    emb/<model_rev>/<prep_v>/   float16 matrices sharded + sqlite (blob_sha → row)
snapshots/<repo_id>/<snapshot_id>/     IMMUTABLE after the VALID marker
    manifest.json               ids, checksums, config hash, parent, validation report
    units.parquet               unit table
    lex/                        BM25 CSR (data, indices, indptr), vocab, df/idf, doc lengths
    vec/<model_key>.npy         matrix aligned to unit order (memory-mapped)
    sym/  graph/                symbol table; CSR adjacency with confidence tiers
    ltr/                        LightGBM model text + feature statistics
    VALID                       written last, after validation
refs/<repo_id>/CURRENT          text file with the active snapshot_id (atomically replaced)
meta.sqlite                     repos, versions, snapshots, refs history, lineage, evolution, jobs, ledger index
```

### 8.2 Identities and keys

| Key | Definition | Purpose |
|---|---|---|
| `blob_sha256` | SHA-256 of file bytes | Dedup across files and versions |
| `chunk_sha256` | SHA-256 of normalised unit text | Detect *which units* changed inside a changed file |
| `unit_key` | `(path, qualname, kind, ordinal)` | Stable identity across versions |
| `unit_id` | hash(`snapshot_id`, `unit_key`) | Per-snapshot identifier returned to clients |
| `snapshot_id` | SHA-256 of the canonical manifest (excluding timestamps) | Deterministic: same inputs → same id |
| Embedding key | `(chunk_sha256, model_revision, prep_version)` | Reuse across versions and runs |
| Result-cache key | hash of the **entire** normalised `SearchRequest` + `snapshot_id` + `config_hash` | Cannot return another version's results by construction |

### 8.3 Manifest (abridged)

```json
{
  "schema": 1,
  "snapshot_id": "sha256:…",
  "repo_id": "apps-benchmark",
  "version": {"label": "v3", "commit_sha": null, "parent_snapshot": "sha256:…", "source": "mteb|git|dir|zip|jsonl"},
  "builder": {"git_sha": "…", "pipeline_version": "p1", "config_hash": "…"},
  "units": {"count": 8765, "file": "units.parquet", "sha256": "…"},
  "indexes": {
    "lexical": {"kind": "bm25-csr", "params": {"k1": 0, "b": 0}, "files": {"…": "sha256"}},
    "vectors": [{"model": "…", "revision": "…", "dim": 0, "dtype": "float16", "file": "vec/A.npy", "sha256": "…"}]
  },
  "validation": {"passed": true, "canary_self_retrieval": {"n": 200, "top1": 1.0}, "checks": ["…"]}
}
```

(Numeric parameters are filled by the tuned config; nothing here is a claimed value.)

### 8.4 REST API (v1)

`POST /v1/search`

```json
{
  "repository_id": "apps-benchmark",
  "version": "latest",                       // label | tag | commit | snapshot_id | "latest" | "*" (all versions)
  "query": "…",
  "top_k": 10,                                // cap 1000
  "mode": "auto",                             // auto | hybrid | dense | lexical
  "agent": "auto",                            // auto | off | force   (ignored unless repo mode)
  "group_by": "none",                         // none | lineage  (all-versions mode)
  "filters": {"path_glob": null, "symbol_kind": null},
  "explain": true,
  "timeout_ms": 5000
}
```

Response:

```json
{
  "request_id": "…", "trace_id": "…",
  "query": {"class": "PROBLEM_STATEMENT", "hash": "…"},
  "snapshot": {"repository_id": "…", "snapshot_id": "sha256:…", "version": "…", "commit_sha": null},
  "results": [{
    "rank": 1, "score": 1.93, "confidence": "high",
    "unit_id": "…", "doc_id": "…", "path": "…", "symbol": "…", "kind": "function",
    "start_line": 42, "end_line": 58, "blob_sha256": "…", "source": "<exact bytes from the blob>",
    "signals": {"bm25": {"rank": 3, "score": 0}, "dense_a": {"rank": 1, "score": 0}, "ltr": 0},
    "flags": ["duplicate_group"],
    "lineage": {"lineage_id": "…", "versions": ["v1", "v2"], "node": "…"}
  }],
  "timing_ms": {"total": 0, "prepare": 0, "embed": 0, "bm25": 0, "dense": 0, "fuse": 0, "ltr": 0, "verify": 0},
  "degradations": [],
  "agent": {"invoked": false, "status": null, "steps": []}
}
```

Every result must resolve to real content: `source` is read from the blob store and its hash verified before it is returned. `agent.steps` contains high-level progress labels only, never model reasoning.

Other endpoints: `POST /v1/index` (async job) · `GET /v1/jobs/{id}` · `GET /v1/repositories` · `GET /v1/repositories/{id}/versions` · `POST /v1/repositories/{id}/activate` · `POST /v1/repositories/{id}/rollback` · `GET /v1/units/{unit_id}` · `GET /v1/lineage/{lineage_id}` · `GET /healthz` · `GET /readyz` · `GET /metrics` · `GET /version`.

Errors: `{"error": {"code", "message", "request_id"}}` with `400` validation · `404` unknown repository/version · `409 index_required` (version not indexed) or `snapshot_invalid` · `429` rate limit · `503 not_ready`.

### 8.5 CLI

`aci index --repo R --from {git|dir|zip|jsonl} --rev X` · `aci search` · `aci versions` · `aci activate|rollback` · `aci validate` · `aci gc` · `aci serve` · `aci eval p0` (the official run: produces JSON, predictions, CSV, manifest) · `aci eval p1|bonus`.

### 8.6 Engine interfaces the adapter depends on

`embed(texts, role, batch_size)`, `build_snapshot(ids, texts, batch_size)`, `search_batch(snapshot, queries, top_k, restrict)` — see Appendix A. The adapter never touches indexes directly.

---

## 9. P1 — version-aware retrieval and incremental updates

### 9.1 Guarantees

| # | Guarantee |
|---|---|
| G-P1-1 | **Isolation:** a request for version *X* can only ever read snapshot(*X*). |
| G-P1-2 | **Atomicity:** readers see a complete old snapshot or a complete new one, never a mixture. |
| G-P1-3 | **Immutability:** a snapshot never changes after its `VALID` marker. |
| G-P1-4 | **Equivalence:** an incrementally built snapshot equals a from-scratch build of the same source (identical rankings, scores within 1e-6). |
| G-P1-5 | **Recoverability:** a crash at any instant leaves the previous active snapshot intact. |
| G-P1-6 | **Cost proportional to change:** embedding cost scales with changed tokens; assembly is linear in total non-zeros. |
| G-P1-7 | **Reversibility:** rollback to any retained snapshot is one pointer swap. |

### 9.2 Version identity

- Git source → `version_id` = full 40-hex commit SHA; tags/branches/`HEAD~n` are resolved to SHAs at request time.
- Non-git source (folder, ZIP, JSONL, MTEB corpus) → `m-` + first 16 hex of SHA-256 over the sorted `(path, blob_sha)` list; a human label is attached separately.
- Metadata per version: repository, label, commit SHA, parent(s), author date, source type, snapshot id, status (`building|valid|failed|retired`).

### 9.3 Update pipeline

```
resolve source → file list (path, mode, blob_sha)                    [git ls-tree / safe walk / safe extract]
   → filter (language, size, binary, vendored/generated rules)
   → diff vs parent snapshot: added / modified / deleted / renamed    [git diff-tree -M, else path+hash]
   → for CHANGED blobs only: sandboxed parse → units, symbols, edges → tokens → embeddings (batched)
   → for UNCHANGED blobs: reuse per-blob artifacts (cache hit)
   → assemble snapshot in .tmp-<uuid>/ (BM25 CSR, vector matrices, symbol table, graph)
   → validate → fsync → rename into place → atomically replace refs/<repo>/CURRENT
```

**Changed-chunk detection:** inside a modified file, units whose `chunk_sha256` is unchanged reuse their embedding even though the file changed; a one-line edit re-embeds one unit.

**BM25 assembly.** IDF depends on corpus-wide document frequency, so any change nudges scores, but assembly is cheap: term-count vectors are cached per blob; `df` is a column sum; CSR is a vertical stack. Cost is O(total non-zeros) — seconds for very large snapshots in numpy `[EST]`. Segment-merge (LSM-style) indexes are **not** built unless a measurement shows assembly exceeding the SLO.

**Graph maintenance.** Recompute outgoing edges for changed files; re-resolve only *dependents* (files importing the changed module or referencing changed symbols), found via a reverse-dependency index.

### 9.4 Validation before activation

1. Manifest checksums match files.
2. Unit counts agree across lexical, vector, symbol and unit tables; unit ids unique.
3. Vectors finite with norms ≈ 1.
4. Every unit's blob exists and its hash matches (full check when small, ≥ 5 % sample when large).
5. **Canary self-retrieval:** a sample of units queried by their own text must return themselves at rank 1 (≥ 99 %).
6. BM25 document frequencies consistent with the term matrix.
7. Graph edges reference existing units.
8. The changed set recorded against the parent equals a fresh diff.

Any failure → snapshot marked `failed`, previous `CURRENT` untouched, alert metric raised.

### 9.5 Activation, rollback, recovery

- Build in `snapshots/<repo>/.tmp-<uuid>/`; `fsync`; `rename` into place (atomic within one filesystem); write `CURRENT.tmp`; `os.replace` onto `CURRENT`.
- Rollback = point `CURRENT` at a retained snapshot; history is kept in SQLite.
- On startup: delete stale `.tmp-*` directories; confirm `CURRENT` points to a `VALID` snapshot, else fall back to the newest valid one in the history.
- One writer per repository (file lock); readers are lock-free.
- Retention: keep the last N and any tagged snapshots; blobs are garbage-collected by mark-and-sweep under the writer lock.

### 9.6 Correctness invariants (tests)

1. **Incremental == full rebuild:** for random change sets, the two snapshots give identical rankings on a query set.
2. **Isolation:** N queries × M versions; results contain only units of the requested snapshot (`aci_cross_version_leak_total == 0`).
3. **Crash safety:** `kill -9` at random points across ≥ 1,000 iterations never leaves a partial active snapshot.
4. **Determinism:** same source + same config → same `snapshot_id`.

### 9.7 Caches and invalidation

Content-addressed caches never need invalidation (keys embed content hash, pipeline version and model revision); they are only garbage-collected. The query-result cache key is generated from the *entire* normalised request plus `snapshot_id` and `config_hash`, so a stale hit is impossible by construction; a property test enumerates every request field to ensure none is omitted from the key. Query embeddings are cached by `(model_revision, prep_version, query_hash)`.

### 9.8 Version isolation in depth

A single `SnapshotHandle` is resolved at request start and passed to every retriever; retrievers cannot query "current". The fusion stage asserts `candidate.snapshot_id == handle.snapshot_id`. The agent's tools are bound to the same handle. A dedicated fixture places the same symbol with different bodies in v1/v2 and checks that neither leaks.

### 9.9 Time-travel queries

`version` accepts a label, tag, branch, commit prefix, `snapshot_id`, `latest` or `HEAD~n` (git sources). If the version is not indexed the API returns `409 index_required`, and `aci index --rev` (or an async job) builds it.

### 9.10 Cost model (report measured values, do not assert)

| Step | Scales with |
|---|---|
| Diff | changed paths |
| Parse/featurise | changed bytes |
| Embed | **changed tokens** (dominant) |
| Assemble BM25/vectors | total non-zeros / total units |
| Validate | sample size + units |

Proposed targets `[EST]`, verified in Phase 10: a one-file commit on a 10⁵-unit repository becomes searchable within seconds to low tens of seconds on the reference CPU; a full rebuild is reported alongside so the incremental advantage is explicit.

### 9.11 Evaluating P1 without an official versioned dataset

Samsung supplies no versioned data, so we build two sets:

**APPS-Versioned (mechanics).** Seeded mutation operators applied to the APPS corpus create versions v1…v5: format-only edits (gold unchanged), identifier renames, behaviour edits (constants, output strings, branch changes) with templated queries that target the new behaviour, deletions (gold vanishes), additions (new candidate solutions), moves (document id/path changes). Ground truth per version is defined by operator semantics. This validates **isolation, time-travel correctness, incremental equivalence, timing** — *not* real-world semantic evolution quality.

**Real-repo history.** 2–3 small permissively licensed Python repositories with ≥ 20 commits (licence check in Phase 10); queries derived from docstrings and commit messages of specific versions; contamination caveats documented.

### 9.12 P1 acceptance

Zero cross-version leaks · incremental == full rebuild on 100 % of test queries · crash-safety fuzz passes · rollback works · measured update-time scaling table published · time-travel accuracy on APPS-Versioned within 0.5 pt of a full rebuild at every version.

---

## 10. Bonus — evolutionary retrieval across versions

### 10.1 The problem, made precise

Searching all versions creates three difficulties: (a) byte-identical snippets repeated across versions flood the top-10; (b) near-identical variants differ only in the details a query may depend on; (c) users want to see *how* a snippet evolved. The full solution infers a rich evolution graph; the practical one delivers the stated Bonus requirement with far less fragility.

### 10.2 Three tiers

| Tier | Method | Status |
|---|---|---|
| **Ideal** | Evolution graph with `SAME, RENAMED, MOVED, MODIFIED, REPLACED, SPLIT, MERGED` inferred from AST edit distance, embeddings, call context and history | `[RISK]` research-grade; false positives; **optional, off by default** |
| **Practical `[DEC]`** | Identity key + git rename detection + similarity alignment + union index over unique nodes + lineage grouping + delta retrieval view | **Build this** |
| **Minimum** | Exact identity + content dedup + per-node version list | Fallback if alignment misbehaves |

### 10.3 Data model

`node = (unit_key, chunk_sha256)`, embedded once, with a **presence set** (bitset over snapshots; a linear history compresses to intervals; branching history uses per-snapshot bits). `lineage_id` groups nodes believed to be the same evolving entity. Edges `node → node` carry `relation` and `confidence`. A byte-identical unit present in 50 versions is **one** node.

### 10.4 Lineage construction

- **Tier 1 (deterministic, high precision).** Same `unit_key` in consecutive snapshots → same lineage (`UNCHANGED` if `chunk_sha` equal, else `MODIFIED`). Git rename detection (`-M`, configurable similarity threshold) carries `unit_key`s across file renames/moves.
- **Tier 2 (alignment).** Within one commit, match unmatched deleted units to unmatched added units using rename-invariant evidence: normalised-AST equality; Jaccard of identifier-normalised token n-grams (local names replaced by positional placeholders); signature similarity (argument count/names, decorators); docstring similarity; call-set overlap. Combine with a small logistic model trained on synthetic rename/move mutations (weights are learned, not hand-set); threshold chosen for ≥ 0.95 precision on that benchmark; Hungarian assignment per commit. Outputs `RENAMED`, `MOVED` or both.
- **Tier 3 `[RISK]`.** One-to-many/many-to-one (`SPLIT`, `MERGED`) and semantic `REPLACED` using embeddings, call-graph neighbourhood and commit metadata. Emitted only as `POSSIBLE_*` with a confidence; excluded from acceptance criteria.
- Uncertain relations are labelled `UNKNOWN`; the system never asserts an evolution link it cannot support.

### 10.5 All-versions retrieval pipeline

```
query ──► temporal-intent detection (regex on "in v3", "before", "originally", "latest", "added", "removed", "history", …)
      ──► retrievers over the NODE index:  BM25 ∥ dense ∥ DELTA (BM25 over added/removed tokens + commit message)
      ──► RRF → ranker (extra features: delta-overlap, is_latest, variants-in-lineage, gap to lineage best)
      ──► group by lineage:  group score = best node score;  within group: node score desc, recency as tiebreak
      ──► response: grouped view (default) and flat node view; each node carries its version intervals and the diff to its predecessor
```

Modes: `as_of=X` (restrict presence to snapshot X), `latest` (newest node per lineage), `all` (grouped), `history(lineage)` (chronological variants). Temporal intent chooses the mode deterministically; explicit request parameters always win.

### 10.6 Why this addresses "similar snippets are hard to rank"

Identical duplicates collapse into one node; near-duplicates remain distinct nodes and are separated by the **delta view** (what changed) and the lineage-aware features; the presentation groups variants so a single lineage cannot monopolise the top-10 while every variant stays reachable.

### 10.7 Evaluation

Graded relevance: gold node = 2, other nodes of the gold lineage = 1, others 0. Metrics: graded NDCG@10, lineage-level Recall@10, **first-variant accuracy** (gold node ranked first within its lineage), crowding rate (share of the top-10 occupied by one lineage in the flat view). Sets: APPS-Versioned-Bonus (queries built from tokens that distinguish one variant from its siblings) and REPO-QA commit-message queries.

### 10.8 Acceptance and limits

Acceptance: grouped view lifts graded NDCG@10 and reduces crowding versus flat concatenation of per-version results; Tier-2 alignment reaches its precision target on synthetic mutations; Tier-3 stays optional. Known limits: squash merges, force-pushes, vendored copies, generated code and heavy refactors defeat alignment; then lineages stay separate (safe), never falsely merged.

---

## 11. Agentic retrieval — bounded investigation

### 11.1 Definition

`[DEC]` "Agentic" here means a **bounded, read-only controller that issues retrieval calls, inspects evidence, refines the search and stops.** Its output is ranked evidence, never generated code or prose answers. It is not an autonomous coding agent, a chatbot or a swarm.

### 11.2 When it runs — and when it must not

| Runs | Must not run |
|---|---|
| Repo mode **and** query class ∈ {`CALLER, CALLEE, REFERENCE, DATA_FLOW, MULTI_HOP, EVOLUTION`}, or calibrated confidence below τ_low on a complex query | **Any MTEB / P0 run** (non-determinism, latency, unproven benefit) |
| Request allows it (`agent` ≠ `off`) | High-confidence exact/symbol/simple semantic queries |
| Snapshot has the required indexes (graph for relationship queries) | Latency budget already exhausted; snapshot without a graph for a relationship query |

### 11.3 Controller tiers

- **Tier A — deterministic controller (default).** A state machine composes tools: `CALLER` → resolve symbol → `find_callers` → verify; `DATA_FLOW` → seed search → breadth-first along call edges to depth *d* → re-score visited units against the query. No LLM.
- **Tier B — optional small local LLM** (quantised, ≤ 4B) used only to decompose a multi-hop question or rewrite a query. Output is a JSON plan validated against a schema: whitelisted tool names, validated arguments, hard token limits.

### 11.4 Tools (all read-only; all bound to one `SnapshotHandle`)

`search_code(query, filters)` · `search_exact(term)` · `search_symbol(name, kind)` · `read_unit(unit_id)` · `read_context(path, lines)` · `find_references(symbol)` · `find_callers(symbol)` · `find_callees(symbol)` · `find_paths(src, dst, max_depth)` · `compare_versions(unit_key, a, b)` (the only tool taking a version pair, validated). No shell, file-write or network tools exist.

### 11.5 State and budgets (defaults, tuned on REPO-QA)

State: original query, candidates, visited units/files, queries already tried, discovered relationships (with confidence tier), snapshot id, remaining budget, evidence set. No chat history.

| Budget | Default |
|---|---|
| max iterations | 3 |
| max tool calls | 8 |
| max units read | 12 |
| max graph depth | 2 |
| LLM tokens (tier B) | ≤ 384 total, each call ≤ 128 |
| wall clock | 1.5 s (tier A) · 10 s (tier B) |

**Stop** when evidence is sufficient (confidence ≥ τ_high), when the last iteration produced no new candidates or no confidence gain, or when any budget is exhausted. **On timeout or failure:** return the pre-agent ranking with `agent.status = "budget_exhausted"`.

### 11.6 Safety and correctness

- Final ranking is always re-scored by the retrieval stack; the LLM never emits ranks, line numbers or code.
- Evidence spans come only from `read_unit`/`read_context` and are hash-verified.
- Tool outputs are wrapped in per-request random delimiters, length-limited and stripped of control characters; the model is told they are untrusted data (Section 13.4).
- The UI shows progress labels only ("Searching… Following call relationships… Verifying evidence…"), never hidden reasoning.

### 11.7 Benefit gate

REPO-QA multi-hop set with ground truth from the static graph ("which function calls X", "how does data reach Y"). The agent becomes default-on for the relevant classes **only if** it beats the no-agent baseline on graded NDCG@10 with p95 latency inside the budget. Otherwise it ships as an opt-in demo feature, clearly labelled experimental.

---

## 12. Fallback, degradation and edge cases

### 12.1 Degradation ladder

Each stage reports a health state; the engine drops to the next level automatically, records it in `degradations[]` and the metrics, and `/readyz` reflects it.

| Level | Active components | Trigger |
|---|---|---|
| L0 | BM25 ∥ Dense A/B → RRF → LTR → (gated CE) → (gated agent) | healthy |
| L1 | same, no CE / no agent | timeout, budget, model unavailable |
| L2 | BM25 ∥ Dense → RRF (no LTR) | LTR model missing/corrupt, feature error |
| L3 | BM25 ∥ Dense A only | Dense B failure |
| L4 | BM25 only (+ exact/symbol in repo mode) | embedder failure / OOM / unavailable |
| L5 | Dense only | lexical index invalid |
| L6 | Bounded linear scan over blobs (grep-like, capped) | both indexes unusable — results flagged `degraded` |

Per-component **circuit breakers**: N failures within a window open the breaker; a half-open probe re-tests it. Every stage has a timeout from the latency budget. Parsing/embedding workers are supervised and restarted.

### 12.2 Edge cases: detection → fallback → recovery

| Case | Detection | Fallback | Recovery |
|---|---|---|---|
| Extremely large repository | File/unit counts, bytes, projected build time vs. limits | Streaming ingestion; bounded worker memory; language/size filters; lexical-only for filtered files | Resume from per-blob cache; partial jobs are never activated |
| Extremely long file | Token count > cap | Head+tail dense view; chunk + max-pool `[EXP]`; BM25 on full text | Re-embed only the affected blob after cap change |
| Tiny snippet (e.g. 5 characters) | Length below threshold | Lexical + exact match weight; flagged low-information | None needed |
| Exact duplicates | Same `chunk_sha256` | Duplicate group; deterministic tie order | Group shown once in UI; both ids in benchmark mode |
| Near-duplicates | MinHash/SimHash similarity above threshold | Keep distinct; group in UI; delta view in Bonus | Tune threshold on the mutation benchmark |
| Irrelevant but lexically similar | Confuser bucket F3 | LTR literal/I-O features; dense agreement | Feature additions via ladder |
| Same meaning, different vocabulary | BM25 recall failure (F1) with dense hit | Dense retriever carries it; optional enrichment `[EXP]` | Model swap through bake-off |
| Missing symbols / anonymous code | No symbol extracted | Snippet/module-body units; lexical + dense only | — |
| Parser failure / syntax error / Python-2 syntax | Parse status ≠ ok | Tolerant tokenizer → regex tokens; AST features disabled, flag set | Reparse if pipeline version changes |
| Generated / minified code | Heuristics: very long lines, header markers, token entropy | Down-weight; excluded by default rule set; flag | User-configurable include list |
| Dynamic language behaviour | Unresolvable call targets | Edge tier `INFERRED/POSSIBLE/UNRESOLVED`; never asserted as certain | — |
| Multiple versions | Version param / temporal intent | Isolation per 9.8; ambiguity → `409` or `latest` with notice | — |
| Renamed function / moved file | Git rename or alignment score | Lineage link (Tier 1/2); otherwise separate lineages | Improve threshold with benchmark |
| Deleted file | Diff | Tombstone in new snapshot; still visible in older versions | — |
| Changed implementation | `chunk_sha` differs | New node; delta view | — |
| Stale index | `snapshot age` vs. latest known revision; failed canary | Serve last valid snapshot with a freshness warning | Re-index job |
| Embedding failure | Exception / non-finite vectors / timeout | Skip unit-level: mark `no_vector`; query-level: L4 | Retry with backoff; quarantine blob |
| Reranker failure | Exception / timeout / feature error | RRF ordering (L2) | Model reload; alert |
| Model unavailable / checksum mismatch | Load-time hash verification | Lower level of the ladder; refuse to start a *write* job | Restore from release artifact |
| Corrupted index | Manifest checksum or canary failure at load | Refuse snapshot; roll back to previous valid | `aci validate`; rebuild from cache |
| Insufficient memory | RSS guard; allocator error | Smaller batch size; memory-mapped vectors; drop to L4; reject huge jobs | Lower profile (`lite`) |
| CPU-only / no GPU | Device probe | Default path *is* CPU | — |
| Network unavailable | Any outbound attempt blocked | Runtime never needs network (`HF_HUB_OFFLINE=1`, models local) | — |
| Malformed query (huge, binary, control chars) | Validation | Reject with `400` or truncate with notice; length cap | — |
| Empty query | Validation | `400`; never return an arbitrary ranking | — |
| Adversarial repository contents | Injection patterns, bidi/zero-width characters, extreme repetition | Treat as data; normalise; flag; cap token repetition; no instruction ever reaches a privileged context | Quarantine source |

---

## 13. Security architecture and secure SDLC

### 13.1 Assets, actors, trust boundaries

Assets: host integrity; index/snapshot integrity; ranking integrity; model artifacts; availability. (No secrets are required — there are no external APIs.)
Actors: malicious repository author, malicious archive uploader, hostile query submitter, compromised dependency or model host, careless operator.
Boundaries: **ingest** (untrusted bytes → parsers) · **API** (untrusted requests) · **model artifacts** (downloaded weights and code) · **agent tools** (untrusted text → LLM).

### 13.2 Threat table

| Threat | Mitigation | Test |
|---|---|---|
| Zip-slip, absolute paths, `..`, symlink/hardlink escape | Extract to a fresh directory; reject links; resolve every path and require it to stay under the root | Path-traversal fixtures |
| Zip/decompression bombs, file-count bombs, nested archives | Ratio, total-size, count and depth limits; streaming extraction | Bomb fixtures |
| Executing repository code | **Never** import, `exec`, `eval` or install repository code; parsing only | Static test: no such calls in ingest paths |
| Parser exhaustion (deep nesting, huge literals) | Parsing in a subprocess with `RLIMIT_AS`, CPU limit and wall-clock timeout; failure → lexical-only | Deeply nested expression fixture |
| Malicious git config/hooks/filters/submodules/LFS | `git` with hooks disabled, no submodule recursion, `GIT_LFS_SKIP_SMUDGE=1`, `GIT_TERMINAL_PROMPT=0`, protocol allow-list, size/time limits, unprivileged user | Hostile-repo fixture |
| Prompt injection via code, comments, filenames, commit messages | Section 13.4 | Injection canaries |
| Index poisoning / keyword stuffing / Trojan-Source unicode | Strip bidi and zero-width characters in tokens; cap repeated-token contribution; comment-ratio and repetition features; provenance and quarantine | Poisoning fixtures |
| ReDoS in tokenisers | Linear-time scanning; no nested-quantifier regexes | Pathological-string tests |
| Resource exhaustion by requests | Query length cap, `top_k` cap (≤ 1000), rate limit, concurrency semaphore, per-stage timeouts, memory ceiling | Load tests |
| Model/dependency supply chain | Lockfile with hashes; digest-pinned base image; models pinned by revision **and** file SHA-256; `safetensors` only; **`trust_remote_code=False`** policy (if a model needs custom code, vendor an audited copy at a pinned revision); SBOM; `pip-audit`/OSV in CI | CI gates |
| Unsafe deserialisation | No pickle anywhere: `.npy`, Parquet, JSON, LightGBM text | Grep-based CI check |
| Unauthenticated remote use | Default bind `127.0.0.1`; optional constant-time bearer token when exposed; strict CORS | API tests |
| Log injection / data leakage | Structured JSON escaping; raw queries not logged unless `debug` | Log tests |

### 13.3 Untrusted-code handling summary

Repository content is **data**. The parsing workers run as an unprivileged user with resource limits and no network. Results are derived from `ast` structure and tokens, never from running anything. Symlinks are not followed; links are recorded as metadata only.

### 13.4 Prompt-injection defence (applies only where an LLM is used)

1. The deterministic ranking path (BM25, dense, LTR) has no instruction channel: imperative text in code cannot alter behaviour beyond its token statistics.
2. LLM inputs place tool outputs inside per-request random delimiters, length-limited and control-character-stripped, with a system message that they are untrusted quotations.
3. The LLM has **no side-effect tools**; tool names and arguments are whitelisted and schema-validated; outputs are parsed as JSON and rejected on violation.
4. The LLM cannot emit ranks, line numbers or code; evidence is always re-read from the blob store and hash-verified.
5. Canary fixtures ("ignore previous instructions…" in comments, docstrings, filenames, commit messages) must not change tool calls or rankings; detections are counted in metrics.

### 13.5 API and container hardening

Pydantic validation on every input · request-size limits · non-root container · read-only root filesystem · `--cap-drop=ALL` · `no-new-privileges` · default seccomp · tmpfs `/tmp` · `--network none` for the serving container · CPU and memory limits · health probes.

### 13.6 Secure SDLC

Pre-commit and CI: `ruff`, `mypy` (strict on core), `bandit`, `semgrep`, `gitleaks`, `pip-audit`/`osv-scanner`, image scan, SBOM (CycloneDX), lockfile drift check. Branch protection; signed release tags; release artifacts accompanied by SHA-256 sums (optional Sigstore attestation). Model table with revision + checksum enforced at load time. Dependency updates go through PRs with the full test suite.

### 13.7 Security tests

Path-traversal, zip-slip, symlink, bomb, deep-nesting, giant-line, binary-as-text, unicode-trick, prompt-injection and poisoning fixtures; fuzzing (Hypothesis/Atheris) of tokenisers, ingest and API schemas; dependency audit; permission checks on the container; a test asserting that no code path opens a network socket at query time.

---

## 14. Observability

### 14.1 Metrics (`/metrics`, Prometheus text format)

| Group | Metrics |
|---|---|
| Latency | `aci_request_seconds{stage}` histograms for prepare / embed / bm25 / dense / fuse / ltr / verify / agent; p50/p95/p99 derived |
| Retrieval | `aci_candidates{retriever}`, `aci_results_returned`, `aci_confidence_bucket` |
| Caches | `aci_cache_hits_total{cache}`, `aci_cache_misses_total{cache}` |
| Failures | `aci_component_failures_total{component}`, `aci_fallback_total{level}`, `aci_parse_failures_total` |
| Agent | `aci_agent_invocations_total`, `aci_agent_iterations`, `aci_agent_tool_calls`, `aci_agent_budget_exhausted_total` |
| Versions | `aci_snapshot_build_seconds`, `aci_snapshot_activations_total`, `aci_index_validation_failures_total`, `aci_index_age_seconds`, **`aci_cross_version_leak_total` (must stay 0)** |
| Resources | RSS, CPU utilisation, open file descriptors |

### 14.2 Logs, traces, health

- **Structured JSON logs** (`structlog`): timestamp, level, `trace_id`, `request_id`, `snapshot_id`, `config_hash`, stage, latency, status. No raw query text unless debug.
- **Tracing:** OpenTelemetry spans per stage; exporter off by default (console/file), OTLP optional.
- **Health:** `/healthz` (process alive) · `/readyz` (models loaded, active snapshot `VALID`, canary passes) · `/version` (git SHA, model revisions, mteb version, config hash).
- **Dashboard:** a built-in `/debug/stats` page (P50/P95/P99, cache hit rate, degradation state, index age); Grafana is optional.
- **Response transparency:** every search response carries a per-stage `timing_ms` breakdown — this is also the demo's "speed" evidence.
- **Alerts** (documented thresholds, no external service): breaker open, validation failure, leak counter > 0, p95 above budget, index older than freshness target.

---

## 15. Deployment and reproducibility

### 15.1 Repository layout

```
aci/                 engine library (ingest, parse, features, index, retrieve, fuse, rank, snapshots, versions, agent)
adapters/mteb/       mteb_adapter.py (Appendix A) + run_official_eval.py
api/  cli/  ui/      FastAPI app, CLI, static bundle
eval/                dev-split builders, ledger, bootstrap, parity tests, ablation runner
config/              profiles, model table (revision + sha256), thresholds
tests/  benchmarks/  unit, property, metamorphic, security, performance
docker/  Makefile  uv.lock  README.md  SECURITY.md  MODEL_CARDS.md
```

### 15.2 Configuration and model pinning

TOML profiles validated by pydantic; environment overrides are explicit and logged. The resolved configuration is hashed (`config_hash`) and written into every manifest, ledger row and result. A model table lists, per model: repository, **commit SHA**, file SHA-256s, licence, whether remote code is needed, tokenizer settings and prompt formats; loading fails on mismatch. Runtime: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

### 15.3 Determinism and its limits

Fixed seeds; `PYTHONHASHSEED=0`; pinned thread counts (`OMP_NUM_THREADS`, torch intra-op); stable sorts; deterministic tokenisation; ties resolved by our key. Floating-point reductions can differ across CPUs and thread counts, so scores may vary at the 1e-6 level and near-ties may swap. Policy: **rankings are compared with a tolerance test** (identical top-10 except swaps within 1e-6), and the manifest records the hardware and thread settings. Determinism claims are limited to "same machine, same config".

### 15.4 Packaging

- `uv` lockfile with hashes; Python 3.11; mteb pinned to one tested 2.x version; CI matrix over 2.0.5 / 2.12.30 / 2.21.0 for the adapter contract.
- Docker: multi-stage, digest-pinned base, non-root, models mounted or baked from release artifacts; CPU image only.
- Non-Docker path: `uv sync --frozen`.

### 15.5 One-command flows

| Command | Result |
|---|---|
| `make setup` | environment + verified model/dataset artifacts (checksummed) |
| `make reproduce-p0` | cold official run → `AppsRetrieval_results.json`, predictions, CSV, `run_manifest.json`, timing table |
| `make reproduce-p0-warm` | same, using optional caches; must satisfy parity P6 |
| `make demo` | API + UI + preloaded demo repositories and versions |
| `make test` / `make test-security` / `make bench` | test suites and benchmarks |

### 15.6 Release artifacts (GitHub release)

Model files or a pinned fetch script with SHA-256s · LTR model · configs · optional embedding caches with a documented generation command · result JSON, predictions, CSV, manifest · SBOM · checksum file · demo video and PPT.

### 15.7 Hardware handling

At start-up the manifest records CPU model, core count, RAM, GPU presence. The official run forces CPU. Profile selection can be automatic (`lite` on ≤ 8 cores/≤ 16 GB) but the run manifest always states the profile used.

---

## 16. Testing strategy

| Layer | Content | Pass criteria |
|---|---|---|
| **Unit** | tokenisers (identifier splitting, numeric normalisation), truncation, RRF maths, snapshot I/O, key generation, feature functions | 100 % of documented cases; coverage ≥ 90 % on core |
| **Property-based** (Hypothesis) | RRF monotonicity and permutation invariance; score-gap rule yields strictly decreasing scores that survive float32 (P7); cache-key covers every request field; manifest hashing determinism | no counter-examples |
| **Metamorphic** | query independence (P4); corpus order/id invariance (P5); cache parity (P6); incremental == full rebuild | exact equality within tolerance |
| **Contract** | MTEB adapter on a tiny task through real `mteb.evaluate` on each pinned version: dispatch takes the `SearchProtocol` path, `encode` not called, `batch_size` forwarded, JSON written, `NDCG@10 == MRR@10` for a gold-at-rank-1 fixture; the dense fallback via the unmodified wrapper equals the search path (P1) | green on the CI matrix |
| **Integration** | full P0 run on a 500-doc fixture; P1 scenarios on APPS-Versioned; Bonus scenarios; API end-to-end | green |
| **Regression** | golden rankings on fixtures; DEV-HOLD NDCG@10 within tolerance of the ledger | no unexplained drops |
| **Failure injection / chaos** | kill embedder, slow reranker, corrupt a vector file, truncate a manifest, `ulimit -v` memory pressure, `kill -9` during activation, disk-full during build | correct degradation level; previous snapshot intact |
| **Security** | Section 13.7 | all blocked/handled |
| **Performance** | latency budgets per stage; cold-run time; memory ceilings; scaling of update time vs. change size | within SLO or report failure |
| **Agent** | budget enforcement, schema rejection of malformed plans, tool whitelist, injection canaries, timeout fallback | no budget overruns; canaries inert |
| **UI/E2E** | smoke test of the demo path | green |

CI stages: lint/type/security → unit/property → contract matrix → integration → chaos (nightly) → performance (nightly) → release build.

---

## 17. Implementation plan (phase by phase)

**Ordering principle:** accuracy first, then P1, then the Bonus, then agent/UI polish. The P0 path is frozen (Phase 8) before repo-mode work begins. Phases 2–8 implement `build_snapshot` as an in-memory object that already conforms to the `Snapshot` interface; Phase 10 adds persistence and content-addressing behind the same interface, and the frozen P0 rankings are a regression test for that change. An import-boundary test guarantees that repo-mode/agent code cannot be imported by the P0 path.

Each phase lists: **Build · Why · In → Out · Depends · Tests · Metrics · Accept · Benchmark · Can fail / Fallback · Artifact.**

### Phase 0 — Foundations, verification, calibration
- **Build:** repo skeleton, lockfile, CI; dataset acquisition (`mteb.get_task("AppsRetrieval")`, plus APPS-train on the dev machine); dataset audit; MTEB contract-test suite with **real torch** on 2.0.5 / 2.12.30 / 2.21.0; hardware calibration; model-card audit; question list to the organizers (Appendix C).
- **Why:** every later decision rests on facts currently tagged `[STATS]`, `[MEM]`, `[EST]`.
- **In → Out:** dev machine with network → `dataset_audit.json` (counts, length percentiles, duplicates, parse rate, Python-2 fraction, contamination check of the 5,000 non-relevant documents against APPS-train), `config/model_table.toml`, measured tokens/s per candidate model and thread count, calibrated compute worksheet.
- **Depends:** none.
- **Tests:** contract tests; P1 dense-parity harness scaffold; P3 metric parity.
- **Metrics:** stats vs. `[STATS]` (tolerance 0.1 %); tokens/s; peak RAM.
- **Accept:** every `[MEM]` item resolved or struck; SLO agreed; APPS-train source and licence confirmed.
- **Benchmark:** embedding throughput for each shortlisted model at 1, 4, 8 threads; parse rate.
- **Can fail / Fallback:** dataset differs from `[STATS]` → re-derive statistics and re-plan splits; a model's licence blocks it → drop it; HF unreachable → mirror needed assets into the release.
- **Artifact:** `docs/PHASE0_REPORT.md`, `dataset_audit.json`, `model_table.toml`.

### Phase 1 — Evaluation infrastructure
- **Build:** dev-split builder (DEV-CV / HOLD / SHORT), experiment ledger, paired bootstrap, strata, per-query metric storage, contamination audit (R6), baselines B0 and B2, parity harness.
- **Why:** without trustworthy measurement nothing else is meaningful.
- **In → Out:** APPS-train + AppsRetrieval → deterministic splits, ledger, baseline numbers.
- **Depends:** Phase 0.
- **Tests:** split determinism; bootstrap on synthetic data with known effect; P3, P7.
- **Metrics:** B0 and B2 NDCG@10, MRR@10, Recall@K, timings.
- **Accept:** B0 reproducible run-to-run; ledger enforces TEST declarations; contamination report written.
- **Benchmark:** B0/B2 cold-run time.
- **Can fail / Fallback:** APPS-train unavailable → build dev from another public text-to-code set for harness validation only and downgrade dev conclusions.
- **Artifact:** `eval/` package, first ledger rows, contamination report.

### Phase 2 — Lexical engine
- **Build:** code-aware tokeniser (identifier splitting, numeric normalisation `10^9+7 ≡ 1e9+7 ≡ 1000000007`, string-literal tokens, keyword removal), own CSR BM25, query-term selection, parameter tuning.
- **Why:** literals and I/O strings are cheap, strong signals; BM25 is also the fallback level L4.
- **In → Out:** documents/queries → ranked lists; tuned parameters.
- **Depends:** Phase 1.
- **Tests:** unit tokeniser cases; **P2 parity** with mteb's `bm25s`; property tests.
- **Metrics:** B1, B5 on DEV-CV; latency.
- **Accept:** P2 passes; B5 beats B1 defaults under G1.
- **Benchmark:** index build time; query time on 8,765 docs and a 10⁶-unit synthetic corpus.
- **Can fail / Fallback:** tokeniser changes hurt → revert to matched tokenisation.
- **Artifact:** `aci/lexical`, tuned config.

### Phase 3 — Dense engine and embedder bake-off
- **Build:** embedder runtime (batching, global length-sort, prompt formats, truncation policies), embedding cache, ONNX/int8 study, the **adapter dense path**, bake-off harness over the shortlist.
- **Why:** dense retrieval is the dominant accuracy and cost driver.
- **In → Out:** shortlist + reference CPU → measured Pareto table; chosen embedder(s) per profile.
- **Depends:** Phases 0–1.
- **Tests:** **P1 parity** (SearchProtocol path == standard wrapper path); prompt-format tests; determinism.
- **Metrics:** B2–B4; NDCG@10, MRR@10, Recall@200, tokens/s, RAM, projected cold time.
- **Accept:** selection made from the measured frontier under the SLO; licence/remote-code/offline checks pass; int8 accepted only if ≤ 0.3-pt loss.
- **Benchmark:** per-model throughput and accuracy.
- **Can fail / Fallback:** best model exceeds SLO → next Pareto point; `max` profile only with cache/accelerator.
- **Artifact:** `aci/dense`, `MODEL_CARDS.md`, bake-off report.

### Phase 4 — Views and preprocessing experiments
- **Build:** query segmenter (fallback to whole text), `Q_task/Q_io/Q_lit`, document views, `D_sig`, truncation policies, E-LONG chunk+max-pool.
- **Why:** long problem statements are the defining feature of the benchmark.
- **In → Out:** raw texts → views; ablations.
- **Depends:** Phases 2–3.
- **Tests:** segmenter on marker/no-marker fixtures; parser-fallback tests (Python-2, syntax errors).
- **Metrics:** ΔNDCG@10 per view/policy with CIs; F1–F7 bucket counts.
- **Accept:** G1–G3 per accepted change; hard-query round 1 documented.
- **Benchmark:** cost of longer query caps.
- **Can fail / Fallback:** segmentation hurts → keep `Q_full`.
- **Artifact:** `aci/views`, ablation report.

### Phase 5 — Candidate generation and fusion
- **Build:** parallel retrieval orchestration, RRF / weighted RRF / z-score fusion, candidate depth study, second embedder (E-SECOND).
- **Why:** recall of the first stage caps everything downstream.
- **Depends:** Phases 2–4.
- **Tests:** fusion properties; parallel-vs-sequential equality.
- **Metrics:** B6–B8; Recall@K curves; complementarity matrix; oracle-rerank bounds.
- **Accept:** G1 vs. best single retriever; depth chosen from curves; second embedder accepted only under the cost-normalised rule.
- **Can fail / Fallback:** no fusion gain → keep the better single signal plus BM25 as safety net.
- **Artifact:** `aci/fusion`, curves.

### Phase 6 — Learned ranker and confidence
- **Build:** feature extraction (shared train/serve), LightGBM LambdaMART with 5-fold OOF, skew test, isotonic-calibrated confidence, RRF fallback.
- **Why:** the main CPU-feasible P0 upside beyond fusion.
- **Depends:** Phase 5 and the contamination audit.
- **Tests:** skew test; training determinism; failure injection (missing model → L2).
- **Metrics:** B9; OOF vs. HOLD gap; ECE.
- **Accept:** G1–G3; ranker does not use forbidden features (R4); HOLD gap small.
- **Can fail / Fallback:** distribution shift → RRF (L2).
- **Artifact:** `aci/ltr`, model text file, feature spec.

### Phase 7 — Gated experiments
- **Build/decide:** E-CE (low-margin gate), E-ENR, E-FT (only if R6 audit allows), E-HUB (only with task-independent queries, R4).
- **Why:** the riskiest levers get an explicit accept/reject with numbers.
- **Depends:** Phases 5–6.
- **Tests:** cold reproducibility of any accepted lever; contamination re-check.
- **Metrics:** B10, B11; gain per added second; per-stratum effects.
- **Accept:** G1–G5 **and** cold reproducibility; otherwise rejected and ledgered.
- **Can fail / Fallback:** any rejected lever simply stays out.
- **Artifact:** decision record per lever.

### Phase 8 — P0 freeze
- **Build:** final adapter, `run_official_eval.py` (Appendix A), artifact writer (JSON, predictions, CSV, manifest), cold/warm parity, profile selection, timing table; TEST runs at declared milestones; README run instructions.
- **Why:** P0 is the competitive core; freeze it before repo-mode work.
- **In → Out:** frozen config → submission artifacts.
- **Depends:** Phases 1–7.
- **Tests:** full contract matrix; P1–P7; clean-machine reproduction.
- **Metrics:** final NDCG@10, MRR@10 (with CIs), cold time, RAM, latency.
- **Accept:** reproduces on a clean machine within tolerance; cold time within the SLO; artifacts complete; no TEST tuning.
- **Can fail / Fallback:** SLO missed → lower profile; harness mismatch → dense fallback path.
- **Artifact:** tagged release `p0-freeze` + submission JSON.

### Phase 9 — Repository mode core
- **Build:** hardened ingestion, `CodeUnit` model, sandboxed `ast` workers, symbols, call/import/reference graph with confidence tiers, exact and symbol retrievers, deterministic query classes with the routing rule, API skeleton.
- **Depends:** Phase 8 (interfaces only).
- **Tests:** ingestion security fixtures; parser-limit tests; resolver tests on dynamic-code fixtures.
- **Metrics:** REPO-QA symbol/exact accuracy; parse success; index time.
- **Accept:** security fixtures blocked; edge tiers never over-claim.
- **Can fail / Fallback:** graph resolution weak → retrieval works without it.
- **Artifact:** `aci/repo`, API v0.

### Phase 10 — P1 versioning
- **Build:** CAS blob store, per-blob artifact caches, snapshot builder/validator, atomic activation, rollback, GC, time-travel resolution, incremental graph update, **APPS-Versioned** generator and real-repo histories.
- **Depends:** Phases 8–9.
- **Tests:** G-P1-1…7; incremental == full rebuild; crash fuzz (≥ 1,000 kills); isolation matrix; determinism of `snapshot_id`; regression that P0 rankings are unchanged after persistence is introduced.
- **Metrics:** update-time scaling table, cache hit ratio, leak counter, time-travel accuracy.
- **Accept:** Section 9.12.
- **Can fail / Fallback:** assembly slower than the SLO → segment-merge indexes.
- **Artifact:** `aci/versions`, P1 report with measured timings.

### Phase 11 — Bonus
- **Build:** node/lineage model, Tier 1–2 alignment (learned combiner), union index, delta view, temporal-intent detection, grouped ranking; Tier 3 optional.
- **Depends:** Phase 10.
- **Tests:** alignment precision on synthetic renames/moves; grouped vs. flat metrics; crowding rate.
- **Metrics:** graded NDCG@10, lineage Recall@10, first-variant accuracy.
- **Accept:** Section 10.8.
- **Can fail / Fallback:** alignment unreliable → minimum tier (identity + dedup).
- **Artifact:** `aci/evolution`, Bonus report.

### Phase 12 — Bounded agent
- **Build:** tools, state, budgets, deterministic controller, optional quantised LLM planner, injection canaries, REPO-QA multi-hop set.
- **Depends:** Phase 9 (and 10 for `compare_versions`).
- **Tests:** budget enforcement; schema rejection; tool whitelist; canaries; timeout fallback.
- **Metrics:** ΔNDCG@10 vs. no-agent on multi-hop; p95 latency; escalation rate.
- **Accept:** Section 11.7 gate.
- **Can fail / Fallback:** no gain → opt-in experimental feature.
- **Artifact:** `aci/agent`, gate report.

### Phase 13 — Product hardening
- **Build:** static UI, observability, security hardening, container images, CI matrix, chaos and performance suites, SBOM and audit gates.
- **Tests:** Sections 13.7 and 16.
- **Accept:** zero high-severity findings; chaos suite green; `/readyz` reflects degradation truthfully.
- **Artifact:** container images, dashboards page, security report.

### Phase 14 — Reproducibility freeze and submission
- **Build:** clean-room rebuild from the release, final docs, PPT, demo video, artifact checksums, final red-team pass against Section 18.
- **Accept:** a reviewer follows the README on a fresh machine and reproduces P0 numbers and the P1/Bonus demos.
- **Artifact:** GitHub release, PPT, video, `SUBMISSION_CHECKLIST.md`.

### 17.1 Demo plan

1. **Act 1 — P0:** a dataset-like problem statement; show ranked results with exact source, retrieval signals and the per-stage timing breakdown; show the official run's cold-time table.
2. **Act 2 — P1:** same query on v1 → v2 → v3 with different results; add a commit; show the diff, the small number of re-embedded units, the measured update time, the validation report and atomic activation; show a rollback.
3. **Act 3 — Bonus:** `version = *`; grouped lineages with per-variant version intervals and diffs; explain why identical snippets are collapsed and near-identical ones remain distinguishable.
4. **Optional Act 4 — agent (only if it passed its gate):** a multi-hop question with progress labels and the budget indicator.

Never show timings or scores that were not measured in that very run.

---

## 18. Red-team review

*"If I were the Samsung PRISM evaluator, where could this system fail?"*
Severity (Sev) and probability (Prob): H/M/L. "Redesign?" says whether the architecture already changed because of the weakness.

| ID | Weakness | Sev | Prob | Impact | Detection | Mitigation | Fallback | Redesign? |
|---|---|---|---|---|---|---|---|---|
| W1 | Cold P0 run too slow for the judges' hardware or patience | H | M–H | Runtime is judged; a timeout means no P0 result | Phase-0 tokens/s; manifest timings; projection tool | Profiles; length-sorted batching; int8; smaller embedder; LTR instead of cross-encoder; SLO; ask organizers for limits | `lite` profile | **Done** (old reranker replaced) |
| W2 | mteb version differs at judging | M | M | Adapter breaks or behaves differently | CI matrix on 2.0.5/2.12.30/2.21.0 | Lockfile; `**kwargs`; `ModelMeta` helper for 2.0.x | Dense fallback | Done |
| W3 | Harness forces the encoder path | M | L | Hybrid gains lost | Contract test; P1 parity | Fallback is a well-preprocessed dense retriever; report its score as a floor | Dense-only | Done |
| W4 | Guideline sample crashes on JSON write (mteb ≥ 2.12) | M | M | Judges' own copy fails | Executed on 2.12.30 and 2.21.0 | Our script uses `write_official_json`; README explains | — | Done |
| W5 | Over-tuning on TEST through repeated runs | H | M | Inflated, non-reproducible score; integrity questions | Ledger audit | Sealed TEST; run budget; DEV-CV/HOLD | — | Done |
| W6 | Contamination: non-relevant test documents may be APPS-train solutions; fine-tuning/LTR could exploit or be tainted | H | M (unverified) | Integrity risk; misleading gains | Phase-0 audit (R6) | R4–R7; default zero-shot + feature-only LTR; separate reporting | Drop the lever | Done |
| W7 | LTR distribution shift (smaller dev corpus) | M | M | Gains vanish on TEST | OOF vs. HOLD vs. TEST gap | Per-query relative features; monotone constraints; skew test | RRF (L2) | Done |
| W8 | My hypotheses about APPS (markers, literal value, I/O features) are wrong | M | M | Wasted effort | Ladder results | Every idea gated; failing ideas removed | Baselines | Done |
| W9 | Hands-on queries are shorter/different from APPS statements | M | M | Poor live demo | DEV-SHORT guard | Query-length-agnostic core; no tuning on long-only cues | BM25 + dense baseline | Watch |
| W10 | Model licence, remote code or offline availability blocks a candidate | M | M | Late model swap | Phase-0 audit | Exclude NC/paid; vendor pinned code; `safetensors` only | Next Pareto model | Done |
| W11 | Dataset/models unreachable at judging time | H | L–M | Cannot run | Offline dry-run on a clean network-less machine | Release artifacts with checksums; documented offline setup | Pre-downloaded bundle | Watch |
| W12 | Unknown format for the P1 evaluation | M | M | P1 demo mismatched | Ask organizers early | Multiple source adapters (git, folder, ZIP, JSONL); time-travel API | Manual demo path | Watch |
| W13 | Unknown Bonus judging criteria | M | M | Bonus under-scored | Ask organizers | Both grouped and flat views; graded metrics; explicit diff view | Minimum tier | Watch |
| W14 | Agent flakiness in a live demo | M | M | Embarrassing failure | Agent tests; canaries | Deterministic default tier; budgets; recorded video | No-agent ranking | Done |
| W15 | Duplicate documents create unbreakable ties | L | H | ≈0.1 % of queries at most | Duplicate audit | Deterministic tie key | — | Accepted |
| W16 | Cross-machine float differences change scores at 1e-6 | L | H | Minor discrepancies vs. our JSON | Tolerance test | Documented; manifest records hardware | — | Accepted |
| W17 | Memory blow-up (289k-char document, large batches) | M | L | Crash | Peak-RSS monitoring | Truncation; length-sorted batches; memory guard | Smaller batches / L4 | Done |
| W18 | Query truncation drops the I/O specification | M | M | Systematic misses | F4 bucket; E-TRUNC | Cap 1,024; segment-aware views | Head+tail | Done |
| W19 | Over-engineering dilutes P0 | H | M | Weak core, flashy extras | Phase gates | P0 freeze before repo mode; import-boundary test | Drop features | Done |
| W20 | UI/demo failure on the day | M | M | Lost impression | Smoke tests | Recorded video; pre-warmed models; offline | CLI demo | Watch |
| W21 | `[MEM]` model facts are wrong | M | M | Wrong shortlist | Phase 0 | Re-read model cards; measure | Adjust shortlist | Watch |
| W22 | Malicious upload during hands-on | M | L | Host compromise | Security tests | Sandboxed parsing; limits; no execution; container hardening | Disable uploads | Done |
| W23 | Judges' CPU weaker than the reference | M | M | Slow runs | Projection tool | Auto profile; documented timings per profile | `lite` | Watch |
| W24 | Ambiguity of "MRR" (which cutoff) | L | M | Mismatch | JSON contains all cutoffs | Report `mrr_at_10` and others; confirm | — | Watch |

### 18.1 Final checklist

| Question | Answer | Evidence status |
|---|---|---|
| Does it satisfy **P0**? | Architecture: yes. The score is unknown until Phase 8; nothing here promises a number | Design + gates |
| Does it satisfy **P1**? | Yes by design: isolation, incremental updates, atomic activation, rollback; acceptance tests defined | Design; measured in Phase 10 |
| Does it satisfy the **Bonus**? | Yes at the practical tier (lineage + union index + delta view); full evolution graph is optional | Design; measured in Phase 11 |
| Is the **MTEB integration technically valid**? | Yes: `SearchProtocol` path on 2.0.5, 2.12.30, 2.21.0 | **`[EXEC]`**; the standard-wrapper path with real torch is `[SRC]` only until Phase 0 |
| Is it **CPU-first**? | Yes; cold-run time is the main risk and is managed by profiles | `[EST]` → measure |
| Does it avoid **paid APIs/keys**? | Yes; hosted models removed; no runtime network | Design |
| Is **latency** practical? | Budgets set (Section 4.4); query embedding dominates | `[EST]` → measure |
| Is it **reproducible**? | Pins, hashes, manifests; float caveat stated | Design |
| Is it **secure**? | Threat model, sandboxing, no execution, supply-chain controls, tests | Design; residual risks listed |
| Is it **robust to failure**? | Seven-level degradation ladder; failure-injection suite | Design |
| Can it handle **arbitrary Python code**? | Yes with fallbacks for unparseable/Python-2/generated code | Measured in Phase 0 |
| Can it handle **version changes**? | Yes (Section 9) | Phase 10 |
| Can it be **demonstrated end-to-end**? | Yes (Section 17.1) | Phase 14 |
| Does every major component have a **measurable reason to exist**? | Yes (Section 6.10); anything failing its gate is removed | Ledger |

### 18.2 Assumptions to verify before implementation

| # | Assumption | How |
|---|---|---|
| A1 | AppsRetrieval statistics match `[STATS]`; the 5,000 non-relevant documents may be APPS-train solutions | Phase-0 audit |
| A2 | APPS-train is available with a permissive licence and multiple solutions per problem | Phase 0 |
| A3 | Model facts (`[MEM]`): size, licence, remote code, prompt formats | Read model cards |
| A4 | Which mteb version the judges use | Ask; pin; CI matrix |
| A5 | CSV vs. JSON submission wording | Ask organizers |
| A6 | Time limit and reference hardware for the cold run | Ask organizers |
| A7 | Format of the P1/Bonus hands-on evaluation | Ask organizers |
| A8 | Standard `SearchEncoderWrapper` path behaves as read (needs real torch) | Phase-0 parity test P1 |
| A9 | CPU throughput and the compute worksheet | Phase-0 measurement |
| A10 | Fraction of Python-2/unparseable documents | Phase-0 audit |
| A11 | Which MRR cutoff the organizers use | Ask; report all |
| A12 | Network/HF availability at judging | Offline dry run |
| A13 | Python 3.11 compatibility with the pinned mteb | Phase 0 |
| A14 | Chosen models' licences allow this use | Phase 0 |

---

## Appendix A — MTEB adapter skeleton (tested on mteb 2.0.5, 2.12.30, 2.21.0)

`[EXEC]` With a toy engine whose raw scores contain ties, this file passed `mteb.evaluate(model, [AppsRetrieval-shaped task], encode_kwargs={"batch_size": 64})` on all three versions: `SearchProtocol` path taken, JSON written, `NDCG@10 = MRR@10`. Torch and sentence-transformers were stubbed in that sandbox, so `encode()` was checked by direct call, not through the standard wrapper (test P1 covers that in Phase 0). The skeleton's zero-argument constructor leaves `engine` unset; the production module must build the default engine lazily on first use, otherwise `PrePostPipelineEncoder()` cannot run the official sample.

```python
"""Reference skeleton: MTEB adapter for the ACI retrieval engine.

Design rules (all verified against mteb 2.0.5 / 2.12.30 / 2.21.0):
  * class name + base class + zero-arg constructor match the hackathon sample;
  * index()+search() BOTH defined  -> mteb uses this object directly as the search model;
  * NEVER define predict()         -> that would route to the cross-encoder wrapper;
  * encode()+similarity() stay valid -> dense-only fallback if a harness forces the encoder path;
  * mteb_model_meta is a real ModelMeta (never None);
  * search() returns strictly decreasing scores (no ties) for every query id.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


class Engine(Protocol):
    """What the adapter needs from the (MTEB-agnostic) engine."""
    revision: str
    def embed(self, texts: Sequence[str], *, role: str, batch_size: int) -> np.ndarray: ...
    def build_snapshot(self, ids: Sequence[str], texts: Sequence[str], *, batch_size: int) -> Any: ...
    def search_batch(self, snapshot: Any, queries: Sequence[str], *, top_k: int,
                     restrict: Mapping[str, Sequence[str]] | None = None) -> list[list[tuple[str, float]]]: ...


def make_model_meta(name: str, revision: str, **extra: Any) -> ModelMeta:
    """mteb>=2.12 has ModelMeta.create_empty(); 2.0.x does not (all 17 fields are required there)."""
    overwrites = {"name": name, "revision": revision, "similarity_fn_name": "cosine", **extra}
    if hasattr(ModelMeta, "create_empty"):
        return ModelMeta.create_empty(overwrites=overwrites)
    base = dict(loader=None, release_date=None, languages=["eng-Latn"], n_parameters=None, memory_usage_mb=None,
                max_tokens=None, embed_dim=None, license=None, open_weights=True, public_training_code=None,
                public_training_data=None, framework=[], use_instructions=False, training_datasets=None)
    base.update(overwrites)
    return ModelMeta(**base)


def strictly_decreasing(ranked: list[tuple[str, float]], gap: float = 1e-5) -> dict[str, float]:
    """Emit scores whose order is OUR order under every scorer mteb uses.

    Verified pitfall: pytrec_eval (NDCG) treats score gaps below ~1e-6 as ties (float32-like resolution) and breaks
    ties by reverse doc-id order, while mteb's own MRR compares Python floats -> NDCG and MRR can silently disagree.
    Fix: squash to (1, 2) (float32 spacing ~1.2e-7 there, no underflow) and enforce an absolute gap >= 1e-5.
    Drift over 1000 results is <= 0.01, so the scale stays positive and the order is exact.
    """
    out: dict[str, float] = {}
    prev = float("inf")
    for doc_id, raw in ranked:
        raw = float(raw) if np.isfinite(raw) else -1e9
        s = 1.0 + 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))      # order-preserving map to (1, 2)
        s = min(s, prev - gap) if prev != float("inf") else s
        out[doc_id] = float(s)
        prev = s
    return out


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, engine: Engine | None = None) -> None:          # zero-arg friendly
        self._engine = engine
        self._snapshot: Any = None
        rev = getattr(engine, "revision", "unbuilt")
        self._meta = make_model_meta("aci/hybrid-retriever", rev)

    @property
    def mteb_model_meta(self) -> ModelMeta:                             # type: ignore[override]
        return self._meta

    @property
    def engine(self) -> Engine:
        if self._engine is None:                                        # lazy: heavy imports/model loads live in the engine
            raise RuntimeError("engine not configured; wire aci.engine.build_default_engine() here")
        return self._engine

    # ---- (1) SearchProtocol: the primary path --------------------------------------------------
    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None,
              encode_kwargs: Mapping[str, Any] | None = None, num_proc=None, **kwargs) -> None:
        ids = [str(i) for i in corpus["id"]]
        titles = corpus["title"] if "title" in corpus.column_names else [""] * len(ids)
        texts = [(f"{t}\n{x}" if t else x) for t, x in zip(titles, corpus["text"])]
        bs = int((encode_kwargs or {}).get("batch_size", 64))
        self._snapshot = self.engine.build_snapshot(ids, texts, batch_size=bs)

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None, top_k: int = 1000,
               encode_kwargs: Mapping[str, Any] | None = None, top_ranked=None, num_proc=None, **kwargs):
        qids = [str(i) for i in queries["id"]]
        ranked = self.engine.search_batch(self._snapshot, list(queries["text"]), top_k=top_k, restrict=top_ranked)
        return {qid: strictly_decreasing(r) for qid, r in zip(qids, ranked)}

    # ---- (2) dense-only fallback: only used if a harness forces SearchEncoderWrapper -------------
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type: PromptType | None = None, **kwargs) -> np.ndarray:
        texts: list[str] = []
        for batch in inputs:
            texts.extend(batch["text"])
        role = "query" if prompt_type == PromptType.query else "document"
        return np.asarray(self.engine.embed(texts, role=role, batch_size=int(kwargs.get("batch_size", 64))), dtype=np.float32)


# ---- artifact helpers ----------------------------------------------------------------------------
def _json_default(o: Any):
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_official_json(task_result, path: str | Path) -> Path:
    """The hackathon sample does json.dump(task_result.to_dict()); that raises on mteb 2.12+/2.21 (datetime)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=_json_default)
    return path
```

### A.2 Official run script (corrected version of the guideline sample)

```python
import json, mteb
from pathlib import Path
from aci.mteb_adapter import PrePostPipelineEncoder, write_official_json   # write_official_json fixes the datetime crash

model = PrePostPipelineEncoder()                     # zero-argument constructor, as in the sample
task = mteb.get_task("AppsRetrieval")
result = mteb.evaluate(
    model, [task],
    encode_kwargs={"batch_size": 64},
    cache=None,                                      # never reuse stale cached results for the submission
    prediction_folder="out/predictions",             # per-query rankings written by mteb
)
task_result = list(result.task_results)[0]
write_official_json(task_result, Path("out/AppsRetrieval_results.json"))
# then: export out/predictions -> out/AppsRetrieval_ranking.csv (query_id, rank, doc_id, score) and write run_manifest.json
```

## Appendix B — What the sandbox runs showed

| Check | Result |
|---|---|
| Dispatch for 7 class shapes | Identical on 2.0.5, 2.12.30, 2.21.0 (table in 3.3) |
| Official call shape, `SearchProtocol` path | `encode` never called; `batch_size` forwarded to `search`; 151 metric keys (2.0.5, 2.12.30), 152 (2.21.0); `main_score = ndcg_at_10` |
| `prediction_folder` | Writes `AppsRetrieval_predictions.json` (keys `mteb_model_meta`, `default`) |
| `json.dump(to_dict())` | Fails on 2.12.30 and 2.21.0 (`datetime`); works on 2.0.5 (`date` is `None`) |
| `ModelMeta.create_empty` | Missing in 2.0.5; `mteb_model_meta = None` crashes prediction saving (executed on 2.0.5) |
| `pytrec_eval` tie behaviour | Gaps below ≈1e-6 are ties, broken by reverse doc-id; naive epsilon gives NDCG@10 = 0.2 with MRR@10 = 1.0 on the same run |
| Adapter skeleton with score-gap fix | NDCG@10 = MRR@10 = 1.0 on all three versions |
| **Not run** | Real torch/sentence-transformers path; real dataset; real models; real CPU timings |

## Appendix C — Questions for the organizers (send before Phase 1)

1. Is the submission a **JSON**, a **CSV**, or both? Which MRR cutoff is used?
2. Is there a **time or hardware limit** for the evaluation run? What reference CPU/RAM?
3. Which **mteb version** will be used to run submissions?
4. May the **release** include precomputed embedding caches, provided cold runs reproduce them?
5. How will **P1** and the **Bonus** be evaluated — what versioned data or repository will be supplied, and in what format?
6. Are models with **non-commercial licences** acceptable?
7. Is fine-tuning on the APPS **train split** acceptable, and is the overlap between train solutions and the test corpus known?