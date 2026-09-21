# Phase 1 report — harness and contracts (no held-out touch)

Scope: `docs/spec/07` Phase 1. Every number below comes from the dev split or from a public third-party task; the
held-out touch counter is **0 of 6**. Nothing here is an accuracy claim about the final system — the dense channel
in this phase is a model-free stand-in encoder that a strict run refuses to serve.

## Acceptance criteria

| Criterion (docs/spec/07 Phase 1) | Result | Evidence |
|---|---|---|
| Metrics equal mteb/pytrec_eval to 1e-9 on random, oracle and BM25 runs (B0) | **PASS** — max deviation 0.0 on every key | `tests/metamorphic/test_parity.py::test_b0_metrics_equal_mteb_and_pytrec_eval_to_1e9` |
| BM25 parity ≥ 95 % top-10 identical (B1) | **PASS** — **100.0 %** top-10 *and* top-100 identical on all 5,000 dev queries; ΔNDCG@10 = 0.00e+00 | `[ledger:dev-c93bfff182ff]`, `tests/metamorphic/test_parity.py::test_p2_bm25_parity_with_the_mteb_baseline` |
| Split lock + sealed-data tests | **PASS** — lock written and re-verified; 30 seal tests green | `configs/splits.lock.json`, `tests/security/test_seal.py`, `tests/integration/test_dev_harness.py` |
| Tie and dispatch contract tests green on the matrix | **PASS** on the pinned mteb 2.21.0 (42 contract tests) | `tests/contract/` |
| Robustness library tests green **and** engine tests run | **PASS** — 37/37, of which 27 exercise the real engine through `acis.robust_hook` over a 256-document fixture | `tests/robustness/`, `tests/fixtures/robust_corpus.jsonl` |
| `AcisEngine` interface frozen (Track B may start) | **DONE** — `acis.engine.protocol.SearchEngine`; changing a signature needs an ADR | `src/acis/engine/protocol.py`, `src/acis/engine/README.md` |
| Adapter v0 validated on dev **and** on a public non-APPS mteb code task | **PASS** in both modes — see below | `runs/reg_validation_mode{A,B}.json`, `tests/integration/test_dev_harness.py` |
| `verify-submission` v0 | **PASS** — 16 tests, each breaking exactly one thing | `tests/integration/test_verify_submission.py` |
| Official script | Written and **never executed here**: it refuses to start unless `HF_HOME` points at the physical seal, and the owner runs it | `src/acis/eval/official.py`, `tests/security/test_seal.py` |
| Not yet: dense encoder runtime, LTR, held-out evaluation | Honoured — `mode="hybrid"` raises rather than inventing an ungated fusion; the dense runtime is Phase 2 | `tests/unit/test_prep_and_engine.py::test_hybrid_mode_is_not_invented_before_its_gate` |

## Ladder (dev split, all 5,000 queries)

| Rung | System | NDCG@10 | MRR@10 | Ledger |
|---|---|---|---|---|
| — | random ranking (seeded) | 0.0003 | 0.0001 | `[ledger:dev-cfe1b31e083f]` |
| — | oracle (gold at rank 1) | 1.0000 | 1.0000 | `[ledger:dev-0869a6607963]` |
| B1 | our BM25, matched tokenisation | 0.2507 | 0.2291 | `[ledger:dev-d135e078a08c]` |
| B0 | `mteb/baseline-bm25s` | 0.2507 | 0.2291 | `[ledger:dev-c93bfff182ff]` |

Read these as harness checks, not as accuracy: random ≈ 0 and oracle = 1 prove the metric path end to end, and B0 ≡
B1 proves our lexical channel is the baseline rather than a different system. Dev BM25 (0.2507) is far above the
third-party held-out figure for BM25 (4.76 [R-02]) because the dev queries and corpus differ from the held-out
split — which is exactly why **decisions use paired deltas, never absolute dev numbers** (docs/spec/03 §5).

## Adapter validation on a public non-APPS task

`StackOverflowQA` (mteb/StackOverflowQA, 1,994 queries / 19,931 documents), same object, both modes:

| Mode | Dispatch | `adapter_invocations` | `encode_calls` | NDCG@10 | MRR@10 | Wall clock |
|---|---|---|---|---|---|---|
| A | `PrePostPipelineEncoder` used directly | 1 | 0 | 0.52235 | 0.502225 | 515 s |
| B | `_EncoderSurface` → `SearchEncoderWrapper` | 0 | 2 | 0.52235 | 0.502225 | 71 s |

Identical scores to six decimals on an unrelated task, at 20k documents, is parity **P1** confirmed outside the
fixture. The time difference is not noise and is worth carrying into Phase 2: Mode A ranks each query over the whole
corpus in a Python loop, while Mode B hands mteb one matmul. Batching the dense channel is the first optimisation
the profiler should look at (docs/spec/06 §6 says to optimise what profiling shows — this is the first thing it
shows).

## Parity suite (P1–P7)

| id | Claim | Result |
|---|---|---|
| P1 | our `SearchProtocol` path ≡ mteb's `SearchEncoderWrapper` for the same encoder | identical on the fixture task and on StackOverflowQA |
| P2 | our BM25 ≡ `mteb/baseline-bm25s` | 100 % top-10 identical, 5,000 queries |
| P3 | metrics re-scored from `run.trec` ≡ direct metrics | equal to 1e-12 on random/oracle/BM25 |
| P4 | a query alone ≡ the same query in a batch | exact, plus a Hypothesis property |
| P5 | corpus order and document relabelling change nothing | exact (ties break on the content hash) |
| P6 | warm cache ≡ cold | exact; the cache is keyed on content, so repeated query texts hit on the cold pass too |
| P7 | adversarial ties and sub-float32 gaps keep NDCG = MRR for a gold-at-rank-1 fixture | exact, at magnitudes 1, 10 and 100 |

## Findings worth carrying forward

1. **NDCG and MRR are graded in different orders.** pytrec_eval compares scores at float32 resolution and breaks
   ties by document id descending; mteb's MRR compares exact floats. Our metrics implement *both* orders, which is
   the only way to reproduce the harness at 1e-9 — and it makes D9 (rank-derived scores) a requirement rather than
   a preference.
2. **Ties must not be broken by corpus position.** Breaking them by the document's content hash makes P5 exact;
   only documents with *identical* content fall through to the corpus ordinal, which is the single permitted use of
   order (D9, INV-4).
3. **`similarity_fn_name` must be the enum member, not the string.** mteb compares it with `is`, so a plausible
   `"cosine"` silently becomes "Similarity function not specified" — in Mode B only.
4. **REG benchmarks ship their own `qrels/test-*` files.** They are other people's labels, but they match the
   sealed-name patterns, so they now live in `.acis-reg/` (`acis.core.paths.reg_home`), outside `ACIS_HOME`. The
   seal check can stay deliberately broad without false alarms.
5. **`.claude/settings.json` deny globs are path-fragment matches.** `Edit(./data/**)` blocks `src/acis/data/**`
   too, hence ADR-0004. The owner can anchor the glob and the package can take its spec name back.
6. **The stand-in encoder was under-dimensioned at first** (256 buckets), and the robustness suite caught it as
   brittleness under sentence dropout. It was hash collisions, not ranking logic; 4,096 buckets fixed it. Worth
   remembering that `tests/robustness` fails for representation reasons as readily as for logic ones.

## What is deliberately absent

The dense encoder runtime, LTR, PRF, hybrid fusion, the code-aware tokeniser, routing v1.1's OOD bank, and any
held-out evaluation. Each belongs to a later phase with its own gate, and each would have made this phase's numbers
unverifiable.
