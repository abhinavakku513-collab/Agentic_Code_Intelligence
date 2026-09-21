# 01 · Requirements, verified facts, open items, traceability

Authority: subordinate to `CLAUDE.md`. This file elaborates and never amends. Evidence tags: **[V]** verified by tool call in the design session (source read or executed) · **[R]** reported by a reference document or third party, not re-run by us · **[I]** inferred · **[E]** estimate, measure before use · **[U]** unknown. No number for *our* system may appear anywhere without `[ledger:<run_id>]` (INV-14).

## 1. Official sources (priority order)
1. `docs/official/theme1_guidelines.md` — OCR of the 4-page PDF [V: all pages read; `has_visual_content=false`].
2. `docs/official/FAQ.md` — provided verbatim by the owner: APPS (Python) throughout, no extra dataset, ignore JS; no method restrictions; running time, GPU requirement, model size etc. are factored into evaluation.
3. Owner's brief and constraints (no paid API/keys, local models, offline-capable, CPU-first, minimal GPU, reproducible, genuine retrieval only).

## 2. Facts verified during design (re-verify with the commands below)
| ID | Fact | Evidence |
|---|---|---|
| V-01 | mteb latest on PyPI = **2.21.0** (2026-09-20); `Requires-Python <3.15,>=3.10`; deps include `torch>=2.0`, `transformers>=4.40,<6`, `pytrec-eval-terrier>=0.5.6`, `pydantic>=2.11`, `datasets>=2.19`, `scipy>=1.14` | `pip index versions mteb`; wheel METADATA |
| V-02 | Retrieval dispatch (`abstasks/retrieval.py`): `EncoderProtocol ∧ ¬SearchProtocol → SearchEncoderWrapper`; `elif CrossEncoderProtocol → SearchCrossEncoderWrapper`; `elif SearchProtocol → model used directly`. Protocols are `runtime_checkable`; a class defining `predict()` risks the cross-encoder branch | source lines ≈392–396, `models_protocols.py` |
| V-03 | `SearchProtocol.index(corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc)` and `.search(queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs, top_ranked=None, num_proc)`; `mteb_model_meta` property required | `models_protocols.py` |
| V-04 | `evaluate()` defaults: `cache=_DEFAULT_CACHE` (persistent `ResultCache`) and `overwrite_strategy="only-missing"` → a rerun with an unchanged model identity can return a stale result. `AbsEncoder.mteb_model_meta` defaults to `None` | `evaluate.py`, `abs_encoder.py` |
| V-05 | Metrics: NDCG/MAP/recall/P/success via `pytrec_eval.RelevanceEvaluator` (`ndcg_cut.<k>` strings); MRR computed by mteb, sorting by `(score, doc_id)` descending; `k_values=(1,3,5,10,20,100,1000)`, `top_k=1000` | `retrieval_metrics.py`, `retrieval.py` |
| V-06 | `AppsRetrieval`: path `CoIR-Retrieval/apps`, revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`, `eng-Latn`+`python-Code`, `main_score=ndcg_at_10`, MIT | `tasks/retrieval/code/apps_retrieval.py` |
| V-07 | `TaskResult.to_dict()` is `model_dump()` and the model has `date: datetime | None` → `json.dump` without a `default=` fails whenever `date` is set (the official sample also omits `import json`) | `results/task_result.py` (crash on a real run not executed) |
| V-08 | pytrec_eval 0.5.10 collapses near-equal scores into ties (broken by reverse doc id) while mteb's MRR compares exact floats. Gold strictly rank 1, 30 docs: magnitude 1 → gap 1e-9: NDCG@10 0.0 / MRR 1.0; 1e-8: 0.5 / 1.0; 1e-7: 1.0. Magnitude 10: 1e-7 → 0.387; ≥1e-6 OK. Magnitude 100: 1e-6 → 0.431; ≥1e-5 OK, but an absolute “gap ≥ 1e-5” rule is unsafe for raw scores ≳ 128 (float32 spacing 1.5e-5). Scores `1 − rank/1000` (gap 1e-3) → 1.0 / 1.0 | executed; pinned by `tests/contract/test_score_ties.py` |
| V-09 | `ModelMeta.create_empty(overwrites=…)` exists in 2.21.0 (absent in some older 2.x → keep the introspecting helper) ; mteb ships `mteb/baseline-bm25s` (parity target) | source |
| V-10 | Claude Code (docs, 2026-09): CLAUDE.md target < 200 lines, `@imports` do not save context, path-scoped `.claude/rules/*.md` load when matching files are read, AGENTS.md is read only if imported by CLAUDE.md when a CLAUDE.md exists, hooks (exit 2 = block) are the only deterministic enforcement, subagent/skill frontmatter as used in `.claude/` | code.claude.com docs |
| R-01 | Dataset (mteb stats + HF card): test 8,765 docs / 3,765 queries / 1 relevant doc per query (distinct); docs avg ≈573 chars (5–≈289k), ≈11 exact-duplicate texts; queries avg ≈1,670 chars (152–5,742); qrels train 5,000 | re-audit in G0.2 |
| R-02 | 3rd-party AppsRetrieval NDCG@10: BM25 4.76; BGE-M3 7.37; E5-Mistral 21.33; CodeSage-large-v2 50.45; CodeXEmbed 400M/2B/7B 48.57/74.99/85.22; Qwen3-Embedding-0.6B 75.22 (Jina paper; fp16, 8k tokens; **not measured by us**); jina-code-embeddings 84.17/86.63 (trained on APPS-train; CC-BY-NC) | calibration only; citations and re-verification status in `docs/reference/third_party_scores.md` |
| R-03 | IBM granite-embedding table (CoIR 10-task average, *not* APPS): gte-modernbert-base (149M) 71.5; granite-embedding-english-r2 (149M) 54.8; granite-small-r2 (47M) 53.4 | bake-off candidate signals only |
| V-11 | Claude Code hook exec form (`command` + `args` array) is documented in current docs (added in v2.1.139); the kit uses the universally supported **string-form** `command` and proves execution with the `/canary` test (G0.0) | code.claude.com docs, release notes |
| I-01 | 8,765 = 5,000 + 3,765 ⇒ the 5,000 train solutions are probably distractors inside the test corpus | G0.2 |

Re-verify: `pip download mteb==2.21.0 --no-deps -d /tmp/m && unzip -q /tmp/m/*.whl -d /tmp/m/x && cd /tmp/m/x` then `grep -n "isinstance(model, SearchProtocol)" mteb/abstasks/retrieval.py`, `grep -n "overwrite_strategy\|_DEFAULT_CACHE" mteb/evaluate.py`, `grep -n "key=lambda" mteb/_evaluators/retrieval_metrics.py`, `sed -n 1,30p mteb/tasks/retrieval/code/apps_retrieval.py`. Tie experiment: `tests/contract/test_score_ties.py` (Phase 0/1) reproduces V-08.

## 3. Open items — status, default, resolution
| ID | Item | Status | Default in force | Resolved by / blast radius |
|---|---|---|---|---|
| O-01 | Is resource use scored? | **Resolved (FAQ)**: yes — running time, GPU requirement, model size; weights unpublished | D2 scorecard; D4 "smallest within 1.0 pt" | Weights → revisit D4 threshold via ADR if published |
| O-02 | Search-level model object acceptable? | **Resolved enough (FAQ)**: no restrictions on methods at any stage | Mode A primary (G-AB), Mode B shipped | Residual: sample subclasses `AbsEncoder` → B is the disclosed fallback |
| O-03 | Hardware / wall-clock limit / judge machine | Open | Reference host declared by `acis doctor`; cold official pass target ≤ 4 h on T-rec CPU [E] | Phase 0 measurement; may force a smaller encoder (D4) |
| O-04 | Organizer mteb version | Open | Pin 2.21.0; CI matrix 2.5.x + 2.12.x (+2.1.x smoke) | Adapter uses keyword-only signatures + `**_` |
| O-05 | CSV schema, MRR cutoff | Open | `query_id,corpus_id,rank,score` top-1000 + every `mrr_at_k` in JSON | Cheap to change |
| O-06 | Fine-tuning on APPS-train allowed | Open (FAQ: methods unrestricted) | Allowed, gated (G3), disclosed; frozen-base fallback | Affects D7 only |
| O-07 | P1/Bonus data format | Open (FAQ: APPS throughout ⇒ snippet-library versions likely) | Unified `SnapshotSource`: keyed JSONL, per-version dirs/ZIPs, git | Track B1 |
| O-08 | Hands-on query style | Resolved: "similar to the dataset" | APPS-style primary; short-query regression guard | – |
| O-09 | Owner GPU access | Open (owner) | None assumed; adaptation skipped if none | Phase 3 |

Questions for the organizers (O-03…O-07 plus non-permissive licences, prebuilt artifacts, PPT template): **`docs/official/ORGANIZER_QA.md`** — nine questions with the decision each moves and the default in force; the owner records answers there verbatim (it is under `docs/official/`, so once filled it is authoritative).

## 4. Requirement traceability
| Req | Statement | Design element | Verified by |
|---|---|---|---|
| R1 | Rank snippets; no generation | Engine returns ranked evidence only (INV-1) | `tests/unit/test_evidence.py`, INV-1 property test |
| R2 | Any method, multiple passes | Two-pass core (dense+lexical → LTR, optional PRF) | ladder B3–B9 |
| R3 | CPU, minimal GPU; speed shown | D15, tiers, scorecard, latency budgets | `make bench`, scorecard in RC manifest |
| R4 | Resource use scored | D2, D4, G-R, cold-run honesty D17 | scorecard + `evaluation_time` in JSON |
| R5 | P0 on AppsRetrieval test via MTEB | Mode A/B adapter, official script | contract tests P1–P7, `verify-submission` |
| R6 | JSON (+ csv) deliverable | datetime-safe writer, `run.csv`, `run.trec` | re-score equality to 1e-9 |
| R7 | P1 versions, rebuild in reasonable time | immutable snapshots, content-addressed reuse | P1 acceptance suite, measured update times |
| R8 | Bonus across all versions | lineage layer | Evolution-NDCG@10 vs flat baseline |
| R9 | Hands-on on dataset-like queries | interactive engine + API/CLI, prebuilt index (D17) | `make demo` E2E |
| R10 | Repo, Release, run instructions (PPT and video are owner-made) | README, release pack, `docs/submission/OWNER_HANDOFF.md` | clean-machine dry run |
| R11 | Hands-on queries are unknown and arbitrary | INV-15, routing v1.1, gate G-OOD (spec 10) | `tests/robustness`, `acis eval robustness`, blind-query protocol |
| R12 | “New commits every minute” | eager heads + coalescing, update-time target, commit-stream demo (spec 04, 09 §6) | update-time measurements, `scripts/demo/commit_stream.py` |

## 5. Conflicts between the reference documents → resolution
| # | Conflict | Resolution |
|---|---|---|
| 1 | Tie handling: `nextafter` nudge vs "gap ≥1e-5" vs rank-derived scores | Rank-derived (V-08). `nextafter` gaps are inside the collapse zone; 1e-5 fails at magnitude ≥100 |
| 2 | Qwen3-0.6B APPS 75.22 called both "verified" and "unverifiable" | It is a 3rd-party report (R-02); we measure it (G0.5) before relying on it |
| 3 | TEST policy: 1 touch per RC vs ≤10 runs vs a TEST reproduction gate | TEST only at RCs via `acis eval official`, budget 6; reproduction/parity on dev only |
| 4 | Encoder-only vs search-level primary, with a "compliance premium" | FAQ removes the premium; G-AB threshold is the ordinary +0.5 pt gate; B always shipped |
| 5 | JS/TS parsing, tree-sitter, call graphs, hierarchical retrieval | Out of scope (FAQ + D14); not in the P0/P1/Bonus path |
| 6 | Cross-encoder rerank "gated" vs "dropped" | Dropped: seconds/pair on CPU, resource use is scored; LTR instead |
| 7 | Doc-side hubness vs reference-query normalisation (one doc forbids all hubness, another allows corpus-only doc–doc centrality) | Reference-query-bank offsets are forbidden (they encode train-doc identity). Corpus-only doc–doc centrality is allowed only as a gated LTR feature (E-HUB-C, spec 08 §3) that must pass `eval-integrity-auditor` and leave D1 unchanged |
| 8 | Different mteb compat matrices and truncation/candidate defaults | One matrix (O-04); defaults in spec 02, chosen by G1/G-cand |
| 9 | Demo video / PPT treated as build deliverables | Owner-made (`docs/submission/OWNER_HANDOFF.md`); Claude Code builds `make demo` + evidence files |
| 10 | v1.0 phase order (BM25 RC0, adaptation fourth) | ADR-0002: dense first, RC0 = best frozen dense Mode B, adaptation Phase 3, LTR on the final encoder |
| 11 | Gates on DEV-H ≈ 1,000 queries are under-powered | ADR-0002: all 5,000 TRAIN queries (non-fit) or K-fold OOF (trained); DEV-H = once-per-milestone confirmation |
| 12 | Spec bugs: INV-10 “exactly 1,000” vs small fixtures; TEST budget arithmetic; G0.6 undefined; D1 used TEST queries | `min(top_k, N)`; budget 6 = 5 + post-hoc clean-pool; G0.6 = guard tests + seal check; D1 on a held-out fold |
| 13 | “Sealed by hooks” vs a regex hook being bypassable | Physical seal (D19); hooks catch accidents only; gaps pinned as `xfail(strict)` tests |
| 14 | APPS-style-only routing/instruction/features vs unknown queries | ADR-0003 (spec 10): INV-15, per-route instruction, OOD-guarded LTR, G-OOD |
