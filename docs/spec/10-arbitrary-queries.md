# 10 · Arbitrary-query design — the system serves unknown queries, not templates
Status: **ACCEPTED 2026-09-20 (ADR-0003)** — adds INV-15, gate G-OOD, routing v1.1, per-route instruction; integrated into CLAUDE.md (D18, INV-15), spec 02 (§2, §4, §6, §6b), spec 03 (§7), spec 04 (§6), spec 06 (§5) and spec 08 (N8, C7). Tags: [READ] kit · [EXEC] executed · [E]/[R]/[MEM] as in spec 01.

## 1. Requirement
Hands-on queries are **unknown**. The official text says "similar to the dataset" (APPS problem statements), while the PDF's own example is a *short natural-language question about code behaviour*. The engine must return a sensible ranking, a truthful confidence and no crash for **any** UTF-8 text. Families to serve, none privileged: long problem statements (with or without markers/examples) · short NL questions · behaviour/intent descriptions · identifiers and API names · code fragments as queries · mixed NL + code · partial or constraint-only statements · typos, case, format noise · non-English or mixed-language text · long noisy text · queries with **no relevant document**.

## 2. Rules
- **INV-15 (query-agnostic).** (a) Behaviour depends on the query only through generic mechanisms: normalisation, tokenisation, encoders, learned models, and heuristics that *fail soft*. (b) No closed list of templates, phrases, headers, problem names, moduli or literals may gate correctness. (c) Every pattern-based extractor returns NaN/neutral when it does not fire and is **masked during training** (feature-group dropout). (d) No lookup keyed on query text, no per-query overrides, no answer caches (a content-hash embedding cache is fine). (e) Every heuristic has an OOD test.
- **R-Q1** Routing only adds or re-orders within bounds; the dense+lexical baseline always runs (CLAUDE.md §4 already).
- **R-Q2** Learned re-ranking (LTR) runs only for statement-like queries (§4); otherwise generic fusion.
- **R-Q3** Natural-language version intent is a *hint*, never a silent filter (§6).
- **R-Q4** Nothing is tuned to the hands-on/demo queries; the blind set (§7.4) is used once.

## 3. Audit of the v1.0 kit and the fix for each spot [READ]
| # | Where | Coupling to APPS-style queries | Fix |
|---|---|---|---|
| Q1 | spec 02 §2 one global `TASK` instruction (T1 "programming problem statement…") picked by G1 on APPS statements; §4 stage 1: the router "only chooses the ranker profile" | a short question or identifier is encoded under a "problem statement" instruction | **per-route instruction**: `statement_like` → G1 winner; every other route → generic T3 ("Given a query, retrieve relevant code"), or none if the encoder has no instruction format; G1 runs per route on that route's dev set |
| Q2 | spec 02 §4: `problem_statement` = ≥ 120 tokens or ≥ 2 section markers → LTR profile | a long non-APPS text (bug report, feature description) goes to an LTR trained only on APPS statements | routing v1.1 (§4): length alone is not enough |
| Q3 | Appendix A bridge features (`out_literal_recall, const_jaccard, modulus_match, io_shape_compat, tc_loop_expected, bridge_pmi`) | hand-built for competitive-programming statements; `modulus_match` risks a hard-coded list | generic extractors only (numeric-literal overlap, quoted-string overlap); no fixed modulus list; NaN when absent; feature-group dropout (≈ 25 % masking per group) in LTR training; LTR abstains (dense order) when fewer than ρ of the groups are available |
| Q4 | spec 02 §6b: short queries → untuned weighted fusion, "a regression guard, not a target" | a whole family of arbitrary queries has no gate and no tuning data | **first-class gate G-OOD** (§5); tune the generic fusion on REG data, not on APPS-derived stubs |
| Q5 | spec 04 §6: "Version intent in the query (`as_of`, `in v3`, `latest`, `what changed`) narrows the set deterministically" | closed phrase list; a false positive silently drops results | §6: explicit selectors; NL cues only suggest |
| Q6 | spec 08 C7: certification on "20 held-out dev statements + 5 hand-written variants" | a fixed set is a fixed template set | keep as a smoke test; certification = G-OOD + blind-query protocol (§7.4) |
| Q7 | LoRA adaptation on APPS-train | over-specialisation to the training query style | REG replay (already 25 %) **plus** WiSE-FT-style interpolation of the merged delta, `θ = θ_base + α·Δ`, α tuned on APPS + OOD dev (zero inference cost) [MEM: technique name] |
| Q8 | demo runbook | scripted queries | free-text first; example chips optional and never the only path |

## 4. Routing v1.1
All signals are per-query and batch-invariant: length · weak marker count · **OOD score** (mean cosine of the query embedding to its k nearest TRAIN-query embeddings; query side only, never used to offset any document score) · **feature availability** (fraction of bridge groups that fired). `statement_like` iff OOD ≥ τ_ood **and** availability ≥ ρ, with τ, ρ set on dev so that ≥ 95 % of held-out statements qualify. Everything else, and every routing failure, → `generic`. Per route: instruction (Q1), views, ranker, confidence calibration. Compatibility with CLAUDE.md §4: the TRAIN-query bank is used for *routing only* and never changes a document's score, so it is not a "reference-query offset" and it respects INV-3.

## 5. Gate G-OOD — relative to the frozen base, no arbitrary absolute numbers
Suites: (i) **REG** — mteb code tasks with human-written queries (CosQA, StackOverflowQA, CodeSearchNet, …; exact names via `mteb.get_tasks` in G0) run through the *same* adapter; (ii) **perturbation families** from `tests/robustness/perturb.py` applied to DEV-CV queries; (iii) **route-shift** — statement-like queries with markers removed and non-statement long texts must take the generic path.
Criterion for every family F: `drop_system(F) ≤ drop_base(F) + 1.0 pt` (paired bootstrap): *the system may not be more brittle than the frozen encoder*; REG non-regression ≤ 1.0 pt per task vs the frozen base; mild rows (format noise: strict identity; heading-strip / lower-case: ≤ 0.3 pt). Applies to every adaptation (G3), every LTR/bridge change (G5) and every prep change (G1).
Label-free properties run in CI: `tests/robustness/test_query_agnostic.py`. **[EXEC]** a generic toy engine passes 37/37; a toy engine keyed on one benchmark-style marker is caught by `test_mild_structure_changes_keep_topk[strip_headings]` (top-3 overlap 0.33 < 0.66), and with no engine plugged in the 27 engine tests skip cleanly while the 10 library tests still run. Note: on the 12-document toy corpus the template-coupled toy is only caught with `ACIS_ROBUSTNESS_K=3` (re-verified at merge; with the default K=10 the corpus is too small to expose a flip) — real runs use a fixture corpus of ≥ 200 documents.

## 6. P1 / Bonus queries
Version scope is an explicit selector (`latest | version | snapshot | commit | as_of | all | range`). `all` returns lineage groups. "What changed?" is `compare_versions`, an explicit action. NL text is never parsed into a filter; a detected cue is returned as `interpreted_intent {as_of, confidence}` and the UI offers a one-click apply.

## 7. Evaluation additions
1. `acis eval robustness --config X`: perturbation families × DEV-CV → paired ΔNDCG@10 / MRR@10 vs unperturbed, per family, system vs frozen base → ledger.
2. REG suite for **all** query-side changes, not only adaptation.
3. **No-gold behaviour:** remove the gold document of a DEV query from the corpus; require `no_strong_match=true` at a calibrated rate and a stable top-1 under paraphrase (consistency, not accuracy).
4. **Blind-query protocol:** ≥ 50 queries written by someone who has seen neither the dev data nor the system (families in §1); run **once** before release; a human rates top-3 relevance; failures become issues, never tuning data.
5. Hostile inputs (empty, 1 char, 16k+, NUL/control, RTL/CJK/emoji, injection strings): no crash, typed errors, bounded latency.

## 8. Wiring
`acis.robust_hook:search(query, top_k) -> list[str]` over a fixed fixture corpus (Phase 1; ≥ 200 documents for real runs). CI runs the property tests on every PR; `make robustness` runs the eval; G-OOD rows join the gate register.

## 9. What this does not promise
No engine can rank correctly when the corpus holds no relevant document. The guarantee is **graceful behaviour and honest confidence**, not a right answer for every query.
