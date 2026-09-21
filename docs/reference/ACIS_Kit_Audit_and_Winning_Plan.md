# ACIS Claude Code Kit — strict audit, redesign points and winning plan

Reviewer stance: chief architect, adversarial. Subject: `acis-claude-code-kit.zip` (56 files, 169 KB: `CLAUDE.md`, `AGENTS.md`, 8 specs, 11 subagents, 11 skills, 6 rules, 2 hooks, `settings.json`). Companion files: `acis-kit-v1.1-patch.zip` (tested fixes) and this report.

Evidence tags: **[READ]** read in the kit · **[EXEC]** executed by me in a sandbox · **[SIM]** simulated under stated assumptions · **[E]** estimate, must be measured · **[MEM]** from my memory, unverified · **[JUDGMENT]** my engineering opinion.

---

## 0. Verdict

**Conditional GO.** The architecture is sound and is the best-aligned artifact this project has produced: dense-first retrieval, an honest MTEB integration, integrity machinery, one engine for P0/P1/Bonus. **Do not redesign the core.**

But four things can make you *fail while believing you are safe*, and none of them is a core-architecture problem:

1. **The enforcement layer is unverified and porous.** Hook wiring may not execute at all; the "sealed TEST labels" have six reachable read paths; the irreversible TEST evaluation sits in the *allow* list. (§4: B1–B3)
2. **The plan order and the gate statistics work against the biggest accuracy lever.** Adaptation comes after four phases of tuning it will invalidate, and the gates on ~1,000 queries cannot see real 0.5–1 point gains. (B5, B6)
3. **Nothing covers hours-long jobs, GPU hand-off, deadlines or parallel work.** The kit is a sequential, deadline-free specification for a task whose decisive steps take hours. (B4, H5)
4. **Four spots couple the design to APPS-style queries**, although the hands-on queries are unknown and arbitrary: one global instruction string, an LTR/bridge path selected by query length, an untuned short-query path, and natural-language version-intent regexes. The dense backbone itself is query-agnostic. (H7, §5)

Scope note: **PPT and demo video are yours, not Claude Code's** (your decision); the kit's only obligation there is a working `make demo` plus ledger-backed evidence files you can lift into slides (§9, `OWNER_HANDOFF.md`).

Apply the patch, take the owner decisions in §12, and it is a **GO for Phase 0**. Starting as-is risks a project that *looks* rigorously governed while the guards are silently inactive and the best lever is reached last.

### Scorecard

| Dimension | Grade | One-line reason |
|---|---|---|
| Alignment with Theme-1 guidelines + FAQ | **Strong** | Every stated requirement maps to a design element and a test (§2) |
| P0 architecture | **Strong** | Dense-first, gated LTR, honest protocol; needs reorder and compute realism |
| MTEB integration | **Strong** | Matches what I executed on mteb 2.0.5/2.12.30/2.21.0; two small spec bugs |
| P1 / Bonus design | **Adequate→Strong** | Sound; no numeric update-time target, no "commits every minute" demo |
| Evaluation integrity | **Strong on design, Weak on enforcement** | The seal is regex-based, not physical |
| Claude Code enforcement layer | **Weak (unverified)** | B1–B3 |
| Operability (long jobs, GPU) | **Missing** | B4 |
| Plan order and pacing | **Weak** | B5, H5 |
| Demo / hands-on readiness | **Adequate** | Runbook exists; winner-deciding stage under-specified (H6) |
| Arbitrary-query robustness | **Adequate core, four template-coupled spots** | Dense backbone is query-agnostic; instruction, routing, bridge features, version intent are APPS-shaped (§5) |
| Spec consistency | **Adequate** | Four concrete bugs (H1) |
| Scope discipline | **Weak–Adequate** | Substantial non-scored work (M9) |

### Fix-first list

| # | Fix | Effort | In patch |
|---|---|---|---|
| 1 | Hook wiring to documented string form + **canary test** (B1) | 30 min | ✔ settings.json, PATCH_NOTES |
| 2 | `ask` on every TEST-touching command (B3) | 5 min | ✔ settings.json |
| 3 | Physical seal: TEST labels outside the repo, separate HF cache for the official run (B2) | 1 h | ✔ spec 09 §3 |
| 4 | Detached long-job runner + GPU hand-off protocol (B4) | 2 h | ✔ scripts, GPU_HANDOFF |
| 5 | Reorder phases: adaptation before hybrid/LTR; RC0 = frozen dense floor (B5) | decision | ✔ spec 09 §1 |
| 6 | Gates on all 5,000 TRAIN queries / OOF (B6) | decision | ✔ spec 09 §2 |
| 7 | Fix four spec bugs (H1) | 30 min | ✔ spec 09 §4 |
| 8 | Bootstrap files + dependency allowlist (H2) | 1–2 h | ✔ ADR-0001 (rest: Phase 0) |
| 9 | Two-track plan + cut list + shippable floors (H5) | decision | ✔ TRIAGE.md |
| 10 | Demo **runbook** incl. commit-stream and channel toggle; PPT/video stay with the owner (H6) | 1 h | ✔ spec 09 §6, OWNER_HANDOFF |
| 11 | Arbitrary-query design: INV-15, per-route instruction, OOD-guarded LTR, G-OOD gate, robustness suite (H7) | decision + ≈1 day | ✔ spec 10, `tests/robustness/` |

---

## 1. What I did, and what I could not check

- **Read:** all 56 files, including the official-document transcriptions. I compared `docs/official/theme1_guidelines.md` with the PDF text I had re-extracted: **identical**. `FAQ.md` is present verbatim.
- **Executed:** 40 behavioural cases against `guard.py`; the lifecycle hooks; a dangling-reference audit; a gate-power simulation; the tie/dispatch contract tests on three mteb versions; the patched hook and the job runner.
- **Could not verify:** whether Claude Code accepts an `args` array in hook definitions; the Hugging Face cache layout for the CoIR-APPS test qrels; every third-party score in spec 01 R-02; the mteb code-task names for the REG suite; Claude Code's OS-sandbox settings.

---

## 2. Alignment with the Theme-1 guidelines and the FAQ

| Official requirement (PDF/FAQ) | Kit coverage | Verdict |
|---|---|---|
| Rank snippets for an NL query; no generation | Engine returns ranked evidence only (INV-1) | ✔ |
| Query categorisation / pre-processing, snippet pre/post-processing, multiple passes | 3-class router, segmentation and views, PRF + LTR two-pass | ✔ (router is thin but sufficient) |
| CPU with minimal GPU | D15; GPU offline only | ✔ — but the cold pass can take hours |
| P0: CoIR APPS test via MTEB; NDCG@10, MRR | Mode A/B adapter; all `mrr_at_k` kept in the JSON | ✔ strong |
| "csv file with the responses" | `run.csv` (schema is a guess) | ✔ (ask Q2) |
| JSON in a GitHub release; repo runnable from its steps | Release pack, README outline, `make reproduce` | ✔ / ⚠ a cold run takes hours |
| PPT reviewed in hands-on | **Owner-produced** (out of Claude Code's scope, by your decision) | ✔ owner |
| Demo video | **Owner-produced**; Claude Code guarantees only `make demo` and the evidence files | ✔ owner |
| Hands-on: dataset-like queries; speed shown | Interactive engine, speed panel, prebuilt index | ✔ / ⚠ query encoding is ≈0.5–1.5 s on CPU [E] |
| P1: versions, "new commits every minute", rebuild "in a reasonable amount of time" | Immutable snapshots, eager heads, coalescing, measured update times | ✔ design / ⚠ no numeric target, no commit-stream demo |
| Bonus: all versions; near-identical snippets hard to rank | Lineage cascade, grouped ranking, must beat flat baseline | ✔ |
| FAQ: running time, GPU, model size are factored in | Scorecard, D4 "smallest within 1.0 pt" | ✔ (weights unknown) |
| FAQ: APPS (Python) only; ignore JS | D14 | ✔ |

**Two readings worth stating explicitly.** The PDF says screening ranks on P0 accuracy and that resource use is factored "during the evaluation of a submission"; hands-on then decides among screened finalists. So *accuracy leads for screening*, and *P1/Bonus/demo/PPT decide the winner*. The kit's D4 ("accuracy leads, size breaks near-ties") reads this correctly; its **plan** (§4, H5) does not act on the second half.

---

## 3. What the kit gets right (keep all of this)

- **Dense-first P0**, with BM25 demoted to "must earn its place". Its own reference table (BM25 ≈ 4.8, frozen Qwen3-0.6B ≈ 75, encoders trained on APPS-train ≈ 84–87; all third-party **[R]**) contradicts my earlier lexical-heavy blueprint. If those numbers hold, the kit is right and I was wrong. (Verify in G0.5.)
- **Verified MTEB facts**, including a tie experiment that extends mine to magnitude 100, and the correct conclusion (**rank-derived scores**, unique `ModelMeta`, `cache=None`, no `predict()`).
- **Honest adaptation protocol:** decontamination, cross-fitting, a train-document exposure diagnostic, a clean-pool score.
- **Integrity discipline:** ledger, split lock, `verify-submission`, "no number without a ledger id", a judge-simulator subagent.
- **One engine, thin surfaces;** content-addressed immutable snapshots; evidence-gated lineage ("unknown beats a guessed merge").

---

## 4. Findings

### Blockers — fix before Phase 0

**B1 · The whole enforcement layer may not run (hook wiring is unverified).**
`settings.json` wires every hook as `"command": "python3", "args": [...]`. I could not verify that Claude Code accepts an `args` array (documented examples use one command string **[MEM]**). If it does not, each hook is a bare `python3` reading the hook JSON from stdin. I ran exactly that **[EXEC]**: exit 0 or 1, and both are non-blocking. The TEST seal, protected files, banned constructs and Stop-time tests would silently vanish, and `/hooks` would still list them.
*Fix:* string-form commands (patch), plus the **canary** in `PATCH_NOTES.md` (create `data/sealed/canary.txt`, ask Claude to read it, expect "ACIS guard blocked"). Make it gate **G0.0**; repeat after every Claude Code upgrade.

**B2 · "Sealed TEST labels" is not actually enforced.**
Executed against `guard.py` v1.0 **[EXEC]**: these all *pass* the guard — a recursive `Grep` rooted at `data` or `.`; `grep -r … data/`; `find data … | xargs head`; `cat data/*/*.tsv`; a script that reads the file internally; and the Hugging Face cache path where mteb/`datasets` actually store the test qrels. The ledger is also writable by `python -c "open('runs/ledger.jsonl','a')…"` and `cp … runs/ledger.jsonl`. A regex over command text cannot see expansion or indirection. The kit's claim N3 ("sealed labels (guard + hooks)") therefore overstates, and INV-14 forbids overstated claims.
*Fix:* make the seal **physical**: keep TEST qrels outside the repo tree (`~/.acis-sealed/`), let only the official run use `HF_HOME` there, keep the dev environment on `split="train"`, and reword N3/INV-8 to "hooks catch accidental access". The patched hook (v1.1) closes the reproducible gaps; three residual gaps are pinned as `xfail(strict)` tests so nobody mistakes the hook for a boundary.

**B3 · The irreversible, budgeted TEST evaluation is in the allow path.**
`permissions.allow` contains `Bash(make *)`; `guard.py`'s `FINAL_EVAL_OK` explicitly *permits* `make rc-official` and `acis eval official`; nothing asks. The `/rc-official` skill says "the owner explicitly confirmed" — that is prompt text, and `disable-model-invocation` only stops the model from invoking the *skill*, not from running the command.
*Fix (5 min):* `ask` rules for `make rc-official*`, `acis eval official*`, `python -m acis.eval.final*` (patch), and run the official pass from the owner's terminal.

**B4 · No operating procedure for long jobs or GPU work.**
Decisive steps take hours: encoder bake-off, corpus embedding, LoRA training, and the official cold pass (spec 02 §7: 1.4–8 h for a 0.6B encoder **[E]**). Claude Code's Bash tool has a bounded timeout and sessions can be interrupted; `/rc-official` tells Claude to run `make rc-official` in the foreground. The adaptation plan is heavy: my estimate is ≈ 7.8 M tokens/epoch and ≈ 2e16 FLOPs/epoch for a 0.6B LoRA, i.e. **≈ 9 GPU-hours on a T4** for 4 fold models + final over 3 epochs **[E]** — and a T4 has no bf16 while the spec says bf16.
*Fix:* `scripts/run_detached.sh` + `job_status.sh` (patch; executed end to end), `docs/GPU_HANDOFF.md` (export → notebook → checkpoint/resume → import + CPU parity gate), and cheaper variants (2-fold cross-fitting, fewer negatives, shorter length, adapt only the smaller encoder).

**B5 · The phase order puts the biggest lever fourth and wastes the work before it.**
Phase 1 builds code-aware BM25 with parity tests and an **RC0 = BM25 JSON** (≈ 4.8 NDCG@10 **[R]**) although gate G2's own default is "dense-only". Phase 3 (bridge features, PRF) precedes Phase 4 (adaptation), yet adaptation changes the embedding space, so spec 07 already says "re-run G-M over adapted variants" — the Phase-3 tuning is redone. Nothing shippable exists between RC0 (garbage) and RC1 (Phase 6).
*Fix:* 1 harness → 2 dense + bake-off → **RC0 = best frozen dense, Mode B** → 3 **adaptation** → 4 hybrid/LTR on the *final* encoder → 5 freeze. (spec 09 §1)

**B6 · The gates are statistically underpowered.**
All gate decisions use DEV-H ≈ 1,000 queries with the rule "Δ ≥ +0.5 pt **and** CI lower bound > 0". **[SIM]** (12 % of queries move ±0.45; paired SD ≈ 0.16; re-estimate from real runs): the CI half-width is ≈ 1 pt, so the rule effectively demands Δ ≳ 1 pt.

| true gain | accepted at n = 1,000 | accepted at n = 5,000 |
|---|---|---|
| +0.5 pt | 17 % | 51 % |
| +0.75 pt | 34 % | 87 % |
| +1.0 pt | 53 % | 99 % |
| +1.5 pt | 86 % | 100 % |

Real improvements of 0.5–1 pt — most prompt, fusion, PRF and feature levers — are mostly rejected; and repeated looks at the same 1,000 queries breed winner's-curse on the ones that pass. Worse, **G-AB defaults to Mode B** whenever Mode A's gain is under ~1 pt, so the Mode-A machinery may go unused for the primary JSON.
*Fix:* gates on components that **fit nothing** use **all 5,000 TRAIN queries**; trained components are judged on **5-fold OOF over all 5,000**; DEV-H becomes a once-per-milestone confirmation.

### High

**H1 · Four spec bugs (each makes a gatekeeper fail or Claude improvise).** [READ]
- **INV-10** demands "exactly `top_k` = 1,000 entries per query", but the spec's own 500-doc fixture and toy G0.1 corpus cannot have 1,000. Use `min(top_k, N)`.
- **TEST budget** "≤ 6: RC0 ×1, RC1 ×2, contingency ×2" sums to **5**. The 6th (presumably the post-hoc clean-pool score) is unassigned.
- **G0.6** appears in the CLAUDE.md phase table but is defined nowhere (STATUS lists G0.1–G0.5).
- The **D1 exposure diagnostic** uses TEST *queries* as an input to the G3 shipping decision — a statistic over the test-query batch, against INV-3's spirit. Compute the analogue on a held-out fold.

**H2 · Bootstrap materials are missing.** No `pyproject.toml`, `Makefile`, `tests/conftest.py`, `configs/*` (my audit found 19 referenced-but-absent paths **[EXEC]**, most legitimately Phase-0 outputs — but nothing says so, and the hooks/skills presume them). There is also no dependency allowlist, so every `uv add` stalls on an ask + ADR. *Fix:* ADR-0001 (patch) and a Phase-0 scaffold task.

**H3 · The "permissive licence only" rule is a strategic choice nobody asked the organizers about.** Spec 01 lists code encoders trained on APPS-train (0.5B/1.5B) at ≈ 84–87 **[R]**, excluded for CC-BY-NC. If adaptation reaches similar levels the rule costs nothing; if not, it could cost several points. Add the question (ORGANIZER_QA Q5) and measure the excluded models as a *reference*, as spec 02 already intends.

**H4 · Adaptation compute is unbudgeted** (see B4): the cross-fit + final training plan needs ≈ 9 T4-hours or ≈ 1.5 A100-hours **[E]**, plus mining passes. Without a hand-off protocol, "skip ⇒ frozen base" becomes the de-facto outcome and P0 stays near the frozen number.

**H5 · No deadline, no parallelism, no cut line.** The kit forbids starting a phase before the previous gate passes, so every long job idles the whole project. P1/Bonus/demo (Phases 7–9) come only after RC1, yet they decide the winner in hands-on. *Fix:* `docs/TRIAGE.md` — two tracks (accuracy ∥ systems/demo), shippable floors F0–F4, an explicit cut order.

**H6 · The hands-on demo *runbook* is under-specified where the winner is chosen.** (PPT and video are yours; Claude Code's obligation is `make demo` plus evidence files.) The UI is "one static search page"; the PDF's "commits added every minute" has **no commit-stream demo**; there is no baseline-comparison panel and no rehearsal/fallback plan. *Fix:* spec 09 §6 (runbook acceptance) and `docs/submission/OWNER_HANDOFF.md` (ledger-backed figures and tables you can lift into slides).

**H7 · Four spots couple the design to APPS-style queries** (hands-on queries are unknown and arbitrary, and the PDF's own example query is a short natural-language question, not a problem statement). [READ] Details and fixes in §5.

### Medium

- **M1 · CLAUDE.md density.** 145 lines but ≈ 22.8 k characters ≈ **6.3 k tokens**, with rows up to 605 characters **[EXEC]**. It complies by line count, not by density; §8 and §8b duplicate specs 04/05/08. Trim to ≈ 4 k tokens.
- **M2 · Guard false positives** [EXEC]: `uv run pytest 2>&1 | tail; git add runs/ledger.jsonl` is denied; a comment containing `eval(` is denied; `Glob tests/**/test_label*.py` is denied. Each block costs turns. *Fixed in guard v1.1.*
- **M3 · The Stop hook is advisory, not an invariant.** It blocks at most twice per session, then permanently waives **[EXEC]** (calls 3–4 exit 0). Soften the wording; keep `make test-fast` under ≈ 60 s.
- **M4 · Web access for the model radar** (G0.4 "re-read live leaderboards") has no `WebFetch` allow rules, so it prompts repeatedly. *In patch.*
- **M5 · `rules/mteb-eval.md`** globs `configs/**` and `scripts/**`, loading it on any config read (context noise).
- **M6 · Third-party scores (R-02) have no citation file.** Claude cannot re-verify "the Jina paper". Add `docs/reference/third_party_scores.md` with exact sources.
- **M7 · "Quick reproduction" is not quick.** `--cache-verify` still encodes 3,765 long queries (≈ 45 % of the compute) and the kit forbids shipping test-query vectors. A judge following the README on a laptop faces hours. State the expected runtime per hardware tier at the top of the README.
- **M8 · SLO ≤ 4 h is generous while resources are scored.** Keep D4, but until the organizers answer Q3 prefer the smaller candidate when the accuracy gap is < ~3 points and the cold pass exceeds ≈ 2 h on the declared host **[JUDGMENT]**.
- **M9 · Over-scope.** Not tied to any Samsung criterion: bubblewrap/seccomp sandbox tiers, cosign signing, `SOURCE_DATE_EPOCH` reproducible containers, ≥ 1 h fuzzing per parser, custom semgrep rules, hash-chained ledger *plus* `acis eval repro` re-derivation, an LLM planner for the agent. Keep parse-only workers, limits and no-exec; cut the rest first (TRIAGE.md).

### Low

- `.gitignore` lacks `.env`, `.DS_Store`, `*.parquet`, `*.safetensors` (partly covered by `models/`).
- INSTALL.md gives no Claude Code version/model guidance or `--permission-mode plan` hint.
- `settings.json`'s `Bash(uv run *)` is broad (arbitrary code) — acceptable only with a working guard (B1).
- Hooks assume `python3` on PATH: native Windows is unsupported (stated).
- `configs/official.yaml` is resolved relative to the CWD; a judge may run from anywhere.


---

## 5. Arbitrary-query audit — "unknown queries, not question templates"

**Verdict.** The *backbone* is query-agnostic: a dense semantic encoder, a generic lexical channel, exact search. I found **no hard-coded query strings or answer tables** in the kit (it contains specifications, no code yet). But the design is coupled to APPS-style text in **four structural spots** plus four smaller ones, and the evaluation lacks a robustness gate. Even the official PDF contradicts the assumption that every hands-on query is a long statement: its own example is a *one-line natural-language question*. All findings [READ]; fixes in `docs/spec/10-arbitrary-queries.md`.

| # | Where in the kit | Coupling | Fix |
|---|---|---|---|
| Q1 | spec 02 §2: **one global** instruction `TASK` (T1 "programming problem statement…") chosen by G1 on APPS statements; §4 stage 1: the router "only chooses the ranker profile" | A short question or an identifier is encoded under a "problem statement" instruction | Per-route instruction; non-statement routes use the generic "retrieve relevant code" instruction; G1 per route |
| Q2 | spec 02 §4: `problem_statement` = ≥ 120 tokens or ≥ 2 markers → LTR | A long non-APPS text (bug report, feature request) is sent to an LTR trained only on APPS statements | Routing v1.1: OOD score + feature availability, not length alone |
| Q3 | Appendix A bridge features (`modulus_match`, `io_shape_compat`, `tc_loop_expected`, …) | Hand-built for competitive-programming statements; risk of a fixed modulus list | Generic extractors only; NaN when absent; **feature-group dropout** while training the LTR; LTR abstains when few groups fire |
| Q4 | spec 02 §6b: short queries → untuned weighted fusion, "a regression guard, not a target" | A whole family of arbitrary queries has no gate and no tuning data | First-class **gate G-OOD** on REG tasks (real human-written queries) + perturbation families |
| Q5 | spec 04 §6: NL version intent ("in v3", "latest", "what changed") **narrows the set** | Closed phrase list; a false positive silently drops results | Explicit selectors; NL cues only *suggest* (`interpreted_intent`) |
| Q6 | spec 08 C7: certified on "20 held-out statements + 5 hand-written variants" | A fixed set is a fixed template set | Smoke test only; certify with G-OOD + a **blind-query protocol** (≥ 50 queries from someone who has not seen the system) |
| Q7 | LoRA adaptation on APPS-train | Over-specialisation to the training query style | REG replay (kit) **plus** interpolation of the merged LoRA delta (`θ = θ_base + α·Δ`, α tuned on APPS + OOD; zero inference cost) |
| Q8 | Demo runbook | Scripted queries | Free-text first; example chips optional |

**New invariant and gate (proposed, needs `/adr 0003`).**
- **INV-15 (query-agnostic):** behaviour depends on the query only through generic mechanisms; no closed list of templates/phrases/headers/moduli gates correctness; every pattern extractor fails soft and is masked in training; no lookups or answer caches keyed on query text.
- **G-OOD**, *relative to the frozen base* so there are no arbitrary absolute numbers: for every perturbation family, `drop(system) ≤ drop(base) + 1.0 pt`; REG non-regression ≤ 1.0 pt per task; format noise must be a strict no-op.

**Executed evidence [EXEC].** `tests/robustness/` (in the patch) holds a generic, template-free perturbation library and label-free properties. A generic toy engine passes **37/37**. A toy engine keyed on one benchmark-style `-----Input-----` marker is **caught** (`strip_headings` flips its top-3, overlap 0.33). With no engine plugged in, the 27 engine tests skip cleanly and the 10 library tests still run. Limit: the toy engines validate the *test logic*, not your real engine; the real gate is `acis eval robustness` + REG once the engine exists.

**What this cannot promise:** when the corpus holds no relevant document, no design ranks correctly. The guarantee is graceful behaviour and honest confidence (`no_strong_match`), not an answer for every query.

---

## 6. Architecture verdict — does anything need redesign?

**No core redesign.** Changes are localised:

| Component | Verdict | Why |
|---|---|---|
| Single engine, thin surfaces (MTEB adapter, API, CLI) | **Keep** | Correct and verified |
| Dense-first P0, whole-snippet unit, head+tail truncation | **Keep** | Supported by the kit's own evidence [R]; measure in G0.5 |
| Code-aware BM25 with parity tests as **Phase 1** | **Demote** | ≈ 4.8 NDCG@10 [R]; use stock `bm25s` first; keep a light BM25 for the degradation ladder and for short/identifier queries (which matter for arbitrary queries) |
| LTR + bridge features + PRF | **Keep, gated and guarded** | Route guard + feature dropout (Q2/Q3); PRF default off |
| LoRA adaptation | **Keep, move earlier, budget the GPU** | Biggest lever [R]; add α-interpolation for robustness (Q7) |
| Distillation into a ≤ 0.3 B student | **Optional add** | Attacks the FAQ's size/time scoring directly |
| Mode A (search-level) + Mode B (encoder-only) | **Keep** | FAQ: "no restrictions on methods"; decide on OOF over all 5,000 queries (B6) |
| Rank-derived scores, unique `ModelMeta`, `cache=None` | **Keep** | Reproduced by my executed tests |
| Cross-encoder, HyDE, graph, hierarchy, ANN | **Cut (agree)** | Cost without evidence |
| Router (3 classes by length/markers) | **Change** | Routing v1.1 (Q2) |
| NL version-intent narrowing | **Change to hint** | Q5 |
| Snapshot store / P1 | **Keep; add** a numeric update-time target and a commit-stream demo | The PDF says "commits every minute" |
| Lineage / Bonus | **Keep** | Evidence-gated, must beat flat |
| Agent layer | **Keep last and optional** | Off the scored path |
| Sandbox tiers S2, signing, fuzz ≥ 1 h, custom semgrep | **Cut first** | Not a Samsung criterion (M9) |
| Hook-based TEST seal | **Change to physical seal** | B2 |
| Phase order | **Change** | B5 |

---

## 7. CLAUDE.md evaluation

**Content quality:** high. Precedence is explicit; invariants are testable; rules are concrete; decisions carry evidence pointers.

**Problems**
1. **Density.** ≈ 6.3 k tokens, rows up to 605 characters. Salience drops; the model may weight all 17 decisions, 14 invariants, ~20 gates and 12 rules equally. §8 and §8b duplicate specs 04/05/08.
2. **Enforcement mismatch.** Many "never" rules have no mechanical enforcement:

| Rule | Mechanism in the kit | Real strength |
|---|---|---|
| TEST labels sealed | deny rule + regex hook + prompt | **Weak** (B2) — fix with a physical seal |
| TEST budget ≤ 6 | ledger counter + prompt; `make *` allowed | **Weak** (B3) |
| Protected files | deny rules + hook | Medium (Edit tool; Bash partly) |
| No `pip install` | hook | Medium |
| Banned constructs in `src/` | hook on Write/Edit content | Medium |
| Phases in order | prompt + `phase-gatekeeper` | Weak |
| Tests first | prompt + Stop hook (waived after 2 blocks) | Weak |
| No number without a ledger id | prompt + `/evidence-pack` scan at the end | Medium |
| Query-agnostic behaviour | *absent* | **Missing** → INV-15 + G-OOD |

3. **Internal conflicts** (H1): INV-10 vs fixtures; TEST budget arithmetic; G0.6; D1 vs INV-3.
4. **Scope line to change:** §1 says the owner records the video (good) but §9/§10/spec 08 still make Claude Code produce a PPT outline and preflight-check the PPT. Per your decision, remove those (§9 below).

**Recommended shape (≈ 4 k tokens):** identity and objective (3 lines) · precedence · official-requirements digest (≤ 10 lines) · invariants INV-1…15 · non-negotiable rules · phase table with pointers · subagent/skill map. Move the D-table detail, §8 essentials and §8b novelty to specs (they already exist there). Add: INV-15, G-OOD, the physical-seal rule, "PPT/video are owner-made".

---

## 8. Claude Code configuration review

| Piece | What works | What breaks or is missing |
|---|---|---|
| `settings.json` | Sensible allow/ask/deny; `$schema`; env pins | Hook wiring unverified (B1); TEST commands not in `ask` (B3); no web allow-list (M4); no timeout env for long commands |
| `guard.py` | Clear contract; fails open on malformed input; stdlib-only | 11 executed failures (B2, M2) → v1.1 fixes 11/11; three gaps are structural |
| `lifecycle.py` | Session context, ruff, Stop tests; never traps a session | Stop is advisory after 2 blocks (M3); `configs/` path is CWD-relative |
| 11 subagents | Frontmatter valid; sensible read-only vs implementer split; two pinned to `sonnet` | No dedicated **training/GPU** agent and no **demo-runbook** agent (the main session must carry them) |
| 11 skills | `disable-model-invocation` on all but two; typed slash commands | `/rc-official` relies on prompt-level confirmation (B3); `/evidence-pack` and `/preflight` still include PPT items |
| 6 rules | Path-scoped, each names its spec | `mteb-eval.md` globs `configs/**`, `scripts/**` (noise) |
| `INSTALL.md` | Clear six steps | No Claude Code version/model guidance; no note that hooks must be canary-tested |

### Files to add (what the kit misses)

| File | Purpose | Status |
|---|---|---|
| `pyproject.toml`, `Makefile`, `tests/conftest.py`, `configs/*.yaml` skeleton | Toolchain the hooks and skills presume | Phase 0 (H2) |
| `docs/adr/0001-dependency-allowlist.md` | Stops `uv add` stalls | ✔ patch |
| `docs/official/ORGANIZER_QA.md` | Nine questions with decision impact | ✔ patch |
| `docs/GPU_HANDOFF.md`, `scripts/run_detached.sh`, `job_status.sh` | Long jobs and GPU work (B4) | ✔ patch (runner executed) |
| `docs/TRIAGE.md` | Tracks, floors, cut order (H5) | ✔ patch |
| `docs/spec/09-…`, `docs/spec/10-arbitrary-queries.md` | Plan v1.1 and arbitrary-query design | ✔ patch (proposals; need ADRs) |
| `tests/security/test_guard_hook.py`, `tests/contract/*`, `tests/robustness/*` | Executable contracts | ✔ patch (all run) |
| `docs/submission/OWNER_HANDOFF.md` | What Claude Code hands to you (PPT/video are yours) | ✔ patch |
| `docs/reference/third_party_scores.md` | Sources for R-02 so Claude can re-verify | **you** (I cannot verify those scores) |
| `docs/reference/` ← `ACI_Architecture_Blueprint.md`, `aci_mteb_adapter_skeleton.py` | Tested adapter skeleton and tie logic | drop them in (non-authoritative) |

---

## 9. Implementation strategy v1.1

**Scope boundary (your decision).** Claude Code builds the engine, CLI/API, a minimal UI, `make demo`, the demo runbook and **ledger-generated evidence files**. **You** make the PPT and the demo video. Concretely: drop `PPT_OUTLINE.md` generation from `/evidence-pack`, check C12 from `/preflight`, and the PPT items from Phase 11 and spec 08 §5; keep C1–C11 and C13. `OWNER_HANDOFF.md` lists what you get.

**Phases and tracks** (spec 09 §1, TRIAGE.md)

| Track A — accuracy | Track B — systems (parallel from Phase 2) |
|---|---|
| 0 Foundations + hook canary + physical seal + runner | — |
| 1 Harness, tie/dispatch contract tests, robustness hook, adapter v0 (Mode B); **no TEST touch** | freeze the `AcisEngine` interface |
| 2 Dense + bake-off on all 5,000 TRAIN queries → **RC0 = frozen dense, Mode B** (1 TEST touch; a shippable floor) | store/P1 |
| 3 **Adaptation** (LoRA + α-interpolation; optional distillation) via GPU hand-off; G3, G-OOD | lineage/Bonus |
| 4 Hybrid/LTR/PRF on the final encoder; G2, G4, G5, G-AB on OOF; routing v1.1 | API/CLI/UI + `make demo` |
| 5 Freeze → RC1 (A+B) → verify-submission → scorecard | agent (last, optional) |

**TEST-touch plan (6):** RC0 ×1, RC1 ×2 (A+B), contingency ×2, post-hoc clean-pool ×1. The official pass is started from **your** terminal (or an `ask`-gated command), never silently by Claude.

**First session script:** apply the patch → run the canary → `/adr 0001` → decide `/adr 0002` (spec 09) and `/adr 0003` (spec 10) → answer the compute questions (§12) → `/phase-start 0`.

**Cut order when behind:** agent → S2 sandbox/signing/reproducible container → 1 h fuzzing → custom semgrep → PMI bridge table → PRF → second encoder. **Never cut:** metric parity + tie test, cold honest run, strict mode, scorecard, verify-submission, P1 isolation + atomic activation, `make demo`, G-OOD.

---

## 10. What makes the system more accurate (ordered by expected value)

Calibration numbers are the kit's third-party figures **[R]** (BM25 ≈ 4.8; frozen Qwen3-0.6B ≈ 75; encoders trained on APPS-train ≈ 84–87). None is measured here. G0.5 replaces them.

1. **Best frozen encoder** from a bake-off, non-permissive models kept as *reference only*; measured on all 5,000 TRAIN queries.
2. **Supervised adaptation early** (LoRA on decontaminated pairs; alternate solutions as positives; false-negative-filtered hard negatives) — the largest documented jump. Merge with **α-interpolation** and gate on G-OOD so accuracy gains do not cost generality (arbitrary queries).
3. **Distillation** (optional): an offline larger teacher into a ≤ 0.3 B student to meet the FAQ's size/time scoring.
4. **Per-route prompt/view/length sweeps** (Q1) on the right dev sets.
5. **Contract-aware LTR + PRF** on the *final* encoder, guarded by routing v1.1 and feature dropout.
6. **Failure-bucket loop:** recall failure vs ranking failure vs sibling-problem confusers vs truncation vs parse; add a feature only if it repairs a bucket.
7. **Second encoder** only if ≥ +1.0 pt at ≤ +60 % cold time.
8. **Calibrated confidence and `no_strong_match`** so unknown queries with no relevant document fail honestly.

**Do not:** use LLMs, HyDE or a cross-encoder in the scored path; normalise over the query batch; use document IDs or train-document detectors; tune on TEST or on the blind-query set.

---

## 11. Hands-on runbook (what Claude Code delivers; the video and slides are yours)

- **Free-text first.** The judge types anything: a full statement, a one-liner, an identifier, a code fragment. Chips are optional.
- **Arbitrary-query moment:** the same intent phrased three ways returns the same top result; typos and case noise do not flip it. This is the visible proof of INV-15 that a template-tuned system cannot show.
- **Speed and resources:** per-stage timing bar, cold vs warm, scorecard (params, MB, RSS, GPU none).
- **Channel toggle:** BM25-only / frozen dense / adapted / full, with ledger-cited numbers.
- **P1 commit stream** (the PDF's "new commits every minute"): replay versions at a fixed cadence, live freshness gauge, units re-embedded, rollback, `kill -9` during a build with recovery.
- **Bonus:** `all` view grouped by lineage with a timeline, against the flat list.
- **Failure honesty:** an unrelated query returns `no_strong_match`, not a confident guess.
- **Owner rehearsal:** offline, models pre-warmed, three backup queries, a recorded fallback clip.

---

## 12. Owner inputs still needed, and status of the earlier questions

**Blocking for B4/B5 and the encoder tier:**
1. Dev machine cores and RAM. 2. GPU access for one-off work (Colab/Kaggle counts) and roughly how many GPU-hours. 3. Who runs the official cold pass, and on what hardware. The kit's own STATUS still says `reference_host: undeclared` and O-09 open. My estimates: a 0.6B encoder cold pass ≈ 1.5–8 h on 8 CPU cores; adaptation ≈ 9 GPU-hours on a T4 or ≈ 1.5 on an A100 **[E]**. If there is no GPU, adaptation is skipped and P0 stays near the frozen number, so the encoder tier should then be chosen for the best frozen accuracy within the time you can afford.

**Also decide:** accept or reject spec 09 (`/adr 0002`) and spec 10 (`/adr 0003`); send `ORGANIZER_QA.md`.

**The three questions from the earlier message**
- *FAQ text:* now supplied verbatim in the kit (`docs/official/FAQ.md`): APPS/Python only; no restriction on methods; "running time, GPU requirement, model size, etc." are factored in. Weights are still unknown.
- *Compute:* still open (above).
- *Is a search-level model acceptable?* The FAQ's "no restrictions on specific methods" supports it, and the class stays an `AbsEncoder` subclass. Residual harness risk is covered by shipping Mode B; ask organizers (Q6).

**The "resolved from the sources" list:** every item matches the PDF text I re-read. Two nuances: (a) an absolute "gap ≥ 1e-5" tie rule is unsafe for raw scores ≥ ~128 — pinned by a test; the kit's rank-derived scores are the right fix; (b) the demo video is in the hands-on paragraph, not the checklist — you make it.

---

## 13. Go / no-go checklist

- [ ] Hook canary passes (blocked read, blocked `cat`, blocked `eval(`) — or the hooks are rewired until it does.
- [ ] `ask` rules for TEST-touching commands applied.
- [ ] TEST labels physically outside the repo tree; the dev environment uses `split="train"`.
- [ ] `tests/security` and `tests/contract` green on the pinned mteb and the compatibility matrix.
- [ ] `run_detached.sh` used for anything longer than ≈ 10 minutes; GPU hand-off protocol agreed.
- [ ] Owner decisions taken: spec 09 (phase order, gate power), spec 10 (arbitrary queries), compute answers, organizer questions sent.
- [ ] Spec bugs fixed: INV-10 `min(top_k, N)`, TEST budget = 5 + 1, G0.6 defined, D1 on a held-out fold.
- [ ] `docs/reference/third_party_scores.md` filled, or R-02 treated as unmeasured until G0.5.
- [ ] PPT/video items removed from Claude Code's scope.

**If all boxes are ticked: GO.** If B1–B3 are skipped, the guards may be inactive and the project would still report itself as protected. If B5/B6 are skipped, expect the gates to reject real gains and the best lever to arrive last.

---

## Appendix A — guard hook results [EXEC]

| | v1.0 | v1.1 (patch) |
|---|---|---|
| Behavioural matrix (40 must-deny / must-ask / must-allow cases) | **11 failures** | **40 pass** |
| Known structural gaps (3 rows) | not covered | pinned as `xfail(strict)` |
| Failures fixed | HF-cache test-qrels path; `cp`/`dd`/python-`open` ledger writes; recursive `grep -r`/`find`/Grep on `data/`; `2>&1 …; git add runs/ledger.jsonl`; `eval(` inside a comment; `Glob tests/**/test_label*.py` | |

## Appendix B — gate-power simulation [SIM]
Model: 12 % of queries change rank materially (|ΔNDCG@10| = 0.45), winners/losers ratio set by the true gain; paired SD ≈ 0.16; normal-approximation CI; rule Δ ≥ 0.5 pt and lower bound > 0; 4,000 trials per cell. The SD is an assumption: re-estimate from real paired runs. Direction of the conclusion is robust: at n ≈ 1,000 the CI half-width is ≈ 1 pt.

## Appendix C — referenced-but-absent paths [EXEC]
`Makefile`, `README.md`, `pyproject.toml`, `uv.lock`, `configs/{dev,official,targets}.yaml`, `configs/gates/G-M.yaml`, `configs/splits.lock.json`, `runs/{ledger.jsonl,hardware.json}`, `src/acis/{mteb_adapter,mteb_meta}.py`, `tests/{contract/test_score_ties,security/test_qrels_guard,unit/test_evidence}.py`, `docs/submission/{evidence,preflight_report,README.generated}.md`. Almost all are legitimate Phase-0/1 outputs; the kit should say so.

## Appendix D — assumptions I could not verify
Claude Code's hook `args` support · HF cache layout for the test qrels · every third-party score in R-02 · REG task names · Claude Code OS-sandbox settings (worth evaluating as a *real* seal) · licences in ADR-0001 (from memory) · model facts (sizes, licences, remote-code needs) in the specs.
