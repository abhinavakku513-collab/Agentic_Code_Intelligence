# ACIS status — single source of progress truth (updated only by /phase-gate on PASS)
kit_version: 1.1 (ADR-0001…0004)
current_phase: 2            # Dense engine and bake-off — started on the owner's explicit instruction
phase_state: in-progress    # not-started | in-progress | gate-pending | done
last_gate: none             # NO phase gate has been recorded as PASS yet — see the note directly below
deadline: undeclared        # OWNER: put the real date here; work backwards from docs/TRIAGE.md floors F0–F4
hook_canary: mechanism-verified, owner sign-off pending   # see docs/PHASE0_REPORT.md §G0.0
test_touch_used: 0 of 6     # RC0 x1, RC1 x2 (A+B), contingency x2, post-hoc clean-pool x1
reference_host: dev host declared in runs/hardware.json (T-min, 8c/9.7GB, no GPU) — NOT the scorecard host
compute: undeclared         # OWNER: dev cores/RAM · GPU access + hours (Colab/Kaggle counts) · who runs the official cold pass
faq: received 2026-09-20 (docs/official/FAQ.md) · organizer answers: none yet (docs/official/ORGANIZER_QA.md)
scope: PPT and demo video are owner-made (docs/submission/OWNER_HANDOFF.md)

## Phase 2 started without a recorded gate — deliberately, on the owner's instruction
Phase 1 is built and independently verified (an external gatekeeper re-derived every acceptance row), but its gate
cannot be *recorded* until the owner acts, and the owner chose to proceed rather than wait. This is a departure
from CLAUDE.md §5 and it is written down here rather than left implicit.

What that costs: Phase 2 decisions (G-M, G1) will be taken against a Phase 1 whose gate row says `none`. If the
owner's canary run later reveals a hook problem, work done after this point is the work that has to be re-examined.
Nothing else is affected — the harness evidence stands on its own, and the held-out touch counter is still 0 of 6.

**Also blocking in practice:** this sandbox currently has no network, so model weights cannot be downloaded.
G0.4/G0.5 and the G-M bake-off therefore cannot run here. Phase 2 work proceeds on the parts that need no weights
(encoder runtime, vector cache, model registry, batching, the bake-off harness itself), each behind an interface a
real encoder plugs into, so the measurement passes can run unchanged the moment weights are available.

## Why the Phase 1 gate is not recorded as PASS
The Phase 1 build and its acceptance criteria are complete and independently verified (docs/PHASE1_REPORT.md), but
`/phase-gate` may only write a PASS when its predecessor has one, and Phase 0 needs three owner actions that this
tool cannot perform or sign for:

1. **Hook canary (G0.0).** The mechanism is verified — all three canary conditions were blocked in-session and the
   guard tests are green — but `docs/CANARY.md` asks the owner to create `data/sealed/canary.txt` in their own
   shell and record the Claude Code version here. Only the owner can do the first and attest the second.
2. **G0.4 / G0.5 were deliberately not run** (model radar, zero-shot on all 5,000 dev queries). Both need encoder
   weights and multi-hour CPU passes, and both feed **G-M, which Phase 2 decides** — no Phase 1 acceptance row
   depends on them. The reasoning, the risk accepted and the alternatives are written up in
   `docs/adr/0005-defer-g04-g05-to-phase-2.md`, which is **proposed and needs owner confirmation**.
3. **The compute and deadline rows above are still undeclared**, and the declared dev host is T-min, so it is not
   the host a resource scorecard may quote (docs/spec/06 §0 reserves that for T-rec).

Nothing is blocked *technically*: the `AcisEngine` interface is frozen, so Track B could start today.

## Owner actions pending
1 Run the canary (docs/CANARY.md) and record the version above · 2 answer the compute/deadline questions ·
3 send docs/official/ORGANIZER_QA.md · 4 confirm or reject ADR-0005 (the G0.4/G0.5 deferral) ·
5 fill docs/reference/third_party_scores.md or leave R-02 unmeasured · 6 set configs/targets.yaml after G0.5 ·
7 optionally anchor the `Edit(./data/**)` deny glob in `.claude/settings.json` so the package map's `data` name can
be restored (ADR-0004)

### Before the first release candidate, in this order
1. `ACIS_ALLOW_SEALED_FETCH=1 uv run acis fetch --sealed` — populates the sealed area with the held-out labels
   **and** the corpus/queries, and warms the datasets cache the offline run reads.
2. `uv run python -m acis.eval.final --smoke` — proves the sealed cache loads the task with the network off.
   **Do not skip this.** It is the one part of the official path that could not be executed here (no network, and
   the labels are not ours to fetch), and its failure mode is a run that dies at data loading, before a single
   vector is encoded. Minutes now, a wasted held-out touch otherwise.
3. `make rc-official RC=RC0 MODE=B` — the run itself, in your own terminal, uninterrupted.

## Gate register
| Gate | Status | Ledger run | Decision / frozen default |
|---|---|---|---|
| G0.0 hook canary | mechanism verified, owner sign-off pending | – | docs/PHASE0_REPORT.md §G0.0 |
| G0.1 mteb toy contract | **PASS** | – | 44 contract tests on mteb 2.21.0 / 2.12.30 / 2.5.1 |
| G0.2 data audit | **PASS** | – | runs/dataset_audit.json; matches R-01; E-LONG not triggered |
| G0.3 hardware + throughput | **PASS (dev host)** | – | runs/hardware.json: T-min, 157.6 GFLOP/s, no GPU |
| G0.4 model radar + pin | deferred to Phase 2 (ADR-0005, proposed) | – | feeds G-M; needs weights + owner decision |
| G0.5 zero-shot, all 5,000 dev queries | deferred to Phase 2 (ADR-0005, proposed) | – | feeds G-M; multi-hour per candidate |
| G0.6 guard tests + physical seal | **PASS** | – | 194 security tests; seal clean; detector matches path components |
| P1-B0 metric parity | **PASS** | – | equal to mteb/pytrec_eval at 1e-9, incl. graded/multi-relevant |
| P1-B1 BM25 parity | **PASS** | `dev-8717922a9ca4`, `dev-ee395f07e048` | 100 % top-10 identical, 5,000 queries |
| G-M encoder | pending | – | seed Qwen3-Embedding-0.6B; smallest within 1.0 pt (3.0 pt if best cold pass > 2 h) |
| G1 prep, per route | pending | – | statement_like: T1+V0, 1024/1024; generic: T3+V0 |
| G2 lexical + bridge | pending | – | dense-only |
| G3 adaptation (OOF) | pending | – | frozen base |
| G-OOD arbitrary-query robustness | pending | – | reject the change |
| G4 PRF | pending | – | off |
| G5 ranker | pending | – | weighted fusion → dense-only |
| G-AB Mode A vs B (OOF) | pending | – | A iff Δ ≥ +0.5 pt (CI>0), else B |
| G6 numeric profile | pending | – | cpu-fp32 |
| G7 invariants | pending | – | must pass |
| PF preflight (judge simulation) | pending | – | must pass before release |
| EV evidence pack (N1–N8) | pending | – | unproven claims are deleted |

## Open items (docs/spec/01 §3)
O-03 hardware/wall-clock · O-04 organizer mteb version (compat matrix now covers 2.21.0/2.12.30/2.5.1) ·
O-05 CSV schema + MRR cutoff · O-06 fine-tuning on APPS-train allowed? · O-07 P1/Bonus data format ·
O-09 GPU access · Q5 non-permissive licences
Resolved by FAQ: O-01 resource use is scored (weights unknown) · O-02 any method allowed · O-08 queries "similar to
the dataset" (treated as arbitrary) · JS out of scope
New in Phase 1: the dataset carries **no alternate solutions**, so D7's "train on a different solution than the
corpus copy" cannot be executed from it (docs/PHASE0_REPORT.md §G0.2) — a Phase 3 decision.

## Built and verified (docs/PHASE0_REPORT.md, docs/PHASE1_REPORT.md)
Toolchain lock (CPU-only torch) · `acis.core/sec/obs/appsdata/cli` · allow-listed fetch + physical seal ·
`acis doctor` · `acis.eval` (metrics, splits + lock, decontamination, bootstrap, hash-chained ledger, guard, dev
task, ladder, run files, verify-submission, official pipeline, `final`) · `acis.engine` with the **frozen**
`SearchEngine` interface · `acis.prep/lexical/rank/embed` · `acis.mteb_adapter` (Modes A and B) · `acis.robust_hook`
over a 256-document fixture · 549 tests + 3 pinned `xfail` guard-gap rows.

## Carry-forward into Phase 2 (not gate blockers, recorded so they are not lost)
- **The sealed-cache repair has no end-to-end proof.** Its paths are now unit-tested offline
  (`tests/unit/test_sealed_paths.py`), but the real fetch-and-load was impossible here. Treat the `--smoke` step
  above as a **precondition for RC0**, not a nicety.
- **Seal detection now also covers the datasets-cache layout** (`<config>/0.0.0/<hash>/<dataset>-<split>.arrow`),
  which is the layout the sealed warm step creates. It is still defence-in-depth, not a guarantee: the boundary
  remains physical (D19).
- **Gate rows are refused from a dirty tree** (dev rows are not — exploration happens on a dirty tree). The nine
  existing rows predate that rule and all carry `dirty: true`; they are chain-intact and the split lock re-derives,
  so they stand as harness evidence, but re-measure anything a Phase 2 gate wants to quote.
- **B1 parity was measured stemmer-free on both sides** because PyStemmer is not installed, while the configs
  declare `lexical.stemmer: english`. Fine as a harness check; decide before BM25 is used for anything more
  (install PyStemmer and re-measure, or declare stemmer-free by choice).
- **`configs/official.yaml` is a Phase 5 deliverable**, so `make rc-official`, `make reproduce` and
  `make reproduce-cache` cannot run until then. The code paths behind them are exercised on a synthetic task.

## Next actions
1. Owner: canary sign-off, the deferral decision for G0.4/G0.5, and the compute/deadline rows.
2. Then `/phase-gate 0` and `/phase-gate 1` write the PASS rows here (this file is not hand-edited on PASS).
3. Phase 2 (dense engine, bake-off, RC0) and Track B1 may then start; the interface they build against is frozen.
