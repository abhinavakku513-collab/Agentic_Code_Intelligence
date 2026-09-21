---
name: gate
description: Execute a pre-declared ACIS gate procedure (G-M, G1–G7, G-AB, G-OOD) and write its decision, applying the frozen default when the accept rule fails. G0.x are run in Phase 0 by hand, not here.
argument-hint: [G-M|G1|G2|G3|G4|G5|G6|G7|G-AB|G-OOD]
disable-model-invocation: true
---
1. Read the gate row in `docs/spec/03-mteb-and-integrity.md` §7 (and `docs/spec/10` §5 for G-OOD); restate the accept rule, the decision set and the frozen default.
2. Delegate runs to `retrieval-experimenter`: dev only; paired bootstrap 10,000 resamples; practical threshold +0.5 pt with CI lower bound > 0; **decision set = all 5,000 TRAIN queries (non-fit components) or K-fold OOF (trained components)**; ties → the cheaper option (G-R); G-OOD for every query-side, adaptation, LTR or prep change.
3. Write `configs/gates/$ARGUMENTS.yaml` with `{run_id, decision, evidence, default_applied}` (the owner confirms this write). Never edit a decided gate file; supersede it via `/adr`.
4. Gate-register updates in `docs/STATUS.md` happen in `/phase-gate`, not here.
