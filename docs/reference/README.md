# docs/reference — non-authoritative background
Precedence: docs/official > CLAUDE.md > docs/spec > .claude/rules > docs/reference. On any conflict follow CLAUDE.md and record the difference in an ADR. Read these only when an ADR or a gate needs the evidence.
| File | What it is | Caution |
|---|---|---|
| `ACIS_Kit_Audit_and_Winning_Plan.md` | the adversarial audit that produced kit v1.1 (ADR-0002/0003) | findings are integrated; keep for evidence tags and the guard/gate-power appendices |
| `kit_v1_1_patch_notes.md` | how the v1.1 patch was built and the hook canary | superseded by `docs/CANARY.md` |
| `ACI_Architecture_Blueprint.md` | an earlier blueprint with executed mteb checks (dispatch, ties) | contains superseded choices: cross-encoder gating, JS/tree-sitter, call graphs, an "epsilon"-style tie fix, a demo-video build task |
| `aci_mteb_adapter_skeleton.py` | an earlier adapter skeleton (executed on three mteb versions) | its `strictly_decreasing` squashing is superseded by rank-derived scores (D9); reuse only the `ModelMeta` helper idea and structure |
| `third_party_scores.md` | sources and verification status of every third-party number used for calibration | all rows unverified by us until G0.4/G0.5 |
