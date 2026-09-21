# Kit changelog

## v1.1 — 2026-09-20 (ADR-0001…0003; source: `docs/reference/ACIS_Kit_Audit_and_Winning_Plan.md` + patch)
- **Enforcement**: hook commands switched to the string form; `/canary` (G0.0) proves the hooks execute; guard v1.1 (11 reproducible bypasses/false positives fixed, 40 + 3 pinned gap rows in `tests/security`) plus the physical-seal directory `~/.acis-sealed`; `ask` rules on every TEST-touching command; official runs are owner-launched.
- **Seal**: TEST labels live outside the working tree; dev loads `split="train"` only; the official run alone gets its own `HF_HOME` (D19). Wording changed from "provably sealed" to "outside the tree; hooks catch accidents".
- **Plan**: adaptation moved before hybrid/LTR; RC0 = best frozen dense, Mode B (was a BM25 baseline); Track B (P1/Bonus/demo) runs in parallel; phases 0–5, B1–B4, 6.
- **Statistics**: gates on all 5,000 TRAIN queries (non-fit) or K-fold OOF (trained); DEV-H is a counted once-per-milestone confirmation (simulation: +0.75 pt accepted 87 % of the time at n = 5,000 vs 34 % at n = 1,000 [E]).
- **Arbitrary queries**: INV-15, per-route instruction, routing v1.1, generic extractors with group dropout, α-interpolated adaptation, gate G-OOD, blind-query protocol, claim N8; NL version intent is a hint.
- **Ops**: `scripts/run_detached.sh` / `job_status.sh`, `docs/GPU_HANDOFF.md`, `docs/TRIAGE.md` (floors F0–F4, cut order), `training-engineer` and `demo-engineer` subagents, `/gpu-handoff` skill, update-time target and commit-stream demo.
- **Spec fixes**: `min(top_k, N)` (INV-10); TEST budget 6 = 5 + post-hoc clean-pool; G0.6 defined; D1 on a held-out fold; config paths from the repo root; encoder rule gains a 3.0-pt tolerance when the best candidate's cold pass exceeds 2 h.
- **Scope**: PPT and demo video are owner-made (`docs/submission/OWNER_HANDOFF.md`); `/evidence-pack` and `/preflight` no longer cover PPT items.
- **Density**: CLAUDE.md trimmed to ≈ 4k tokens (details in specs 02–10); ADR-0001 dependency allowlist; Phase-0 seeds (`pyproject.toml`, `Makefile`, `tests/conftest.py`, `configs/*`, `src/acis/{__init__,cli}.py`); `docs/reference/` now holds the audit, the earlier blueprint/skeleton and the third-party-score sources.

## v1.0 — 2026-09-20
Initial kit: CLAUDE.md, AGENTS.md, specs 01–08, 11 subagents, 11 skills, 6 rules, hooks, settings.
