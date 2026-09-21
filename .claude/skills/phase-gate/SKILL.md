---
name: phase-gate
description: Close an ACIS phase (0–6 or B1–B4) — run checks, the independent phase-gatekeeper (plus integrity/mteb/security/perf auditors where relevant), then update docs/STATUS.md and tag on PASS.
argument-hint: [0-6|B1-B4]
disable-model-invocation: true
---
Phase $ARGUMENTS.
1. Run `make lint typecheck test-fast`; stop on failure.
2. Delegate to `phase-gatekeeper` (phase $ARGUMENTS). In parallel where relevant: `eval-integrity-auditor` (phases 1–5), `mteb-guardian` (1, 2, 5), `security-reviewer` (B1, B3, 6), `perf-profiler` (2, B3, 6), `code-reviewer` (all).
3. Any FAIL or High finding → list the minimal fixes and stop; do not touch `docs/STATUS.md`.
4. All PASS → update `docs/STATUS.md` (phase_state done, last_gate, gate-register rows with ledger ids, next phase, TEST touches), commit `chore(status): phase $ARGUMENTS gate PASS`, tag `phase-$ARGUMENTS-done`.
