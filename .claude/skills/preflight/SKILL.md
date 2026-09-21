---
name: preflight
description: Judge-simulation pre-flight — verifies every Samsung criterion in a clean room (P0 JSON, reproducibility, CPU/GPU/model-size/runtime, unknown-query hands-on, P1 commit stream, Bonus, robustness, artifacts, docs truth) before release. PPT/video are owner-made and not checked.
disable-model-invocation: true
---
1. Preconditions: clean git tree; `make lint typecheck test test-security` green; an RC exists in the ledger (RC1 or later); Phase 5 and Track B B1–B3 done in `docs/STATUS.md` (B4 optional); `hook_canary: passed`.
2. Delegate to `judge-simulator` for checks C1–C11 and C13 in `docs/spec/08-novelty-evidence-submission.md` §4 (C12 = owner-made PPT/video, skipped). In parallel run `eval-integrity-auditor` and `mteb-guardian` on the release candidate. Include the blind-query protocol (`docs/spec/10` §7.4) exactly once, with queries written by someone who has seen neither the dev data nor the system.
3. Collect `docs/submission/preflight_report.md`. Any FAIL → list the minimal fixes, stop, and do not mark Phase 6 done. All PASS → record the report path in `docs/STATUS.md` (via `/phase-gate 6`).
4. Never load TEST labels; never start a new RC from here.
