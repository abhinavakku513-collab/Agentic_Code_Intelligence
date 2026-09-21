---
name: phase-start
description: Start an ACIS phase (0–6 or B1–B4) — read the phase spec, propose a plan for approval, create the branch, and have test-engineer write failing tests first.
argument-hint: [0-6|B1-B4]
disable-model-invocation: true
---
Phase $ARGUMENTS.
1. Read `docs/STATUS.md`; refuse to continue if `hook_canary` is not `passed`, or if the predecessor gate is not PASS (Phase 1 needs 0; 2 needs 1; 3 needs 2; 4 needs 3; 5 needs 4; 6 needs 5 and Track B; **B1–B4 need the Phase 1 gate**, B2 needs B1, B4 needs B3). Say exactly which condition fails.
2. Read `CLAUDE.md` §2–§5, `docs/spec/07-phases-dod-audit.md` (that phase), `docs/TRIAGE.md`, and every spec file the phase cites (`docs/spec/01`–`10`).
3. Enter plan mode and present: files/modules to create, interfaces, tests, acceptance criteria, jobs that need `scripts/run_detached.sh` or the owner's terminal, GPU needs, and the "not yet" list. Wait for owner approval.
4. On approval: `git switch -c phase/$ARGUMENTS-<short-name>`; set `phase_state: in-progress` in `docs/STATUS.md`; delegate failing tests to `test-engineer`; implement in small Conventional Commits; run `code-reviewer` after each non-trivial change. Track A and Track B may run in parallel worktrees; do not change the frozen `AcisEngine` interface without an ADR.
