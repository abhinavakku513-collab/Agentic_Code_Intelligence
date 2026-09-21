---
name: status
description: Report ACIS progress — current phase, gate register, owner actions pending, deadline, hook-canary state, TEST-touch budget, git state and the single next action. Use when the owner asks where things stand, at the start of a session, or after /compact.
allowed-tools: Read, Grep, Glob, Bash(git status *), Bash(git log *), Bash(git diff *)
---
1. Read `docs/STATUS.md` and the tail of `runs/ledger.jsonl` (read-only).
2. Run `git status -sb`; summarise uncommitted work by package.
3. Report: current phase/state (and Track B state); gates PASS/pending; **`hook_canary` (must be passed), deadline (must be declared), compute declared?**; TEST touches used vs 6; owner actions pending; open items; blockers.
4. Recommend exactly ONE next action (a slash command or a concrete task) and the spec file to read first.
Do not modify any file.
