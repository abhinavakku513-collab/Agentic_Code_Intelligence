# Installing the ACIS Claude Code kit (v1.1)

1. Unzip into your repository root (keep the dot-folders `.claude/` and `.gitignore`); `git init` if new; commit the kit as the first commit.
2. If unzipping lost the executable bit, run `chmod +x scripts/*.sh`. Install `uv`; make sure `python3` is on PATH for the shell Claude Code uses (hooks are stdlib-only Python; Linux, macOS or WSL2 recommended).
3. Use a current Claude Code release: the kit was checked against the 2026-09 docs (skills, subagents, path-scoped `.claude/rules/`, string-form hooks). Start sessions in plan mode for phase starts and gate work (`claude --permission-mode plan`), with the strongest available model for `/phase-start`, gates and adapter design and a faster model for tests and boilerplate.
4. **Canary first** (`docs/CANARY.md`): create `data/sealed/canary.txt` in your own shell, run `/canary`, expect three blocks, delete the file, record `hook_canary: passed` in `docs/STATUS.md`. Repeat after every Claude Code upgrade. A green `/hooks` listing does not prove the hooks run.
5. `uv run pytest tests/security tests/contract tests/robustness -q` (expect: guard tests pass, three pinned `xfail` gap rows, dispatch tests skip until mteb is installed, robustness engine tests skip until Phase 1).
6. Check `/memory` (CLAUDE.md + AGENTS.md), `/hooks`, `/agents` (13 subagents), `/permissions`, and type `/` for the 13 skills.
7. Fill the owner items at the top of `docs/STATUS.md` (deadline, compute), send `docs/official/ORGANIZER_QA.md`, then `/status` and `/phase-start 0`.
Precedence: docs/official > CLAUDE.md > docs/spec > .claude/rules > docs/reference. PPT and demo video are yours (`docs/submission/OWNER_HANDOFF.md`).
