---
name: security-sweep
description: Run the ACIS security and supply-chain sweep and summarise findings by severity.
allowed-tools: Read, Grep, Glob, Bash(make test-security), Bash(uv run bandit *), Bash(uv run semgrep *), Bash(uv run pip-audit *), Bash(gitleaks *), Bash(osv-scanner *)
---
Delegate to `security-reviewer`: bandit, semgrep (banned constructs), pip-audit/osv-scanner against `uv.lock`, gitleaks, and `make test-security` (hostile archives, traversal, symlinks, bombs, injection corpus, sandbox-kill). Report High/Medium/Low with file:line and a minimal fix for each. Zero open High findings are required for Phase 6; respect the cut list in `docs/TRIAGE.md` (S2 sandbox tier, signing, ≥ 1 h fuzzing and custom semgrep are cut-first).
