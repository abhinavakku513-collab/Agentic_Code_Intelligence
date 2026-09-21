# `acis.sec` — contract

Spec: `docs/spec/05-agent-security-failure.md` §2; rules: `.claude/rules/security.md`.

**Responsibility.** The single `safe_path()` used by every filesystem access, untrusted-path normalisation, size and
depth limits, and (from Track B1) the parse-only sandbox workers.

| # | Guarantee | Test |
|---|---|---|
| S-1 | `safe_path(root, candidate)` never returns a path outside the real `root` (traversal, absolute paths, drive letters, `~`, NUL and control bytes are rejected) | `tests/security/test_safe_path.py` |
| S-2 | Symlinked components between the root and the target are rejected, never followed | `tests/security/test_safe_path.py` |
| S-3 | Limits raise `ResourceLimit`, never truncate silently | `tests/security/test_safe_path.py` |
| S-4 | Untrusted content is parsed, never imported, executed, compiled or `eval`'d (INV-5) | `tests/security/test_no_exec.py` |

**Non-goals.** No network, no archive extraction to disk (archives stream into the CAS in Track B1).
