# `acis.ingest` — contract

Spec: `docs/spec/04-storage-p1-bonus.md` §2; security: `docs/spec/05-agent-security-failure.md` §2 and
`.claude/rules/security.md`.

**Responsibility.** Turning somebody else's layout — a JSONL file, a directory sequence, a ZIP, a git repository —
into `VersionUnits(label, units)` in history order, without ever trusting it. Everything here treats its input as
hostile, because in the threat model it is.

| # | Guarantee | Invariant | Test |
|---|---|---|---|
| I-1 | Untrusted content is **parsed** with `ast` and never imported, executed, compiled for execution or `eval`'d; a file that will not parse is still ingested, flagged `parse_ok=false` | INV-5 | `tests/security/test_ingest_hardening.py` |
| I-2 | Archives are streamed, never extracted; a member naming a path outside the archive, or one that is a symlink, is refused | INV-5 | `tests/security/test_ingest_hardening.py` |
| I-3 | A member's **declared** size is checked before it is decompressed, so a bomb is refused rather than materialised; unit, byte and count limits raise `ResourceLimit` instead of truncating | — | `tests/security/test_ingest_hardening.py`, `tests/unit/test_ingest_sources.py` |
| I-4 | Symlinks are never followed out of a directory tree, whether the link is a file or a version directory | — | `tests/security/test_ingest_hardening.py` |
| I-5 | Git is read with `ls-tree`/`cat-file` and no checkout, with hooks, symlinks, submodules, prompts and network protocols disabled and a scrubbed environment; nothing is ever written to the repository | INV-5 | `tests/integration/test_git_source.py` |
| I-6 | Versions are ordered naturally (`v9` before `v10`) and git history oldest-first — the order P1 replays them in | — | `tests/unit/test_ingest_sources.py` |
| I-7 | Unit keys and version labels are validated: a key is never a path, a label is never `../`, and both are bounded | — | `tests/security/test_ingest_hardening.py` |
| I-8 | Source identity covers the filters that changed what was read, so an ingest with different extensions is a different corpus | INV-2 | `tests/unit/test_ingest_sources.py` |

**Non-goals.** No ranking, no embedding, no snapshot writing — a source yields units and stops. Persisting them is
`acis.store`; deciding what they mean is the engine.
