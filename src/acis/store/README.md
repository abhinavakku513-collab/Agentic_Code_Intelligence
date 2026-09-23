# `acis.store` — contract

Spec: `docs/spec/04-storage-p1-bonus.md` §1 and §3. Rules: `.claude/rules/storage-versions.md`.

**Responsibility.** The P1 substrate: an append-only content store, the SQLite catalog (repositories, versions,
snapshots, jobs, journal), immutable snapshots with validate-then-activate, atomic refs, rollback and crash
recovery. It owns *durability*, not retrieval: nothing here ranks anything, and no query ever reaches it except
through the engine.

| # | Guarantee | Invariant | Test |
|---|---|---|---|
| T-1 | A body is stored under the hash of its bytes, written atomically, and never rewritten; identical content is one blob | INV-1 | `tests/unit/test_store_cas.py` |
| T-2 | A blob that cannot be read back as its own hash is reported as corruption, never returned | INV-1 | `tests/unit/test_store_cas.py` |
| T-3 | Repository ids, version labels and blob keys are validated before they become paths; nothing escapes `ACIS_HOME` | — | `tests/unit/test_store_cas.py` |
| T-4 | The catalog is WAL + `synchronous=FULL` + foreign keys, and refuses a store written by a newer schema | — | `tests/unit/test_store_cas.py` |
| T-5 | `snapshot_id` is a hash of content **and** build configuration: the same tree built twice is the same snapshot, and a different config is a different one | INV-2, D11 | `tests/unit/test_store_snapshots.py` |
| T-6 | Builds happen in `.tmp-<token>/`; `VALID` is written last and holds the manifest hash, so a snapshot that is missing it or disagrees with it is refused | INV-9 | `tests/unit/test_store_snapshots.py` |
| T-7 | Activation is an atomic ref rename keeping `PREV`, so rollback re-points rather than rebuilds and touches no snapshot | D11 | `tests/unit/test_store_snapshots.py` |
| T-8 | `expected_active` is a compare-and-swap: a stale writer loses with `VersionConflict` rather than overwriting | — | `tests/unit/test_store_snapshots.py` |
| T-9 | Recovery deletes unfinished builds, falls back to `PREV` when the active snapshot is damaged, and records the fallback as an incident rather than absorbing it | INV-7 | `tests/unit/test_store_snapshots.py` |
| T-10 | A real `kill -9` at **every** build step leaves a store that serves either the old snapshot or the new one, never a mixture; a build step without a crash point fails the suite | D11 | `tests/chaos/test_kill_during_build.py` |

**Non-goals.** No vector database, graph database, search server or broker (D10). No ranking, no embedding, no
lexical index construction — those belong to the engine, which builds them *into* a snapshot directory this
module has created and will seal.
