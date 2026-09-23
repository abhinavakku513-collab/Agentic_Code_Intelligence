# Track B1 report — the snapshot store and P1 (retrieval across versions)

Scope: `docs/spec/07` Track B1, `docs/spec/04` §1–§5. Track B is parallel from Phase 2 and starts once the Phase 1
gate freezes `AcisEngine`, which it did; Phase 3 cannot start (it needs a chosen encoder and a GPU), so this is
the next phase that could. It needs no model weights at all. The held-out touch counter is **0 of 6**.

## What P1 asks for, and where it is answered

| Requirement (`docs/spec/04` §5) | Status | Evidence |
|---|---|---|
| Version isolation — a query pinned to v2 never returns v3's content | **PASS** — asserted over randomly generated histories, not only written examples | `tests/property/test_version_isolation.py`, `tests/integration/test_p1_versions.py` |
| Identical-tree dedup | **PASS** — identical content is the same snapshot id, so a history returning to an earlier state costs nothing | `tests/property/test_version_isolation.py::test_a_snapshot_is_reused_whenever_the_tree_repeats` |
| `kill -9` at every build step, then recovery | **PASS** — a real child process is SIGKILLed at each of six named build points; the store then serves either the old snapshot or the new one, never a mixture | `tests/chaos/test_kill_during_build.py` |
| Concurrent search during activation (no torn reads) | **PASS** — a reader thread searches continuously while builds and rollbacks race it | `tests/property/test_version_isolation.py::test_a_search_running_during_an_activation_never_sees_a_torn_result` |
| Rollback correctness | **PASS** — instant, and the snapshot it rolls back *from* is untouched | `tests/integration/test_p1_versions.py` |
| Incremental ≡ from-scratch | **PASS** — same snapshot id, same rankings, and copied vector rows are bit-identical to re-embedded ones | `tests/integration/test_p1_versions.py` |
| Determinism of `snapshot_id` | **PASS** — content plus build config, independent of ingestion order and of which source it was read from | `tests/unit/test_store_snapshots.py` |
| Measured update times, published with hardware | **PASS** — see below | `[ledger:bench-9153e49050de]`, `runs/p1_update.json` |
| Apps-Evolve generator | **PASS** — nine deterministic mutation operators with ground truth by construction | `tests/unit/test_appsevolve.py` |

## Update times `[ledger:bench-9153e49050de]`

Dev host, `cpu-fp32`, 2,000 units, 5 repeats, stand-in encoder (`acis/hashing-4096`, not submission-capable),
peak RSS 419 MB, first build 9.2 s. The clock measures what the requirement actually asks about — from "here is a
new version" to "a query against that version returns the new content" — because a build that finishes quickly
and then takes twenty seconds to open has made nothing searchable.

| Change | p50 | p95 | Target |
|---|---|---|---|
| 1 unit | 1.91 s | 1.95 s | ≤ 30 s |
| 10 units | 1.89 s | 1.95 s | ≤ 30 s |
| 100 units | 2.35 s | 2.37 s | — |
| full rebuild (all 2,000) | 10.93 s | 11.27 s | — |

The full-rebuild column is the comparison that matters: if a ten-unit change cost what a rebuild costs, the
incremental path would be decoration. Read the absolute numbers with the encoder in mind — with the stand-in this
measures the **store**, and the embedding cost that dominates a real run is absent. The incremental path is what
keeps that cost proportional to the change: a unit whose body already exists in the superseded snapshot has its
vector row copied, so ten changed units cost ten embeddings whatever the encoder is.

## How it works, in four sentences

A snapshot is never modified after it exists. Builds happen under `.tmp-<token>/` and become visible by a rename;
`VALID` is written last and carries a hash of the manifest, so a snapshot missing it or disagreeing with it is
refused rather than served (INV-9). Activation is an atomic ref rename that keeps `PREV`, which is why rollback is
instant and costs nothing. Every request resolves its version selector **once** and is answered from that snapshot
alone (INV-2), so results cannot mix versions even when a build activates halfway through a query.

## Sources are treated as hostile

`acis ingest` accepts JSONL, directory sequences, ZIPs and git repositories, and the interesting behaviour of each
is what it refuses: an archive member naming a path outside the archive or one that is a symlink; a member whose
*declared* size exceeds the cap, checked before anything is decompressed; a symlink out of a directory tree,
whether it is a file or a whole version directory. Git is read with `ls-tree`/`cat-file` and no checkout, with
hooks, protocols, prompts and submodules disabled and a scrubbed environment — a test installs real hooks and
asserts none of them ran. Content is parsed with `ast` and never imported, executed, compiled for execution or
`eval`'d (INV-5); a file that will not parse is still ingested, flagged, and stays retrievable.

## Defects found and fixed

Five, all of which would have passed a demo:

* **`latest` read the newest built version rather than `refs/ACTIVE`**, so a rollback changed nothing that anyone
  searching could see.
* **`units_new` came from the encoder's own counter**, which the stand-in does not keep, so every build reported
  all of its units as new. It now comes from the content store — bodies never seen before — which is true for any
  encoder.
* **Activation upserted the snapshot row with `n_units=0`**, so every activated snapshot claimed to be empty.
* **`as_of` compared a float timestamp against microsecond-truncated ISO input**, so asking for the exact instant
  a version was created answered "the version before it" for some values and not others.
* **Builds embedded every unit** and leaned on the vector cache to make unchanged ones cheap — a full rebuild that
  happens to be fast rather than an incremental build.

## Deviations from the spec

`ADR-0006` records three, with the reasoning: `snapshot_id` drops the source (including it defeats the dedup D11
promises), vectors are a per-snapshot matrix rather than shared CAS segments (the expensive resource is embedding,
already deduplicated by content, and the retrieval core wants a contiguous matrix), and the lexical index is
rebuilt on open rather than persisted (seconds from bodies already in the CAS, and one less on-disk format to
validate, version and recover).

## What is deliberately absent

Lineage: the alignment cascade, the lineage store and evolution-aware ranking are Track B2, and
`retrieve_evolution` still raises `NotReady` naming it. `compare_versions` therefore reports the *certain* half of
a difference — added, removed and changed by content hash — and does not guess at renames, moves or replacements,
because a confident answer nobody can check is worse than an honest gap. The Apps-Evolve generator, which B2 needs
to measure those, is built and tested here.

Also absent: a background build queue (the spec's answer for a long history; a synchronous `JobHandle` that is
already `done` keeps the API shape honest at demo scale), and garbage collection of retired snapshots.
