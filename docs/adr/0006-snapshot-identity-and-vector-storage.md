# ADR-0006 — snapshot identity, vector storage and the lexical index (Track B1)

Status: **accepted 2026-09-23** (implementation decision inside Track B1; no owner action required, recorded
because `docs/spec/04` §1 says otherwise and CLAUDE.md §6 requires an ADR for every deviation).

## Context

`docs/spec/04` §1 specifies three things this implementation does differently:

1. `snapshot_id = sha256(source_id ‖ tree_hash ‖ build_config_hash)`;
2. vectors live in shared, append-only segments under `cas/segments/<model_fp>/<profile>/seg-<id>/`;
3. the BM25 index is built during the snapshot build and stored under `snapshots/<id>/lexical/`.

## Decision

**1. `snapshot_id` is `sha256(tree_hash ‖ build_config_hash)`. The source is manifest provenance, not identity.**

Including the source defeats the deduplication D11 promises. The same corpus ingested from a JSONL file and from
a directory would be two snapshots of identical content; re-ingesting one version of a history under a narrower
source spec would not recognise the version already built; and an incremental build could not be compared against
a from-scratch build, which is a standing acceptance test (spec 04 §5). The source id is still written into the
manifest, where it answers "where did this come from" without deciding "is this the same content".

**2. Vectors are a per-snapshot `vectors.npy` beside the manifest, not shared segments.**

The expensive resource is *embedding*, not disk, and embedding is already deduplicated by content: the vector
cache (`acis.embed.cache`) is keyed on model fingerprint, numeric profile, prep hash and text, so a body that has
been embedded once is never embedded again — across versions, across repositories, across runs. Segments would
additionally deduplicate the *bytes on disk*: about 36 MB per 8,765-unit snapshot at 1,024 dimensions.

What segments would cost is a per-query gather of rows from several memory-mapped files into one contiguous
matrix, or an index that makes the matmul non-contiguous. The retrieval core is the thing spec 02 §7 puts a p95
target on, and a per-snapshot matrix is exactly the shape the matmul wants.

So: the P1 *time* target — the measured, scored one — is met by the vector cache, and the disk cost is accepted.
If a corpus with many versions makes that cost real, segments become a storage optimisation with no behavioural
change, and this ADR is superseded rather than rewritten.

**3. The lexical index is rebuilt when a snapshot is opened, and the opened snapshot is cached per process.**

`bm25s` can persist an index, but rebuilding one from bodies already in the CAS takes seconds at demo scale and
removes a second on-disk format to validate, version and recover. Opening is therefore slightly slower and
building slightly faster; a process that serves many queries pays it once. Per-snapshot term statistics are
unchanged, which is what matters for correctness (no cross-version leakage, exact df/avgdl).

## Consequences

* Identical content built twice is the same snapshot, whatever it was read from — so "incremental ≡ from-scratch"
  is testable as an equality of snapshot ids rather than as an approximation.
* Disk grows linearly with versions for the dense channel. Measured and reported rather than hidden; the commit
  stream demo is the place it would first be visible.
* A first query against a freshly opened snapshot pays the lexical build. If that shows up in the B3 latency
  numbers on the reference host, persisting the index is the fix, and it is a local change behind `_load_snapshot`.
* `docs/spec/04` §1 and this ADR disagree; this ADR is what is shipped, and spec 04 should be read with it.
