# `acis.engine` — contract

Spec: `docs/spec/06-contracts-testing-ops.md` §1 and `docs/spec/02-p0-engine.md` §4. Rules: `.claude/rules/p0-engine.md`.

**Responsibility.** The one engine: snapshots, channels, routing, composition, the degradation ladder, and the batch
surface the mteb adapter uses. `protocol.SearchEngine` is **frozen at the end of Phase 1** — Track B builds against
it, and changing a signature needs an ADR.

| # | Guarantee | Invariant | Test |
|---|---|---|---|
| N-1 | Every `Hit.source` is re-read from the content store by `body_hash`; nothing is generated | INV-1 | `tests/unit/test_prep_and_engine.py` |
| N-2 | Results come only from the requested snapshot; the query-vector cache key carries `snapshot_id` and `config_hash` | INV-2 | `tests/property/test_invariants.py` |
| N-3 | A query's ranking never depends on the other queries in its batch | INV-3 | `tests/property/test_invariants.py`, P4 |
| N-4 | External ids are opaque; ties break on the **content hash**, and only exact duplicates reach the corpus ordinal | INV-4, D9 | `tests/property/test_invariants.py`, P5, P7 |
| N-5 | Same snapshot, config, profile and thread count ⇒ same ranking | INV-6 | `tests/robustness`, P6 |
| N-6 | Every fallback increments a counter and appears in `degradations`; strict mode raises instead | INV-7 | `tests/unit/test_prep_and_engine.py` |
| N-7 | Only VALID snapshots are searchable unless `allow_partial=True` | INV-9 | `tests/unit/test_prep_and_engine.py` |
| N-8 | `agent_calls == 0`: there is no agent on this path | INV-13 | `tests/unit/test_prep_and_engine.py` |
| N-9 | Arbitrary queries never crash: empty raises `InvalidInput`, over-long is head+tail truncated, hostile input returns a ranking | INV-15 | `tests/robustness/test_query_agnostic.py` |
| N-11 | The retrieval core ranks only what it returns: the top-`k` cut keeps every document tied with the k-th best, so it equals the full ranking document for document | INV-4, INV-10 | `tests/metamorphic/test_parity.py` (P8) |
| N-10 | The route reaches the encoder **and** the query-vector cache key, so two routes are two vectors of the same text | INV-15, INV-2 | `tests/unit/test_dense_wiring.py` |

**Phase boundaries.** Channel fusion is decided by gate G2 in Phase 4, so `mode="hybrid"` raises `NotReady` rather
than inventing an ungated ranking; routing v1.1 falls back to `generic` until the OOD bank exists (Phase 2+); Track B
methods (`ingest`, `index`, `update_version`, `compare_versions`, `retrieve_evolution`) carry their frozen signatures
and raise `NotReady`.
