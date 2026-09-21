# 04 · Storage, P1 (versions), Bonus (evolution)

Authority: subordinate to `CLAUDE.md` (D10–D12, INV-1/2/9). FAQ: the data is APPS/Python, so the **primary unit is a whole snippet** with an optional external key; git and function-level units are secondary sources.

## 1. Storage technology (D10)
Files + stdlib SQLite (WAL, `synchronous=FULL`) + memory-mapped `.npy` vectors + bm25s + NumPy exact search. No vector DB, graph DB, Elasticsearch, Redis, broker. Rationale: 8,765×1,024 fp32 ≈ 36 MB; exact search is deterministic and lossless; a service adds latency, ops and attack surface. HNSW only above ≈250k vectors per snapshot as a per-segment scale profile with recall@100 ≥ 0.99 vs exact.

```
$ACIS_HOME/
├─ catalog.sqlite                    repos, versions, refs, jobs, journal
├─ cas/                              append-only, immutable, sharded by hash prefix
│  ├─ blobs/ab/cd…                   snippet/file text (sha256, zstd)
│  ├─ tok/<tokenizer_ver>/…          token arrays per body_hash
│  ├─ feat/<feature_ver>/…           features per body_hash
│  └─ segments/<model_fp>/<profile>/seg-<id>/{vectors.f16.npy, embed_keys.bin, meta.json}
├─ repos/<repo_id>/
│  ├─ snapshots/<snapshot_id>/       IMMUTABLE after activation
│  │  ├─ manifest.json               ids, counts, config_hash, model_fp, parent ids, file hashes
│  │  ├─ units.sqlite                unit_id, key, body_hash, embed_key, seg_id, row, meta
│  │  ├─ lexical/                    bm25s index (per-snapshot df/avgdl)
│  │  ├─ lineage.sqlite              links to parent snapshot(s)
│  │  └─ VALID                       written last: sha256(manifest ‖ sizes ‖ hashes)
│  ├─ .tmp-<uuid>/                   in-flight build, deleted by crash recovery
│  └─ refs/{ACTIVE, PREV, tags/…}    atomic pointers
├─ cache/{qemb,results}/             keys contain snapshot_id + config_hash
└─ runs/<run_id>/…                   evaluation manifests and artifacts
```
Keys: `body_hash=sha256(normalised text)`; `embed_key=(body_hash, prep_hash)`; vector key adds `model_fp` + profile; `tree_hash=sha256(sorted[(key_or_path, body_hash)])`; `snapshot_id=sha256(source_id ‖ tree_hash ‖ build_config_hash)[:32]` (identical content ⇒ identical snapshot; each commit/label is a ref to it).

## 2. Sources → one internal model
`Unit(unit_id, key, body_hash, text, version, meta)`. Adapters (`ingest`): **JsonlSource** (`{id|snippet_id, version, text}` per line — the likely P1/Bonus shape) · **DirSequenceSource** / **ArchiveSequenceSource** (`v1/`, `v2/`, ZIP/TAR per version; order by manifest or natural sort; ZIPs streamed, never extracted) · **GitSource** (`git ls-tree -r -z --long <commit>` + `cat-file --batch`; no checkout, hooks off, no submodule/LFS; files or `ast` functions become units; `git diff -M -C` is a rename *hint* only). Non-git sources form a linear chain. Filters (extensions, size caps) are part of the config hash. Python files are parsed with `ast`; failure → tolerant tokeniser → raw text (`parse_ok=false`, never an error).

## 3. Snapshot lifecycle and build (P1)
States: `BUILDING → READY_LEX → READY → ACTIVE → RETIRED → DELETED`; `FAILED` from any build state. Build algorithm:
1. List the version's units; classify each by `body_hash` against the CAS (known ⇒ reuse tokens/features/vector).
2. Read/parse unknown content in sandboxed workers (bounded), write tokens/features to CAS.
3. Embed only `embed_key`s absent from the segment index — length-sorted, token-budget batches — append **one** new immutable segment.
4. Write `units.sqlite`; build BM25 from cached token arrays (exact per-snapshot statistics; seconds).
5. Lineage pass against each parent (section 6).
6. **Validate**: units = vector rows = BM25 docs; `PRAGMA integrity_check`; recompute 64 random vectors (cosine ≥ 0.9999); SHA-256 of 256 random blobs; manifest hash; self-retrieval probe (1 % of units return themselves at rank 1 via dense and BM25, excluding exact duplicates); write `VALID`.
7. **Activate**: fsync files + directory → rename `.tmp-<uuid>` → `snapshots/<id>` → journal entry → atomic `os.replace` of `refs/ACTIVE` (old ⇒ `PREV`).
Cost model [E, measure]: `T_build ≈ list + parse(new) + embed(new) + bm25(all) + lineage + validate`; only embedding is heavy (≈ new_units × avg_tokens × per-token FLOPs ÷ sustained GFLOPS). Unchanged content ≈ free; a mass rewrite is honestly slower and reported. Policy for "commits every minute": eagerly build branch heads/tags (coalescing bursts to the newest head); history builds lazily or in a background queue; `acis versions` shows coverage.
**Partial snapshots**: while vectors are pending, `READY_LEX` serves lexical-only for that version; default search returns the newest complete snapshot; `allow_partial=true` queries the in-progress one and the response carries `snapshot.complete=false, missing_channels=["dense"], units_pending=N`. Responses never mix versions (INV-2).
**Concurrency/recovery**: one writer per repo (advisory lock); readers hold leases on snapshot dirs; GC never deletes a leased snapshot (grace period, liveness < 40 % segments compacted offline). Startup: delete `.tmp-*`; verify ACTIVE's `VALID` else fall back to PREV and log an incident; replay the journal. Rollback = re-point ACTIVE (instant; snapshots immutable). Cheap check on open (sizes + manifest hash); full SHA-256 via `acis verify` and after unclean shutdown.

## 4. Version operations (semantics)
Selectors: `latest` · `version:<label>` · `snapshot:<id>` · `commit:<sha|prefix>` · `tag:<name>` · `as_of:<ISO-8601>` (newest version at or before the instant on the first-parent/linear chain) · `all` · `range:<a>..<b>`. Resolved once per request; unresolvable ⇒ `NotFound`, never a guess; not built ⇒ `IndexRequired`; ambiguous `as_of` ⇒ `VersionConflict` with candidates. `search_version` = `search` with a pinned selector; `update_version(source_delta, expected_active)` builds a new snapshot (409 on stale `expected_active`); `compare_versions(a, b)` returns unit-level added/removed/modified/renamed/moved with evidence and certainty (lineage tables if present, else hash set-difference) and, with a query, side-by-side rankings.

## 5. P1 acceptance (automated)
Version isolation (property test over random histories: v2 query never returns v3 content) · identical-tree dedup · `kill -9` at every build step then recovery (≥ 1,000 iterations) · concurrent search during activation (no torn reads) · rollback correctness · **incremental ≡ from-scratch** (same rankings, same units modulo ids) · determinism of `snapshot_id` · measured update times (1/10/100-unit changes vs full rebuild) published with hardware · **Apps-Evolve** generator (APPS solutions → v2…v5 via deterministic mutation operators: reformat, comment edit, identifier rename, helper reorder, constant/branch edit, replace with another solution to the same problem, delete, add, key change) gives ground truth by construction; metrics: version-correct NDCG@10, leak rate = 0, stale-answer rate = 0.

## 6. Bonus — evolution-aware retrieval (D12)
**Problem**: across versions a unit exists as many near-identical revisions; a flat all-version list is flooded by duplicates and hides which revision is best. **Design**: (a) recognise revisions of one lineage, (b) rank lineages, (c) show the best-matching revision and the evolution, (d) never lose genuinely different implementations.
**Alignment cascade** (between a snapshot and its parent(s); cheapest, most certain first; every relation stores an evidence vector and confidence, and only links above a confidence floor join a lineage):
| Stage | Rule | Relation | Certainty |
|---|---|---|---|
| S0 | same key, same `body_hash` | identical | certain |
| S1 | same key, similarity ≥ θ_mod | modified | high |
| S2 | same/similar body under a different key/path | moved | high |
| S3 | name-insensitive normalised body equal | renamed (or renamed+moved) | medium–high |
| S4 | leftover removed×added pairs, blocked by shared tokens/dense top-5, Hungarian on `w₁·token_sim + w₂·structure_sim + w₃·dense_cos` ≥ θ_rep | replaced (inferred) | low–medium, always labelled |
| S5 | unmatched | added / removed; ambiguous ⇒ `unknown` | – |
Similarities reuse cached vectors and token arrays (no new embedding). θ values are [E], fitted on Apps-Evolve; `git diff -M -C` hints raise priors for S2/S3.
**Lineage store**: `lineage(lineage_id, snapshot_id, unit_id, parent_unit_id, relation, confidence, evidence_json)` per snapshot plus a repo-level index built by union-find (canonical id = hash of the earliest member); a unit may link to several parents (kept, flagged).
**Search (`versions=all`)**: one dense matrix over unique `body_hash`es plus a `present_in` bitmap, so identical revisions are one candidate with a span (first_seen…last_seen) → normal pipeline (spec 02) → group by lineage → `score(lineage)=max member score` (tie-break `prefer=latest|best`; optional consensus bonus, default 0, enabled only if it helps) → one result per lineage: best revision, `members[]` (version span, score, relation to predecessor, certainty), `timeline` (change points with diffs). `expand=false` returns heads; `flat=true` returns ungrouped revisions. Unlinked distinct implementations stay separate. Natural-language cues (`as_of`, `in v3`, `latest`, `what changed`) never filter: they come back as `interpreted_intent {as_of, confidence}` and the UI offers a one-click apply; scope comes only from explicit selectors, and “what changed” is the explicit `compare_versions` action (spec 10 §6, INV-15).
**Evaluation** (no official Bonus metric [U]): lineage pairwise F1, per-relation precision/recall, **wrong-merge rate ≤ 1 %**; Evolution-NDCG@10 (best revision 3, other members 1, unrelated 0); Duplicate-Rate@10 ≈ 0 in grouped mode; Best-Revision-Hit@1. **Acceptance**: grouped search beats flat all-version search on Evolution-NDCG@10 with paired CI lower bound > 0. Known limits (squashed histories, vendored copies, heavy refactors): lineages stay separate — safe, never falsely merged.
**Update-time target [E, measure]**: ≤ 10 changed units searchable within ≤ 30 s p95 on T-rec (embedding-dominated; lexical-first `READY_LEX` within ≈ 3 s) and one change per minute sustained without backlog; results are published with hardware and any revision needs an ADR. **Commit-stream demo**: `scripts/demo/commit_stream.py` replays versions at a fixed cadence with a live freshness gauge, units re-embedded, rollback and a `kill -9` mid-build with recovery (spec 09 §6).
