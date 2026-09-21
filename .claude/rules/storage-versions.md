---
paths:
  - "src/acis/store/**"
  - "src/acis/ingest/**"
  - "src/acis/lineage/**"
---
# Storage / P1 / Bonus guardrails — read `docs/spec/04-storage-p1-bonus.md` first
- Snapshots are immutable after `VALID`: build in `.tmp-<uuid>`, validate, fsync, rename, atomic ref switch. Nothing partial is ever visible (INV-9).
- The CAS is append-only; every cache key contains `snapshot_id` + `config_hash`; each request pins exactly one snapshot (INV-2).
- Only embedding is expensive-incremental; BM25 is rebuilt per snapshot from cached tokens (exact df/avgdl, no cross-version leakage).
- Lineage links are evidence-gated and store evidence + confidence; `unknown` beats a guessed merge; never merge without evidence.
- P1 update-time target: ≤ 10 changed units searchable ≤ 30 s p95 on T-rec [E]; lexical-first `READY_LEX` ≈ 3 s; the commit-stream replay (`scripts/demo/commit_stream.py`) must run without backlog at one change per minute.
- Every write path has a `kill -9` recovery test; incremental ≡ from-scratch is a standing differential test.
