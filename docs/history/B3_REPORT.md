# Track B3 report — API, CLI, UI and `make demo`

Scope: `docs/spec/07` Track B3, `docs/spec/09` §6 and R9. The held-out touch counter is **0 of 6**. No slides and
no video (owner-made). **Track B3 is not complete**: the latency targets of `docs/spec/02` §7 have not been
measured with the real encoder, and the reference host (T-rec) does not exist yet.

## Acceptance

| Acceptance (`docs/spec/07` B3) | Status | Evidence |
|---|---|---|
| API end-to-end tests | **PASS** — search, evolve, versions, diff, rollback, typed errors, health, metrics, no evaluation endpoint, no remote asset in the UI, and the P0 corpus searchable from the first request | `tests/integration/test_api.py` (19 tests) |
| `make demo` runs offline end to end | **PASS on this host** — each act run with `HF_HUB_OFFLINE=1`: act 1 (P0 search) 62 s; act 2 (commit stream, `kill -9` + recovery, rollback, Bonus grouping) 80 s; act 3 (the page) serves, answers `/healthz` and `/readyz`, and searches the APPS corpus | run on 2026-09-26; see the defects below |
| `make demo` on a clean machine | **PASS with the demo index** — a cache-less `ACIS_HOME` imported the pack in 2 min 19 s (98 vectors recomputed, worst cosine 1.000000) and then built the 8,765-unit snapshot in 23.6 s, with the same snapshot id as the warm machine | `tests/unit/test_demo_index.py`; the pack is a release asset (below) |
| `docs/spec/02` §7 targets measured on the reference host | **NOT MEASURED** — the only `make bench` row is the stand-in encoder `[ledger:bench-b3586309bd8e]`, and the dev host is T-min | waits for the G1 sweep to free the CPU; T-rec is an owner decision |

## Defects found by running the demo rather than its tests

| Commit | Defect | Consequence if shipped |
|---|---|---|
| 4f97b18 | `acis serve` never built the APPS snapshot; the API tests hid it by building one in their fixture | the page's default corpus answered "no snapshot has been built yet" to a judge's first free-text query |
| 4e3904b | no prebuilt demo index existed (spec 09 R9) | on a clean machine act 1 embeds the whole corpus first: about 2.6 h on this CPU |
| 110aa87 | Apps-Evolve's `rename` renamed imported modules and attributes | the demo showed `import heapq_v85`, a revision no refactor produces |
| 304a91a | ingest let a parsed file's own `SyntaxWarning`s print into our logs | noise from the data in the demo output |
| 34428c1 | vector-cache keys became file names unchecked, and a demo pack supplies its own keys | a crafted pack could write files outside the cache; keys must now be SHA-256 hex, in the cache itself |

## The prebuilt demo index (D17)

`acis demo-index export` writes the corpus's document vectors, their cache keys and a manifest as `.npy` + JSON
in a zip (25 MB for `gte-modernbert-base`, no pickle). `acis demo-index import` refuses an altered pack
(checksums), another model's or another preparation's pack (fingerprint and prep hash), and a forged pack with
honest checksums (1 % recomputed, cosine ≥ 0.9999). It caps member sizes before decompressing, and a second
import is a no-op. It fills only the vector cache, and a cold official run does not read that cache (c5b122e),
so the pack can make a demo fast but never a scored number faster.

Residual risk, accepted: the 1 % recompute cannot catch a *handful* of deliberately poisoned vectors. The pack
affects only demo rankings and is trusted as far as the release that ships it.

**Owner action:** attach `dist/acis-demo-index.zip` to the GitHub Release alongside the JSON. Regenerate it with
`uv run acis demo-index export` whenever the encoder or the document preparation changes. An outdated pack is
refused on import rather than silently mis-keyed.

## Known rough edges (not defects)

* The hybrid channel re-scores its head by position (`docs/spec/02` §4 stage 9), so `acis search` prints scores
  0, −1, −2, …. Mode A turns them into rank-derived scores. On screen they read as ranks, not similarities.
* `search()` still routes every query, because its response reports the route. On the hybrid channel's generic
  path, that is a second full-length encode per interactive query. The batch surfaces (`search_batch`, Mode B) no
  longer pay it (068a695).
