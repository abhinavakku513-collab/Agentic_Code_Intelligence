# Questions for the organizers (send before Phase 1; record answers verbatim below — this file is `docs/official/`, so once filled it is authoritative)

| # | Question | Why it matters (decision it moves) | Default in force until answered |
|---|---|---|---|
| Q1 | Which **mteb version / Python version / OS** will run our submission, and is there a **wall-clock or memory limit** for the run? | D4 encoder size, cold-run SLO, compat matrix | pin mteb 2.21.0; SLO ≤ 4 h on the declared host |
| Q2 | Is the **CSV** a run file? Schema (`query_id, corpus_id, rank, score`?), how many rows per query, and **which MRR cutoff** is scored (`mrr_at_10`, `mrr_at_1000`, …)? | run.csv, ranking depth | top-1000, explicit rank; every `mrr_at_k` stays in the JSON |
| Q3 | How are **running time, model size and GPU use weighted** against NDCG@10/MRR? Is there a size or time budget? | D2/D4 (`size_tolerance_pts`) | smallest encoder within 1.0 NDCG@10 point of the best |
| Q4 | Is **supervised adaptation on the CoIR-APPS train split** permitted? Is it known that train solutions also appear in the test corpus? | D7/G3 (biggest accuracy lever) | allowed, gated, disclosed (D1, clean-pool score) |
| Q5 | Are models under **non-commercial licences** (e.g. CC-BY-NC code embedders) acceptable for the hackathon? | widens the encoder shortlist | excluded (permissive only) |
| Q6 | We implement retrieval at mteb's **SearchProtocol** level (`index()`/`search()`) inside a class derived from `AbsEncoder`, with an encoder-only `encode()` fallback shipped as a second JSON. Is that acceptable? | Mode A vs B (D8) | Mode A primary iff it beats B by the G-AB rule; both JSONs disclosed |
| Q7 | For **P1/Bonus**, what will the versioned data look like (git history, per-version folders, keyed JSONL of snippets)? How will "new commits every minute" be simulated? | ingest adapters, update-time SLO | unified `SnapshotSource` (JSONL / dirs / ZIP / git) |
| Q8 | Please share the **PPT template** (or its slide list). *Owner-side item: PPT and video are not Claude Code tasks.* | owner's slides only | – |
| Q9 | Will judges run our code on **their** hardware, offline? May we ship **prebuilt indexes/embedding caches** as Release artifacts? | D17 `--cache-verify`, prebuilt demo index | allowed as labelled artifacts; cold JSON is never replaced |

## Answers (fill in: date · organizer · verbatim text · decision/ADR affected)
_(empty)_
