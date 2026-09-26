# Track B2 report — the Bonus (retrieval across all versions)

Scope: `docs/spec/07` Track B2, `docs/spec/04` §6 (D12). The held-out touch counter is **0 of 6**; everything
below is on the dev split. **Track B2 is not complete**: the lineage machinery meets its acceptance rows, but the
ranking claim the spec makes — grouped search beats flat all-version search on Evolution-NDCG@10 — did not hold on
the benchmark, and that is reported as measured rather than re-run until it passes.

## What the Bonus asks for, and where it stands

| Acceptance (`docs/spec/04` §6, `docs/spec/07` B2) | Status | Evidence |
|---|---|---|
| Wrong-merge rate ≤ 1 % | **PASS** — 0.0 % over 2,895 predicted revision pairs; pairwise precision, recall and F1 all 1.0; 323 of 323 lineages recovered | `[ledger:bench-52d48f680740]` |
| Duplicate-Rate@10 ≈ 0 in grouped mode | **PASS** — 0.00 grouped against 0.77 flat | `[ledger:bench-52d48f680740]` |
| Grouped beats flat on Evolution-NDCG@10, paired CI lower bound > 0 | **NOT MET** — 0.801 grouped vs 0.802 flat, Δ −0.08 pt, CI [−0.41, +0.24] | `[ledger:bench-52d48f680740]` |
| Best-Revision-Hit@1 beats flat | **NOT MET** — 0.72 for both | `[ledger:bench-52d48f680740]` |
| Cascade S0–S5, evidence and confidence on every link, `unknown` over a guessed merge | **PASS** | `src/acis/lineage/README.md` G-1…G-8, `tests/unit/test_lineage.py` |
| Grouping, best revision, span and timeline per lineage | **PASS** | `tests/integration/test_bonus_evolution.py` |

## The benchmark (`scripts/bench/bonus_evolution.py`)

300 real APPS dev solutions seed an Apps-Evolve history of 5 versions with 60 edits per version (the six in-place
edits, plus moves, additions and deletions), so every lineage is known by construction. Each seed's own dev
statement is its query. Encoder `gte-modernbert-base` (the G-M winner), dense channel, `cpu-fp32`, dev host.

Definitions were fixed in the script before the benchmark ran:

* **best revision** = the seed's original body, the only revision guaranteed to be an accepted solution;
* **Evolution-NDCG@10** = gain 3 for the best revision, 1 for any other body of the query's lineage, 0 otherwise,
  and a body already ranked gains nothing again; a grouped answer is judged as it is shown, each group expanded
  into its best revision and then its other members;
* **Duplicate-Rate@10** = share of the top ten answers whose lineage already appeared above.

Two corrections were made along the way. Both are recorded in the script and in the commits, and neither
touched a number after it was seen:

1. A first definition judged a grouped answer by its head alone. A 20-query smoke run showed that made the ideal
   list unattainable for *any* grouped answer, which penalises grouping by construction. It was replaced before
   the benchmark ran.
2. The first recorded run (`bench-0ae8ddf9d4be`) computed the grouped duplicate rate on the expanded list, so it
   counted a group's own members as duplicates of it (0.78). The rerun (`bench-52d48f680740`) is the same
   deterministic run with that fixed. Its NDCG numbers are identical.

## Why grouping ties rather than wins here — and what it would take

Each query has exactly one relevant lineage, and the encoder puts it first in 72 % of queries. When the target
lineage is on top, flat search's duplicates fall *below* it and cost almost nothing on NDCG. Grouping earns its
keep when other lineages' duplicates push the target down, and this regime rarely produces that. The duplicate
rate shows what grouping changes (0.77 → 0.00). The ranking metric shows it does not change *which* unit comes
first.

Rerunning variants until one passes would be tuning the benchmark to the claim. The honest options are the
owner's:

* **Drop the claim.** Present the Bonus as exact lineage recovery plus a duplicate-free, lineage-level answer with
  history, measured as above, without claiming a ranking gain.
* **Pre-declare one harder regime and run it once.** More versions and denser edits make flat duplicates crowd
  the list. The regime must be written into the script and committed *before* it runs.

## What is not measured yet

* Per-relation precision/recall (the spec lists it next to pairwise F1). The benchmark records pairwise figures only.
* The update-time target for P1 with the **real** encoder. B1's `[ledger:bench-9153e49050de]` used the stand-in.
  The commit-stream demo showed 1.4–4.7 s from commit to searchable with `gte-modernbert-base` on a loaded host,
  but that is a demo observation, not a ledgered measurement. It waits until the G1 sweep frees the CPU.
