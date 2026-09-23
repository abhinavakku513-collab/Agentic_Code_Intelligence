# `acis.lineage` — contract

Spec: `docs/spec/04-storage-p1-bonus.md` §6 (D12). Rules: `.claude/rules/storage-versions.md`.

**Responsibility.** The Bonus: recognising that many revisions across versions are one unit, closing those links
into lineages, and answering a search with lineages instead of revisions.

| # | Guarantee | Invariant | Test |
|---|---|---|---|
| G-1 | The cascade runs cheapest-and-most-certain first (S0 identical → S1 same key → S2 moved bytes → S3 name-insensitive equality → S4 inferred → S5 added/removed), and every link stores its evidence and confidence | D12 | `tests/unit/test_lineage.py` |
| G-2 | A link below the confidence floor joins nothing: the revisions stay separate lineages and the caller sees two answers rather than one confident wrong one | D12 | `tests/unit/test_lineage.py` |
| G-3 | S4 — the only inferential stage — is a global best assignment, never a greedy first match, and its links are labelled `inferred` all the way into the result | D12 | `tests/unit/test_lineage.py`, `tests/integration/test_bonus_evolution.py` |
| G-4 | Similarity reuses cached vectors and text already in the store; recognising revisions embeds nothing | — | `tests/unit/test_lineage.py` |
| G-5 | A lineage scores as its **best** member, so a unit improved in a later version is found by a query only that revision answers | D12 | `tests/unit/test_lineage.py` |
| G-6 | A grouped answer contains each lineage once; the flat baseline's duplicate rate is reported alongside it | D12 | `tests/integration/test_bonus_evolution.py` |
| G-7 | Lineage ids are derived from the earliest member, so they are stable as history grows | — | `tests/unit/test_lineage.py` |
| G-8 | Every group carries its version span, its members and a timeline of what changed | — | `tests/integration/test_bonus_evolution.py` |

**Non-goals.** No merging without evidence — `unknown` is a valid answer and a better one than a guess. No
embedding: every similarity is computed from what a snapshot already holds.
