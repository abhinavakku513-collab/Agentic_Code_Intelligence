# `acis.lexical` — contract

Spec: `docs/spec/02-p0-engine.md` §2; parity target: `mteb/baseline-bm25s`.

**Responsibility.** One BM25 index per snapshot, with tokenisation matched to the mteb baseline so the B1 ladder
rung is a true parity check rather than a different system.

| # | Guarantee | Test |
|---|---|---|
| X-1 | Per-snapshot statistics: a query only ever sees the documents of its own snapshot (INV-2) | `tests/property/test_invariants.py` |
| X-2 | A single query equals its row inside any batch (INV-3) | `tests/unit/test_prep_and_engine.py` |
| X-3 | Defaults mirror `mteb/baseline-bm25s` (`k1=1.5, b=0.75, delta=0.5, lucene`), so B1 matches B0 rank-for-rank | `tests/metamorphic/test_parity.py` (P2) |
| X-5 | A corpus with no indexable term is an empty channel, not an exception: the snapshot reports `lexical` missing and the dense channel is unaffected | `tests/unit/test_prep_and_engine.py` |
| X-4 | PyStemmer is optional: absent, the index and the baseline are both built stemmer-free | `tests/metamorphic/test_parity.py` |

**Phase boundary.** The code-aware tokeniser of spec 02 §2 (identifier splitting, operator tokens, literal folding)
belongs to the hybrid phase and gate G2; shipping it in the harness phase would make the parity result meaningless.
