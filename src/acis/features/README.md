# `acis.features` — contract

Spec: `docs/spec/02-p0-engine.md` §2 and Appendix A. Rules: `.claude/rules/p0-engine.md`, `.claude/rules/security.md`.

**Responsibility.** Parse-only features of a document, generic features of a query, and the bridge between them —
the signals the dense channel cannot see because a problem statement and its solution share shape rather than
vocabulary.

| # | Guarantee | Invariant | Test |
|---|---|---|---|
| F-1 | Features are derived by `ast.parse` alone; nothing is imported, executed, compiled for execution or `eval`'d, including constant folding | INV-5 | `tests/unit/test_features.py` |
| F-2 | An unparseable file still produces a record (`parse_ok=False`) and stays retrievable | — | `tests/unit/test_features.py` |
| F-3 | Extractors are generic: an unfamiliar modulus behaves exactly like a famous one, and no closed list gates any value | INV-15 | `tests/unit/test_features.py` |
| F-4 | An absent signal is `NaN`, never a confident zero — "nothing to compare" and "no overlap" are different claims | INV-15 | `tests/unit/test_features.py` |
| F-5 | The bridge separates a matching solution from a non-matching one on output literals, I/O shape and test-case structure | — | `tests/unit/test_features.py` |
| F-6 | `availability()` reports the share of bridge features that fired — the ρ the router and the abstaining ranker read | INV-15 | `tests/unit/test_features.py` |

**Non-goals.** No ranking (that is `acis.rank`), no execution of any kind, and no feature that encodes which split
a document came from.
