# `acis.prep` — contract

Spec: `docs/spec/02-p0-engine.md` §2 and §5.

**Responsibility.** Deterministic normalisation (`q1` for queries, `d1` for documents), soft segmentation, views
(V0/V1) and head+tail truncation. Every function is pure: same text in, same text out, on any machine.

| # | Guarantee | Test |
|---|---|---|
| P-1 | `q1` is idempotent and folds format noise (NFKC, CRLF, trailing spaces, blank-line runs, control characters) | `tests/unit/test_prep_and_engine.py`, `tests/robustness` |
| P-2 | `d1` keeps tabs — they are Python syntax — and strips only BOM, `\r` and trailing whitespace | `tests/unit/test_prep_and_engine.py` |
| P-3 | Numeric folding is generic arithmetic over any base/exponent/offset, never a list of interesting moduli (INV-15) | `tests/unit/test_prep_and_engine.py` |
| P-4 | Segmentation is a hint: no marker list gates correctness, and failure falls back to the whole text | `tests/unit/test_prep_and_engine.py` |
| P-5 | Truncation keeps head **and** tail and reports `truncated`, which becomes an LTR feature in Phase 4 | `tests/unit/test_prep_and_engine.py` |

**Non-goals.** No model-specific instruction strings here (they live in `configs/models/<name>.yaml`, per D4); no
comment stripping until an ablation earns it.
