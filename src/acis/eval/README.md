# `acis.eval` — contract

Spec: `docs/spec/03-mteb-and-integrity.md` (all sections) and `docs/spec/06` §5. Rules: `.claude/rules/mteb-eval.md`.

**Responsibility.** Everything that turns a ranking into a number a reader can trust: metrics, splits and the split
lock, decontamination, the paired bootstrap, the hash-chained ledger, the ladder, run files, submission
verification, and the single sanctioned reader of the held-out labels.

| # | Guarantee | Test |
|---|---|---|
| E-1 | `metrics.score_run` equals mteb/pytrec_eval to 1e-9 on random, oracle and BM25 runs, **including** inside the float32 collapse zone | `tests/metamorphic/test_parity.py` (B0, P7) |
| E-2 | NDCG/MAP/Recall/P/Success are graded in trec_eval order (float32, doc id descending); MRR in mteb's exact-float order | `tests/unit/test_eval_primitives.py` |
| E-3 | Metrics re-scored from `run.trec` reproduce the direct metrics to 1e-9 (parity P3) | `tests/metamorphic/test_parity.py` |
| E-4 | Folds come from `sha256(query_text) mod k`; the split lock refuses to be rewritten with a different hash | `tests/integration/test_dev_harness.py` |
| E-5 | The ledger is append-only and hash-chained; editing a row is detected; the held-out touch budget is enforced | `tests/unit/test_eval_primitives.py` |
| E-6 | The paired bootstrap implements the one gate rule (Δ ≥ +0.5 pt **and** CI lower bound > 0), seeded and reproducible | `tests/unit/test_eval_primitives.py` |
| E-7 | Decontamination can only remove training pairs, never add or reweight them; it reads query texts, never labels | `tests/unit/test_eval_primitives.py` |
| E-8 | Dev tasks load the dev split only; any other split raises `SealedDataAccess` (INV-8) | `tests/security/test_seal.py` |
| E-9 | `verify-submission` fails for one reason at a time and never counts a SKIP as a PASS | `tests/integration/test_verify_submission.py` |
| E-10 | `acis.eval.final` is the only module that reads held-out labels, and refuses outside the official environment | `tests/security/test_seal.py` |

**Non-goals.** No ranking logic (that is `acis.engine`); no mteb types cross into the engine (INV-11).
