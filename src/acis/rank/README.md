# `acis.rank` — contract

Spec: `docs/spec/03-mteb-and-integrity.md` §2.6 (D9, INV-10) and `docs/spec/02` §4 stage 10.

**Responsibility.** Score composition. Candidates, PRF, the LightGBM ranker and confidence calibration arrive in
Phase 4 behind gates G4/G5.

| # | Guarantee | Test |
|---|---|---|
| R-1 | Mode A scores are rank-derived `(top_k + 1 − rank) / top_k` — finite, strictly decreasing, gap `1/top_k` | `tests/unit/test_eval_primitives.py` |
| R-2 | The gaps survive float32, so NDCG and MRR can never disagree about the same ranking (V-08) | `tests/property/test_invariants.py`, P7 |
| R-3 | Exactly `min(top_k, N)` entries per query; a duplicate document id is an error, not a silent collapse | `tests/contract/test_adapter_contract.py` |
| R-4 | Exact-content duplicates get distinct adjacent ranks ordered by corpus ordinal — the only use of order (INV-4) | `tests/metamorphic/test_parity.py` |

**Non-goals.** No raw model score ever reaches the harness in Mode A; no basis-vector or score-vector tricks (D8).
