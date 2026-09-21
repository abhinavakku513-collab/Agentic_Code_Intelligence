# `acis.embed` — contract

Spec: `docs/spec/02-p0-engine.md` §3.

**Responsibility.** The dense-channel interface (`Encoder`), exact search (one matmul, D5), and — in Phase 1 only —
a model-free stand-in encoder that exists to validate the harness.

| # | Guarantee | Test |
|---|---|---|
| M-1 | `encode` returns L2-normalised float32 rows in input order; batch size never changes a vector (INV-3) | `tests/property/test_invariants.py` |
| M-2 | Search is exact (`q · Dᵀ`): no ANN, no vector database below 250k vectors (D5) | `tests/metamorphic/test_parity.py` |
| M-3 | A stand-in encoder reports `submission_capable = False` and is refused by a strict run | `tests/security/test_seal.py` |
| M-4 | Cache keys include the model fingerprint and the numeric profile (docs/spec/02 §3) | `tests/property/test_invariants.py` |

**Phase boundary.** The real runtime — token-budget batching, the content-addressed vector cache, numeric profiles,
pooling per model card — is Phase 2. Nothing above this package knows which encoder is behind the interface.
