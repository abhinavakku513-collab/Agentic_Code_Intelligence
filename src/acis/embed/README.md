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

| M-5 | Batching groups by **padded** token budget and restores input order exactly; row *i* out is row *i* in | `tests/unit/test_embed_phase2.py` |
| M-6 | The vector cache key covers model fingerprint, numeric profile, prep hash, text and prompt — omit one and a number is about a different model | `tests/unit/test_embed_phase2.py` |
| M-7 | Cache writes are atomic and a corrupt entry is a miss, never an exception at the caller | `tests/unit/test_embed_phase2.py` |
| M-8 | Instruction strings, pooling and truncation come from `configs/models/<name>.yaml`; only `statement_like` uses the APPS-tuned task string (INV-15) | `tests/unit/test_embed_phase2.py` |
| M-9 | A model may ship only if pinned, permissively licensed and free of remote code (D4); others may be measured as reference | `tests/unit/test_embed_phase2.py` |

**Phase boundary.** The encoder *runtime itself* (loading weights, pooling, `torch.inference_mode`) lands with the
G-M bake-off, which needs downloaded weights. Everything around it — batching, caching, cards, pinning — is built
and tested here without them, so the measurement passes can run unchanged the moment weights are available.
Nothing above this package knows which encoder is behind the interface.
