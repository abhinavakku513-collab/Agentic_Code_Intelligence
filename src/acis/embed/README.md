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
| M-10 | Pooling honours the mask and the padding side; identical texts cost one forward pass within a call as well as across calls; the fingerprint moves with weights, card or numeric profile | `tests/unit/test_embed_runtime.py` |
| M-11 | The scorecard measures cold before warm and reports an unknown projection as unknown, never as a passing one (D2, D17) | `tests/unit/test_bakeoff.py` |
| M-12 | One factory builds the encoder for Mode A, Mode B and the bake-off; a model whose weights are absent raises `NotReady` naming the step that fetches them, and is never silently replaced by the stand-in | `tests/unit/test_dense_wiring.py` |
| M-15 | Fetching a model downloads safetensors only, resolves the revision to a commit before anything downloads, and pins the commit plus every file hash into the card (G0.4) | `tests/unit/test_model_fetch.py` |
| M-14 | A card maps route → task key: only `statement_like` uses the APPS-tuned instruction by default, G1 may repoint one route without touching another, and doing so moves the fingerprint | `tests/unit/test_dense_wiring.py` |
| M-13 | `encode` takes the route: a query's instruction is chosen per route (INV-15), and V2 is the normalised mean of V0 and V1 with no second encode when the query has no structure | `tests/unit/test_dense_wiring.py` |

**Phase boundary.** The runtime is built against an injected `Backend`, so loading, pooling, batching, caching,
normalisation and ordering are all tested without weights — when real weights arrive the only untested thing is
the model. `load_runtime` raises `NotReady` until the owner has fetched and pinned them (G0.4); the measurement
passes then run unchanged. Nothing above this package knows which encoder is behind the interface.
