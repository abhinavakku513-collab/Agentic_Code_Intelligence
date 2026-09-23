# Phase 2 progress report — dense engine and gates, stopped at the weights

Scope: `docs/spec/07` Phase 2 (dense engine + encoder bake-off → RC0). **The phase is not complete**, and cannot
be, from this machine: the sandbox has no network, so no model weights exist here. The held-out touch counter is
**0 of 6** and no accuracy claim appears below — the only encoder that has ever run in this repository is the
model-free stand-in, which reports `submission_capable = False` and which a strict run refuses to serve.

What that leaves is everything *around* the model, and it is finished: both gates run as a single command, and
the measuring pass they call is the shipping path rather than a notebook approximation of it.

## Built and tested

| Component | What it guarantees | Evidence |
|---|---|---|
| Model cards (`acis.embed.registry`) | Instruction strings, pooling, truncation and licence come from `configs/models/<key>.yaml`; `trust_remote_code` is refused at load; a card maps route → task key, and only `statement_like` gets the APPS-tuned instruction by default (INV-15) | `tests/unit/test_embed_phase2.py`, `tests/unit/test_dense_wiring.py` |
| Encoder runtime (`acis.embed.runtime`) | Length-sorted token-budget batching that is never observable in a vector (INV-3); pooling that honours the mask and the padding side; duplicate texts cost one forward pass within a call as well as across calls; the fingerprint moves with weights, card or numeric profile | `tests/unit/test_embed_runtime.py` |
| Vector cache (`acis.embed.cache`) | Atomic writes, mmap reads, a corrupt entry is a miss rather than an exception; the key covers model fingerprint, numeric profile, prep hash, text and prompt | `tests/unit/test_embed_phase2.py` |
| Encoder factory (`acis.embed.factory`) | One constructor for Mode A, Mode B and the bake-off; a model whose weights are absent raises `NotReady` naming the fetch step and is never silently replaced by the stand-in | `tests/unit/test_dense_wiring.py` |
| Views (`acis.embed.views`) | V2 is the normalised mean of V0 and V1, and costs no second encode when a query has no structure | `tests/unit/test_dense_wiring.py` |
| Scorecard (`acis.embed.scorecard`) | Cold measured before warm and kept apart (D17); an unknown cold-pass projection is reported as unknown, never as a passing one | `tests/unit/test_bakeoff.py` |
| Gate G-M (`acis.eval.bakeoff`) | Smallest model within tolerance, tolerance widened only for a slow baseline, reference and non-permissive candidates measured but never selectable — and never able to change which shippable model is picked | `tests/unit/test_bakeoff.py`, `tests/integration/test_bakeoff_pass.py` |
| Gate G1 (`acis.eval.sweep`) | Cheapest cell the paired bootstrap cannot separate from the best; a costlier cell must beat the incumbent by the gate rule; decisions are per route and written once | `tests/unit/test_sweep_g1.py` |
| Model supply chain (`acis.embed.modelfetch`) | Safetensors only — never a pickle checkpoint; the revision is resolved to a commit before anything downloads; a non-permissive model needs an explicit `--reference`; `--pin` writes the commit and every file hash into the card (G0.4) | `tests/unit/test_model_fetch.py` |
| `acis search` | Exactly `min(top_k, N)` hits, evidence re-read from the content store by hash (INV-1), and a stand-in encoder labelled as one | `tests/integration/test_search_cli.py` |
| `make bench` | Model latency, cold-query latency and retrieval-core latency measured separately, written to `runs/bench.json` and recorded in the ledger | `tests/integration/test_bench.py` |

Suite at the time of writing: 686 passed, 3 xfailed (the declared guard-hook gaps), `make lint` and
`make typecheck` clean, ledger chain OK, seal clean.

## The two gates are one command each

```
uv run acis eval gate --gate G-M                      # measure every carded model, apply the D4 rule
uv run acis eval gate --gate G1 --route statement_like # sweep task string x view x length for one route
```

Both refuse to *write* a decision from anything but all 5,000 dev queries (`docs/spec/09` §2), both record a
ledger row per measured candidate, and both stop with the fetch step named when weights are missing rather than
scoring an absent model as a bad one. Writing them before the weights arrived was the point: when the weights
land, G-M is a run rather than a project.

## What is blocked, and by what

| Blocked | Why | Unblocked by |
|---|---|---|
| G0.4 (model radar and pin), G0.5 (zero-shot on 5,000) | No weights on this machine | `make fetch-models MODELS=...` on a networked machine |
| G-M bake-off, G1 sweep, RC0 | Same | Same |
| A meaningful scorecard | The only measurable encoder is the stand-in, whose cost says nothing about a real model | Same |

## What the owner has to run

```
make fetch-models MODELS=qwen3-embedding-0.6b       # add more keys as G0.4's radar shortlists them
uv run acis fetch --models qwen3-embedding-0.6b --verify   # offline re-check against the pins, any time after
```

`--pin` (which `make fetch-models` passes) writes the commit and every file hash into the card; from then on the
runtime refuses to encode with weights that do not match it. A non-permissively licensed candidate needs
`--reference` — it is measured for the gap the permissive-only rule costs us and can never be selected (D4).

Still outstanding from earlier phases, all owner-only: the hook canary sign-off (G0.0), confirming or rejecting
ADR-0005, the `deadline` and `compute` rows in `docs/STATUS.md`, and — before RC0 — `acis fetch --sealed`
followed by `python -m acis.eval.final --smoke`.

## Defects found and fixed

Four, of which three were invisible in the sense that everything still produced plausible output:

* **Mode B built its own encoder.** It hard-coded the stand-in, so the moment weights landed, gate G-AB would
  have compared two different models while reporting one name. Both modes now go through one factory.
* **The route stopped at the engine.** The instruction a query is encoded with is chosen per route (INV-15), but
  the route never reached the encoder: every query would have been encoded generically while the routing code
  looked correct. It is now part of the `Encoder` contract and part of the query-vector cache key, so two routes
  cannot share one cached vector of the same text.
* **The G-M tolerance was measured from the best *measured* candidate.** Measuring a strong non-permissive model
  would then have pushed every shippable candidate out of tolerance and forced the frozen default — a reference
  measurement changing the answer, which is exactly what it must never do. The baseline is now the best
  *selectable* candidate and the gap is reported separately, which is the number the reference exists to produce.
* **`bm25s` raised `ValueError` on a corpus with no indexable term** — found by the property suite. A corpus of
  bare numbers and punctuation is a real corpus, so the index now matches nothing and the snapshot reports the
  missing channel, instead of a dependency exception escaping as the engine's answer.

## The one number that exists

`[ledger:bench-b3586309bd8e]` — dev host, 8 threads, `cpu-fp32`, 8,765 units, 50 queries, stand-in encoder
(`acis/hashing-4096`, not submission-capable), peak RSS 543.04 MB, snapshot built in 2.868 s:

| Measurement | p50 | p95 | Target (`docs/spec/02` §7) |
|---|---|---|---|
| encode one query | 1.189 ms | 1.954 ms | ≤ 150 ms |
| search, new query | 9.257 ms | 14.993 ms | — |
| search, repeated query (retrieval core) | 7.995 ms | 16.396 ms | ≤ 15 ms |

The core is over its p95 target on this host, and it is recorded rather than tuned. The stand-in's vectors are
4,096-dimensional against the 1,024 a real candidate is likely to produce, so the dense matmul is roughly four
times the work it will be; and the figure that matters is the one the chosen encoder produces on the declared
scorecard host, which is a measurement that waits for weights. Ranking was already made exact-but-cheap this
phase (the top-`k` cut of parity P8), so the remaining cost is the matmul itself.

## Three network paths, not one

`AGENTS.md` says only `make fetch` uses the network; the Phase 1 report already recorded a second, dev-only path
(`scripts/validate_reg_task.py`, into `~/.acis/reg`). Phase 2 adds a third: `acis fetch --models`
(`make fetch-models`), which downloads model weights into `ACIS_HOME/models/<key>`. The set is closed and
asserted — `tests/security/test_hardening.py::test_only_the_declared_modules_go_online` names the two modules
allowed to leave the offline default, so a fourth has to be argued for where the assertion lives. None of them
exists at query time and the official run is offline. Editing `AGENTS.md` needs owner confirmation, so this is
recorded here instead.

## What is deliberately absent

Hybrid fusion, LTR, PRF, the code-aware tokeniser and routing v1.1's OOD bank: all Phase 4, each with its own
gate, and `mode="hybrid"` still raises rather than inventing an ungated ranking. Adaptation is Phase 3. No
release candidate has been produced, because producing one without a real encoder would mean shipping the
harness stand-in.
