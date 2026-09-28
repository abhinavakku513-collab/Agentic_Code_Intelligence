# ADR-0008 — A GPU is available: Phase 3 runs, and G3 decides whether it ships

Status: **accepted 2026-09-28** — the owner confirmed Colab/Kaggle GPU access. Supersedes ADR-0007, whose premise
("no GPU is available to this project") no longer holds.

## Context

ADR-0007 recorded Phase 3 as not run because only a CPU was available. The owner now has Colab or Kaggle GPU time.
The encoder is `gte-modernbert-base` (G-M, `[ledger:gate-055152620da6]`), about 4× smaller than the 0.6B model
`docs/GPU_HANDOFF.md` was sized for, so the full plan (five fold models, out-of-fold vectors and a final refit)
is estimated at 2–3 T4-hours. That estimate is unmeasured until `TRAIN_MANIFEST.json` reports it.

## Decision

Run Phase 3 through the hand-off, exactly as `docs/spec/02` §6 and `docs/GPU_HANDOFF.md` describe, with the
choices below made **before** any GPU time is spent:

* **One declared configuration**, frozen in the bundle (`acis.eval.adapt_export.TRAIN_CONFIG`): InfoNCE
  τ = 0.05, 7 hard negatives + in-batch negatives, 64 queries per step via GradCache, LoRA r = 16 on
  `Wqkv`/`Wo`/`Wi`, lr 1e-4 cosine, 2 epochs, re-mining after epoch 1, α ∈ {0.25, 0.5, 0.75, 1.0}, seed 0. Another
  configuration is a new, logged trial and needs a new export.
* **Negatives come only from the positives of training-fold dev queries.** The model never sees an unlabelled or
  held-out-fold document in training. This is stricter than the spec requires: it keeps the out-of-fold score
  honest and makes a learned "is this a train solution" signal (CLAUDE.md §4) untrainable.
* **Out-of-fold evaluation covers all 5,000 dev queries**, including queries dropped from training by
  decontamination (never trained on, so honestly scorable). The baseline is the unadapted base on the same GPU and
  precision, so G3 compares like with like.
* **The α-merge of the shipped model runs on the CPU** at import, behind the parity gate (cosine ≥ 0.999 against
  the GPU's 1 % sample).
* **Not in this trial:** the spec's 25 % replay of general code-retrieval pairs. Generality is protected by the α
  interpolation and checked by REG and G-OOD, both of which G3 requires.

## Consequences

* The adapted encoder ships **only if G3 and G-OOD pass**. Otherwise the frozen base stays, as ADR-0007 decided,
  and that outcome is recorded rather than tuned around.
* The reproducibility story gets heavier. The Release must carry the bundle manifest, `TRAIN_MANIFEST.json`, the
  adapter and the merged safetensors with their hashes, and the manifest must disclose `gpu_used=true` with
  purpose and hours (AGENTS.md, D15).
* The D1 exposure diagnostic and the clean-pool report stop being moot: the model now trains on APPS documents.
* G-OOD needs `acis eval robustness`, which is not built yet. G3 cannot be decided until it is.
