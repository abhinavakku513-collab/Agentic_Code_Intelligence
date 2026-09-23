# ADR-0007 — Phase 3 adaptation is not shipped: no GPU, and the frozen base stands

Status: **accepted 2026-09-23** (the spec's own contingency: "No GPU ⇒ decision recorded, frozen base stays",
`docs/spec/07` Phase 3).

## Context

Phase 3 fine-tunes the chosen encoder with LoRA on decontaminated APPS-train pairs and merges it back with
α-interpolation, gated by G3 and G-OOD. `docs/GPU_HANDOFF.md` describes the offline hand-off it needs: an
A100/L4-class GPU for a few hours, or a T4 with fp16 and loss scaling.

No GPU is available to this project. `runs/hardware.json` records `gpu: none` on the development host, and the
declared compute in `docs/STATUS.md` is still `undeclared`.

Training LoRA on CPU is not a smaller version of the same thing. The Phase 0 measurement puts a single *inference*
pass over the corpus at 0.72–2.6 h for the carded models; a training run is that cost multiplied by epochs, by
batches of eight negatives per positive, and by the backward pass. The K-fold cross-fitting G3 requires would
multiply it again. It is not hours, it is weeks.

## Decision

**Ship the frozen base encoder.** Phase 3 is recorded as not run, with the reason, rather than attempted at a
scale that could not produce a gate-quality answer.

The machinery stays where it is and stays honest about its state: `acis train export|import` and
`scripts/train/` are the hand-off path, and nothing in the scored path imports `peft` or `accelerate`.

## What is done instead

The accuracy Phase 3 would have bought is pursued where it is affordable on CPU, and measured the same way:

* **G-M** chose the encoder on measured accuracy and measured cost, and the choice was consequential — the
  candidates differ by 17 NDCG@10 points, which is more than a LoRA run would plausibly recover on the loser.
* **G5** — the cross-fitted learned ranker — is worth **+1.32 NDCG@10 points** out-of-fold on all 5,000 dev
  queries with the selected encoder, against a frozen-dense baseline of 71.03 `[ledger:gate-055152620da6]`.
* PRF and the code-aware tokeniser are built and gated, so a later run on better hardware can measure them
  without rebuilding anything.

## Consequences

* The submission's encoder is a public, pinned, permissively licensed model used zero-shot. That is a weaker
  accuracy story than an adapted one and a considerably stronger reproducibility story: no training artefact to
  publish, no seed to trust, no contamination argument to have.
* The D1 exposure diagnostic and the clean-pool report are moot — nothing was trained on APPS documents, so
  there is nothing for them to detect.
* If a GPU becomes available before the deadline, this ADR is superseded rather than edited, and the hand-off
  path is already in place: `docs/GPU_HANDOFF.md`, `acis train export`, then `import` behind the parity gate
  (CPU fp32 vs GPU vectors, cosine ≥ 0.999 on a 1 % sample).
