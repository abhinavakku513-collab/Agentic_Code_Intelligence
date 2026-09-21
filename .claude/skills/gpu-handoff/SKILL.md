---
name: gpu-handoff
description: Prepare or import an offline GPU job (LoRA training / bulk embedding) per docs/GPU_HANDOFF.md — export the decontaminated bundle, or verify and import results with the CPU parity gate.
argument-hint: [export|import <path>]
disable-model-invocation: true
---
Mode: $ARGUMENTS. Delegate to the `training-engineer` subagent and follow `docs/GPU_HANDOFF.md`.
- **export**: confirm G0.6 (seal) passed; run `uv run acis train export`; verify the bundle contains no TEST queries/labels (`MANIFEST.sha256`, `drop_list.json`); print the GPU-hour estimate for the chosen variant (T4 / L4 / A100) as [E] and the cheaper-variant options; tell the owner what to upload (bundle, pinned model SHA, `scripts/train/` at a tagged commit) and the exact `train_lora.py --resume auto` command.
- **import <path>**: verify SHA-256s; run `uv run acis train import <path>` (CPU fp32 vs GPU parity, cosine ≥ 0.999 on a 1 % sample); write `TRAIN_MANIFEST.json`; append `kind=gpu` ledger rows; then hand OOF vectors to `retrieval-experimenter` for G3 and G-OOD.
Never upload or process TEST queries/labels, never put tokens or keys in notebooks, never ship GPU-only artifacts without the parity check.
