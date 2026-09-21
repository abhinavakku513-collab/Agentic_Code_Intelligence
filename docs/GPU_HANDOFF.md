# GPU hand-off protocol (offline training / bulk embedding on Colab or Kaggle)

GPU is allowed **only offline** and must be disclosed in the manifest (AGENTS.md). The official inference run is CPU-only.
Everything here is an **estimate [E] until measured**; record real numbers in the ledger (`kind=gpu`).

## 1. Size of the job (LoRA on a ~0.6B encoder, spec 02 §6)
Per training example ≈ query 420 tok + positive 190 + 7 hard negatives × 190 ≈ **1,940 tokens**; 4,000 examples ≈ **7.8 M tokens/epoch**.
LoRA + gradient checkpointing ≈ 6 · P_nonemb · T FLOPs with P ≈ 0.44 B → **≈ 2.0e16 FLOPs/epoch**.
| GPU (assumed effective throughput) | per epoch | 3 epochs × 5 models (4 cross-fit folds + final) |
|---|---|---|
| T4 (~10 TFLOPS eff., **fp16 only, no bf16**) | ≈ 35 min | **≈ 9 h** |
| L4 / P100-class (~15–20 TFLOPS eff.) | ≈ 20–25 min | ≈ 5–6 h |
| A100-class (~80–100 TFLOPS eff.) | ≈ 4–5 min | < 1.5 h |
Plus hard-negative re-mining (embed 8,765 docs per fold model: minutes on any GPU). Free tiers have session caps → **checkpoint/resume is mandatory**.
Cheaper variants to consider before committing 9 GPU-hours: 2-fold cross-fitting; 3 hard negatives instead of 7; max length 512; train ≤ 2 epochs; adapt the smaller encoder only.

## 2. Protocol
1. **Export** (local, Claude Code): `acis train export` → decontaminated TRAIN bundle (`pairs.jsonl`, `folds.json`, `drop_list.json`) + `MANIFEST.sha256`. Contains **no TEST queries or labels**.
2. **Upload** the bundle + pinned base model (or its HF SHA) + `scripts/train/` at a tagged commit to the notebook. Log git SHA, package versions, seed, GPU name in the run header.
3. **Train** with `scripts/train/train_lora.py --resume auto`: save every N steps (adapter + optimizer + RNG state + step) to persistent storage; fp16 + loss scaling on T4; deterministic seeds.
4. **Export**: merged `model.safetensors` (+ tokenizer, config) and OOF vectors per fold; write `TRAIN_MANIFEST.json` (hashes of inputs/outputs, seeds, hyper-parameters, wall-clock, GPU).
5. **Import** (local): verify SHA-256; parity gate — CPU fp32 vs GPU fp16 vectors on a 1 % sample must have cosine ≥ 0.999 before GPU vectors enter the shared cache; run G3 on the *OOF* results.
6. **Disclose**: `manifest.gpu_used=true`, purpose, hours, in the README and the owner's slides (INV-14: numbers cite ledger ids).

## 3. Never
Upload TEST queries/qrels; run TEST evaluation on the GPU host; put tokens/keys into notebooks; ship GPU-only artifacts without the CPU parity check.
