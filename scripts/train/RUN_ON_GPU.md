# Phase 3 on Colab or Kaggle — the owner's runbook

This is the GPU half of `docs/GPU_HANDOFF.md`. Everything around it (the export, the import, the gates) runs on
the dev machine. **Upload nothing but what is listed in step 1**: the bundle holds dev-split data only, and no
held-out query or label may ever reach the notebook.

Expected cost with `gte-modernbert-base` (149M parameters, about 4× smaller than the 0.6B model `GPU_HANDOFF.md`
was sized for): about 2–3 GPU-hours on a T4 for five fold models, their out-of-fold vectors and the final model.
That is an estimate; the real figure lands in `TRAIN_MANIFEST.json`.

## 1. On the dev machine

```bash
uv run acis train export dist/train-bundle      # decontaminated dev-only bundle + MANIFEST.sha256
tar czf dist/train-bundle.tgz -C dist train-bundle
git rev-parse HEAD                               # note the commit: the notebook records it
```

Upload two files to the notebook (Colab: Drive; Kaggle: a private dataset): `dist/train-bundle.tgz` and
`scripts/train/train_lora.py` **from that commit**.

## 2. In the notebook (GPU runtime)

```python
# Cell 1 — environment. Versions are recorded in TRAIN_MANIFEST.json either way.
!pip -q install "peft>=0.11" "safetensors>=0.4.3"
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
```

```python
# Cell 2 — the pinned base model, by commit, verified against the hashes in configs/models/gte-modernbert-base.yaml
from huggingface_hub import snapshot_download

BASE = snapshot_download(
    "Alibaba-NLP/gte-modernbert-base",
    revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
    allow_patterns=["*.json", "*.safetensors", "1_Pooling/*"],
)
import hashlib, pathlib

for name, want in {
    "model.safetensors": "3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba",
    "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "config.json": "8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed",
}.items():
    got = hashlib.sha256(pathlib.Path(BASE, name).read_bytes()).hexdigest()
    assert got == want, f"{name}: {got} is not the pinned file"
print("base model verified")
```

```python
# Cell 3 — unpack the bundle onto PERSISTENT storage, so a lost session keeps its checkpoints.
#   Colab:  OUT = "/content/drive/MyDrive/acis-phase3"   (mount Drive first)
#   Kaggle: OUT = "/kaggle/working/acis-phase3"          (save a version before the session ends)
OUT = "/content/drive/MyDrive/acis-phase3"
!mkdir -p $OUT && tar xzf /content/drive/MyDrive/train-bundle.tgz -C $OUT
```

```python
# Cell 4 — train. Re-run this same cell after any disconnect: finished stages are skipped and the rest resumes
# from its last checkpoint.
GIT_SHA = "<the commit from step 1>"
!python train_lora.py --bundle $OUT/train-bundle --base $BASE --out $OUT/out --resume auto --git-sha $GIT_SHA
```

If one session cannot fit everything, split it: `--folds 0,1 --skip-final`, then `--folds 2,3 --skip-final`, then
`--folds 4` (which also trains the final model). The same `--out` directory must be used every time.

## 3. Back on the dev machine

Download `$OUT/out` whole (it is small apart from the fp16 vectors, a few hundred MB), then:

```bash
uv run --extra train acis train import path/to/out --bundle dist/train-bundle
```

The import refuses outputs that do not match `TRAIN_MANIFEST.json`, merges the final adapter at every alpha on the
CPU, and stops unless each merged model reproduces the GPU's vectors at cosine ≥ 0.999. It writes
`runs/g3/import_report.json` with the out-of-fold scores. **It decides nothing**: G3 also needs REG and G-OOD.

## Never

- upload anything but the bundle and `train_lora.py`: no held-out query, no label, no token or key;
- run any held-out evaluation on the GPU host;
- change `config.json` in the bundle — the manifest check will refuse it. A different configuration is a new,
  logged trial: re-export it.
