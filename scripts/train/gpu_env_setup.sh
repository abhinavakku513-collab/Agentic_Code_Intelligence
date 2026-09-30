#!/usr/bin/env bash
# OWNER-RUN, once. An offline GPU *tool* environment for bulk embedding (ADR-0008: GPU only offline, behind a
# parity gate). It is NOT the project environment: `uv.lock`, serving and the official run stay CPU-only, and
# nothing here is imported by `acis`. The project guard refuses package installs from the agent, so the owner
# decides whether this environment exists.
#
#   bash scripts/train/gpu_env_setup.sh
#
# Then the agent (or you) can run:  scripts/run_detached.sh gpu-embed ~/.acis/gpu/venv/bin/python scripts/train/gpu_embed.py
set -euo pipefail
cd "$(dirname "$0")/../.."
read -r TRANSFORMERS TOKENIZERS SAFETENSORS NUMPY < <(uv run python -c \
  "import transformers, tokenizers, safetensors, numpy; print(transformers.__version__, tokenizers.__version__, safetensors.__version__, numpy.__version__)")
VENV="$HOME/.acis/gpu/venv"
uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV/bin/python" "transformers==$TRANSFORMERS" "tokenizers==$TOKENIZERS" \
  "safetensors==$SAFETENSORS" "numpy==$NUMPY" pyyaml structlog orjson pydantic
"$VENV/bin/python" -c "import torch; ok = torch.cuda.is_available(); print('cuda available:', ok, torch.cuda.get_device_name(0) if ok else '')"
