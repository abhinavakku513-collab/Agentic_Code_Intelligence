#!/usr/bin/env bash
# Refit the shipped ranker on the gte + Qwen pipeline's own 500-candidate pools, then measure that pipeline end to end
# through the engine, out of fold, on all 5,000 dev queries (TRAIN split; the held-out labels are never read).
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
RANKER=artifacts/ranker/p0-gte-qwen-pool500.txt
echo "$(date -u +%FT%TZ) 1/2 shipped ranker: refit on all 5,000 dev pools (engine candidate_pool + pool_features)"
uv run python scripts/bench/ltr_build.py --model gte-modernbert-base --tokenizer code --limit 0 --rounds 400 --save "$RANKER"
echo "$(date -u +%FT%TZ) 2/2 P0 dev evaluation through the engine (out of fold)"
uv run python scripts/bench/eval_pipeline.py --no-route-analysis
echo "$(date -u +%FT%TZ) done"
