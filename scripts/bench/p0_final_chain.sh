#!/usr/bin/env bash
# Train the shipped ranker on the served pipeline's own pools, then measure that pipeline (P0 dev, REG, G-OOD).
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
RANKER=artifacts/ranker/p0-gte-pool500-symbols.txt
echo "$(date -u +%FT%TZ) 1/4 shipped ranker: refit on all 5,000 dev pools (engine candidate_pool + pool_features)"
uv run python scripts/bench/ltr_build.py --model gte-modernbert-base --tokenizer code --limit 0 --rounds 400 --save "$RANKER"
echo "$(date -u +%FT%TZ) 2/4 P0 dev evaluation through the engine (out of fold)"
uv run python scripts/bench/eval_pipeline.py
echo "$(date -u +%FT%TZ) 3/4 REG: generic route, ranker vs fusion"
uv run python scripts/bench/reg_route.py
echo "$(date -u +%FT%TZ) 4/4 G-OOD through the served pipeline"
uv run python scripts/bench/g_ood_engine.py --record
echo "$(date -u +%FT%TZ) done"
