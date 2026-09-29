#!/usr/bin/env bash
# Queue (2026-09-29): long measurements back to back, so none shares the CPU with another. Run from a clean tree.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "$(date -u +%FT%TZ) 1/6 P0 pipeline evaluation (clean tree)"
uv run python scripts/bench/eval_pipeline.py
echo "$(date -u +%FT%TZ) 2/6 G-OOD through the served pipeline"
uv run python scripts/bench/g_ood_engine.py --record
echo "$(date -u +%FT%TZ) 3/6 G1: the unmeasured V2/1024 cell"
uv run python scripts/bench/g1_run.py --route statement_like
echo "$(date -u +%FT%TZ) 4/6 G1: gate rows for every measured cell"
uv run python scripts/bench/g1_run.py --route statement_like --record-only
echo "$(date -u +%FT%TZ) 5/6 G1: the rule's decision (printed only; writing configs/gates/G1.yaml needs the owner)"
uv run python scripts/bench/g1_run.py --route statement_like
echo "$(date -u +%FT%TZ) 6/6 latency on a quiet CPU"
uv run python scripts/bench/run.py --out runs/bench.final.json
echo "$(date -u +%FT%TZ) done"
