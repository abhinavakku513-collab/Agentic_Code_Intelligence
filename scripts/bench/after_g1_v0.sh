#!/usr/bin/env bash
# Queue for when G1's V0/1024 cell lands (2026-09-28): the running G1 process predates the V2 fix (8801415), so
# it is stopped, its invalid V2 result removed, the ranker re-measured from a clean tree (G2/G5/G-AB inputs) and
# the Phase 3 bundle exported — all reusing the V0/1024 query vectors that cell just cached — and G1 restarted on
# the fixed code for its V2 cells.
set -euo pipefail
cd "$(dirname "$0")/../.."
cell=runs/g1/statement_like/notask_V0_1024.json
until [ -f "$cell" ]; do sleep 60; done
echo "$(date -u +%FT%TZ) V0/1024 landed; stopping G1"
[ -s runs/jobs/g1-statement_like/pid ] && pkill -P "$(cat runs/jobs/g1-statement_like/pid)" || true
ps -eo pid,cmd | grep "[g]1_run.py" | awk '{print $1}' | xargs -r kill || true
sleep 5
rm -f runs/g1/statement_like/notask_V2_*.json
uv run python scripts/bench/ltr_build.py --model gte-modernbert-base --limit 0 --rounds 400 --tokenizer stock --save runs/ltr/gte-stock.txt
uv run python scripts/bench/ltr_build.py --model gte-modernbert-base --limit 0 --rounds 400 --tokenizer code --save runs/ltr/gte-code.txt
HF_HUB_OFFLINE=1 uv run acis train export dist/train-bundle
echo "$(date -u +%FT%TZ) restarting G1 for the V2 cells"
scripts/run_detached.sh g1-statement_like uv run python scripts/bench/g1_run.py --route statement_like
