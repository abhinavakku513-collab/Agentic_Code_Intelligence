#!/usr/bin/env bash
# `make demo` — the scripted, offline runbook (docs/spec/09 §6).
#
# Three acts, in the order they answer a judge's questions:
#   1. P0  — free-text retrieval over the APPS corpus, with the evidence for every hit.
#   2. P1  — a corpus changing under a live index: commits, freshness, rollback, crash recovery.
#   3. Bonus — the same history answered once per lineage instead of once per revision.
# Then it leaves the UI running so the same engine can be driven by hand.
#
# Everything is local. Nothing here reaches the network, and nothing reads held-out labels.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-uv run}"
PORT="${PORT:-8000}"
SPEED="${SPEED:-1.2}"
VERSIONS="${VERSIONS:-6}"
QUERY="${QUERY:-shortest path in a weighted graph using a priority queue}"

bold() { printf '\n\033[1m%s\033[0m\n\033[2m%s\033[0m\n' "$1" "$(printf '─%.0s' $(seq ${#1}))"; }

# The prebuilt demo index (D17, spec 09 R9): on a clean machine it saves embedding the whole corpus before the
# first answer. Verified on import (checksums, model fingerprint, 1 % recomputed); a no-op once imported.
DEMO_INDEX="${DEMO_INDEX:-dist/acis-demo-index.zip}"
if [ -f "$DEMO_INDEX" ]; then
  bold "0 · the prebuilt demo index"
  $PY acis demo-index import "$DEMO_INDEX" || echo "  (demo index not imported; the corpus will be embedded on first use)"
fi

bold "1 · P0 · free-text retrieval over the APPS corpus"
$PY acis search "$QUERY" --top-k 5 || {
  echo "  (the APPS corpus is not fetched; run \`make fetch\` for act 1)"; }

bold "2 · P1 · a corpus changing under a live index"
$PY python scripts/demo/commit_stream.py --versions "$VERSIONS" --speed "$SPEED" --query "$QUERY" --kill

bold "3 · the UI"
echo "  starting the API and the demo page on http://127.0.0.1:${PORT}/"
echo "  the same engine, driven by hand: free text, channel toggle, version pinning, lineage grouping."
echo "  Ctrl-C to stop."
exec $PY acis serve --port "$PORT"
