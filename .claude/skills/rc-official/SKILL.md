---
name: rc-official
description: Guided release-candidate flow — the ONLY path that loads TEST labels. Claude prepares and verifies; the OWNER runs the official command in their own terminal (or confirms the ask-gated command). Manual only.
disable-model-invocation: true
---
Never start the official evaluation silently. Steps:
1. Preconditions (stop and report if any fails): clean git tree; `make lint typecheck test test-security` green; `hook_canary: passed`; gates G-M, G1–G7, G-AB, G-OOD decided in `configs/gates/` as the RC requires (RC0: G-M + G1 only, Mode B); `configs/official.yaml` hash equals the ledger `frozen` entry; TEST touches used + planned evaluations (RC0: 1, RC1/RC2: 2 for modes A+B) ≤ 6; the physical seal check (G0.6) is green; the RC id is new.
2. Tell the owner the exact command to run **in their terminal** and its expected runtime from the scorecard: `make rc-official RC=<id> [MODE=B|AB]` (or `scripts/run_detached.sh rc-<id> make rc-official RC=<id>` when they want it in the background). The official cold run must finish uninterrupted so `evaluation_time` stays honest. If the owner explicitly wants Claude to run it, the `ask` permission prompt is the confirmation.
3. After the run: `uv run acis eval verify-submission runs/<id>`; record the resource scorecard (params, MB, RSS, cold `evaluation_time`, warm time, latency, GPU none); read only the ledger row and verify output; build the release pack via `/evidence-pack`; update status through `/phase-gate`.
Never inspect per-query TEST results; never rerun the same RC id; never read `~/.acis-sealed/`.
