# AGENTS.md — ACIS quick start (tool-agnostic; CLAUDE.md imports this file)

ACIS = local, offline, CPU-first code-retrieval engine (Samsung PRISM GenAI Hackathon, Theme 1). Read `CLAUDE.md` (constitution) and `docs/spec/` (contracts). Authority: `docs/official/` > `CLAUDE.md` > `docs/spec/`.

## Environment
- Python 3.12, **uv** (hash-locked `uv.lock` after Phase 0); `make setup` → `make doctor` (writes `runs/hardware.json`). Linux, macOS or WSL2; hooks need `python3` on PATH (native Windows unsupported).
- Runtime is offline (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`). Only `make fetch` uses the network. `ACIS_ROOT` = repo root; `ACIS_HOME` (outside git) = data, CAS, caches. TEST labels live in `~/.acis-sealed/` — never touch them.
- Official inference is `device=cpu`. A GPU (Colab/Kaggle) is allowed only offline, via `docs/GPU_HANDOFF.md`, and is disclosed in the manifest.

## Commands
| Command | Purpose |
|---|---|
| `make lint` / `make typecheck` | ruff format+check / mypy `--strict` on `acis.core`, `acis.eval`, `acis.engine` |
| `make test-fast` / `make test` / `make test-security` | quick suites (`-m "not slow and not gpu and not network"`; the Stop hook runs `tests/unit`) / everything / guard + adversarial + fuzz |
| `make robustness` | INV-15 property tests (+ `acis eval robustness` once the engine exists) |
| `make eval-dev CONFIG=configs/dev.yaml` · `make gate GATE=G3` | dev-split evaluation (never TEST) · pre-declared gate procedure |
| `scripts/run_detached.sh <name> <cmd…>` · `scripts/job_status.sh [name]` | anything longer than ~10 min (bake-off, embedding, training export) |
| `make rc-official RC=RC0 [MODE=B]` | **owner runs this in their own terminal** (or confirms the ask-gated command): cold, strict, uses the sealed HF cache |
| `make reproduce` / `make reproduce-cache` / `make demo` | judge quick start: cold official run / cache-verify / scripted hands-on runbook |
| `make bench` · `make preflight` · `make evidence` | latency/memory benchmarks · judge simulation · ledger-backed evidence files |

## Conventions
- `src/acis/<package>/` (map in `docs/spec/06` §0), each with a `README.md` contract; tests in `tests/{unit,property,metamorphic,contract,integration,security,chaos,perf,robustness}`.
- Type hints everywhere; `from __future__ import annotations`; structlog (no `print`) with hashes/lengths/timings, never query text or code by default. Pure functions in prep/features/rank; seeded randomness; config only through the frozen, hashed config object.
- Conventional Commits; small PRs; branch `phase/<id>-name`; tag `phase-<id>-done` after `/phase-gate <id>` PASS. Numbers in docs must cite `[ledger:<run_id>]`.

## Boundaries (enforced by hooks and permissions as speed-bumps; the seal itself is physical)
- Protected: `configs/splits.lock.json`, `runs/ledger.jsonl`, `data/**`, `docs/official/**`, `uv.lock`, `.claude/hooks/**`, `.claude/settings*.json`. No `pip install`, `trust_remote_code=True`, pickle/`eval`/`exec`/`shell=True`, paid APIs, query-time network.
- After copying this kit into a repo: run `/canary` (docs/CANARY.md), then check `/hooks`, `/permissions`, `/agents`; restart the session after adding subagents.
