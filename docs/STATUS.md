# ACIS status — single source of progress truth (updated only by /phase-gate on PASS)
kit_version: 1.1 (ADR-0001…0003 accepted 2026-09-20)
current_phase: 0            # Foundations and trust gates (Track B not started)
phase_state: not-started    # not-started | in-progress | gate-pending | done
last_gate: none
deadline: undeclared        # OWNER: put the real date here; work backwards from docs/TRIAGE.md floors F0–F4
hook_canary: not-run        # OWNER + /canary (G0.0) must pass before Phase 0 work; repeat after every Claude Code upgrade
test_touch_used: 0 of 6     # RC0 x1, RC1 x2 (A+B), contingency x2, post-hoc clean-pool x1
reference_host: undeclared  # `make doctor` in Phase 0 (T-min 4c/8GB, T-rec 8c/16GB)
compute: undeclared         # OWNER: dev cores/RAM · GPU access + hours (Colab/Kaggle counts) · who runs the official cold pass
faq: received 2026-09-20 (docs/official/FAQ.md) · organizer answers: none yet (docs/official/ORGANIZER_QA.md)
scope: PPT and demo video are owner-made (docs/submission/OWNER_HANDOFF.md)

## Owner actions pending
1 Run the canary (docs/CANARY.md) · 2 answer the compute questions above · 3 send docs/official/ORGANIZER_QA.md · 4 fetch TEST qrels into ~/.acis-sealed/ (Phase 0, `acis fetch --sealed`) · 5 fill docs/reference/third_party_scores.md or leave R-02 unmeasured · 6 set configs/targets.yaml after G0.5

## Gate register
| Gate | Status | Ledger run | Decision / frozen default |
|---|---|---|---|
| G0.0 hook canary | pending | – | must pass |
| G0.1 mteb toy contract | pending | – | – |
| G0.2 data audit | pending | – | – |
| G0.3 hardware + throughput | pending | – | – |
| G0.4 model radar + pin | pending | – | – |
| G0.5 zero-shot, all 5,000 TRAIN queries | pending | – | – |
| G0.6 guard tests + physical seal | pending | – | must pass |
| G-M encoder | pending | – | seed Qwen3-Embedding-0.6B; smallest within 1.0 pt (3.0 pt if best cold pass > 2 h) |
| G1 prep, per route | pending | – | statement_like: T1+V0, 1024/1024; generic: T3+V0 |
| G2 lexical + bridge | pending | – | dense-only |
| G3 adaptation (OOF) | pending | – | frozen base |
| G-OOD arbitrary-query robustness | pending | – | reject the change |
| G4 PRF | pending | – | off |
| G5 ranker | pending | – | weighted fusion → dense-only |
| G-AB Mode A vs B (OOF) | pending | – | A iff Δ ≥ +0.5 pt (CI>0), else B |
| G6 numeric profile | pending | – | cpu-fp32 |
| G7 invariants | pending | – | must pass |
| PF preflight (judge simulation) | pending | – | must pass before release |
| EV evidence pack (N1–N8) | pending | – | unproven claims are deleted |

## Open items (docs/spec/01 §3)
O-03 hardware/wall-clock · O-04 organizer mteb version · O-05 CSV schema + MRR cutoff · O-06 fine-tuning on APPS-train allowed? · O-07 P1/Bonus data format · O-09 GPU access · Q5 non-permissive licences
Resolved by FAQ: O-01 resource use is scored (weights unknown) · O-02 any method allowed · O-08 queries "similar to the dataset" (treated as arbitrary) · JS out of scope

## Paths that do not exist yet (Phase 0/1 outputs — not errors)
`uv.lock`, `README.md`, `configs/official.yaml`, `configs/splits.lock.json`, `runs/ledger.jsonl`, `runs/hardware.json`, `src/acis/{mteb_adapter,mteb_meta,engine,eval,...}`, `tests/unit/*` beyond the smoke test, `docs/submission/{evidence,preflight_report,facts_sheet}.md`, `scripts/{demo,train,bench}/`. Seeded: `pyproject.toml`, `Makefile`, `tests/conftest.py`, `configs/{dev.yaml,official.example.yaml,targets.yaml,gates/G-M.yaml,models/qwen3-embedding-0.6b.yaml}`, `src/acis/{__init__,cli}.py`.

## Next actions
1. Unzip the kit into the repo root, commit, run the canary, then `/status` and `/phase-start 0`.
