# ACIS kit v1.1 patch — what it is and how to apply it
Unzip over the repo root. **The owner applies `.claude/settings.json` and `.claude/hooks/guard.py` by hand** (both are protected from Claude by design).

| File | Purpose | Tested |
|---|---|---|
| `.claude/settings.json` | hooks switched to the documented single-string `command` form; **`ask` on every TEST-touching command**; read-only web allow-list for the model radar; sealed-dir deny outside the repo; longer foreground timeouts | JSON validated; hook strings executed |
| `.claude/hooks/guard.py` | v1.1: fixes 11 failures of v1.0 found by execution (ledger-write bypasses, 3 false positives, HF-cache path, recursive-search ask) | `tests/security/test_guard_hook.py`: 40 pass + 3 documented gaps (xfail) |
| `tests/security/test_guard_hook.py` | behavioural contract for the hook | executed against v1.0 (11 fail) and v1.1 (all pass) |
| `tests/contract/test_score_ties.py`, `test_dispatch_matrix.py` | G0.1 seeds: tie collapse (V-08), rank-derived scores exact, dispatch matrix, source-anchor drift detector | 25 pass on mteb 2.0.5, 2.12.30, 2.21.0 (torch stubbed in my sandbox) |
| `scripts/run_detached.sh`, `job_status.sh` | long jobs outside the tool-call timeout | executed: start, running, exit code, double-start guard, name validation |
| `docs/official/ORGANIZER_QA.md` | 9 organizer questions with decision impact | – |
| `docs/GPU_HANDOFF.md` | Colab/Kaggle protocol + GPU-hour estimates [E] | – |
| `docs/TRIAGE.md` | two tracks, shippable floors, cut order | – |
| `docs/adr/0001-dependency-allowlist.md` | pre-approved dependencies (licences [MEM], verify) | – |
| `docs/spec/09-v1.1-plan-power-seal-ops-demo.md` | **proposal** (needs `/adr 0002`): phase order, gate power, physical seal, spec fixes, hands-on runbook | simulation for §2 |
| `docs/spec/10-arbitrary-queries.md` | **proposal** (needs `/adr 0003`): INV-15, per-route instruction, OOD-guarded LTR, gate G-OOD, blind-query protocol | see next row |
| `tests/robustness/` (`perturb.py`, `test_query_agnostic.py`, two toy engines) | generic perturbation library + label-free robustness properties; engine plugs in via `acis.robust_hook:search` | generic toy: 37 pass; template-coupled toy: caught; no engine: 27 skip, 10 run |
| `docs/submission/OWNER_HANDOFF.md` | what Claude Code hands to you; **PPT and demo video are yours** | – |

## Apply order
1. Copy files. 2. Read `docs/spec/09`; accept/reject it via `/adr 0002` (CLAUDE.md is **not** edited by this patch). 3. Canary (below). 4. `uv run pytest tests/security tests/contract -q`.

## Hook canary (do this once, and after every Claude Code upgrade) — a green `/hooks` listing does not prove the hooks run
1. In your own shell: `mkdir -p data/sealed && echo canary > data/sealed/canary.txt`.
2. In Claude Code ask: "Read data/sealed/canary.txt". Expected: a message starting **"ACIS guard blocked this call"**. Also ask it to `cat` the file via Bash — expected: blocked.
3. Ask it to write `src/acis/x.py` containing `eval(x)` — expected: blocked.
4. If any of these is **not** blocked, the hooks are not executing: run `claude --debug`, re-check `/hooks`, and fall back to the other hook wiring form before doing anything else.
5. Delete the canary file.

## Not fixed by this patch (owner decisions)
Compute answers (cores/RAM, GPU access, who runs the cold pass), organizer answers, CLAUDE.md trimming, the phase reorder (spec 09 §1), removing PPT-outline items from `/evidence-pack`, `/preflight` C12 and Phase 11 (OWNER_HANDOFF.md).
