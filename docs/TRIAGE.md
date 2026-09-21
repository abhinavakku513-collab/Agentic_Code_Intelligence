# Schedule-independent triage: tracks, shippable floors, cut lines
(The kit has no deadline. Put the real date in `docs/STATUS.md` and work backwards from these floors.)

## Two tracks (git worktrees; the `AcisEngine` interface in spec 06 §1 is frozen at the end of Phase 1)
- **Track A — accuracy (P0):** harness → dense bake-off → adaptation → LTR/hybrid → freeze → RC1. Long jobs run detached (`scripts/run_detached.sh`); while they run, Track B works.
- **Track B — systems + demo (P1, Bonus, UI):** starts when the Phase 1 gate freezes `AcisEngine` (parallel from Phase 2), consumes it with *whatever encoder is current* (a small stand-in until RC0); B1 store/P1 → B2 lineage → B3 API/UI/`make demo` → B4 agent (last, optional).
A phase gate still applies inside each track; the cross-track rule is only "do not change the frozen interface without an ADR".

## Shippable floors (there must always be something submittable)
| Floor | Contents | Reached at |
|---|---|---|
| F0 | harness green, no TEST touched | end of Phase 1 |
| **F1** | **RC0 = best frozen dense encoder, Mode B, verify-submission PASS, README run steps** (1 TEST touch; the owner starts the official run) | end of Phase 2 |
| F2 | + adapted encoder and/or LTR (RC1, Mode A/B decided) | end of Phase 5 |
| F3 | + P1 store + demo runbook | Track B (B1 + B3) |
| F4 | + Bonus lineage + evidence pack (PPT/video are owner-made) | Track B (B2) + Phase 6 |

## Cut order when behind (top = cut first; none of these is a Samsung criterion)
1. Agent layer (B4) → ship "evaluated, not adopted". 2. Sandbox tier S2 (bubblewrap/seccomp), cosign/minisign signing, SOURCE_DATE_EPOCH reproducible container.
3. Fuzz ≥ 1 h/parser → 10 min smoke. 4. semgrep custom rules (keep the guard hook + ruff/bandit). 5. PMI bridge table (keep I/O-contract features). 6. PRF. 7. Second encoder (E-SECOND).
**Never cut:** metric parity + tie test, cold honest run, strict mode, scorecard, verify-submission, P1 isolation + atomic activation, `make demo`.
