# ADR-0005 — defer G0.4 (model radar) and G0.5 (zero-shot on all dev queries) to Phase 2

Status: **proposed — needs owner confirmation** (drafted 2026-09-23 at the Phase 1 gate; `docs/STATUS.md` records
it as an outstanding owner action). Everything else in Phase 0 is decided: G0.0 mechanism verified, G0.1, G0.2,
G0.3 and G0.6 pass with evidence in `docs/PHASE0_REPORT.md`.

## Context

`docs/spec/07` lists six gates for Phase 0. Four were closed during this work. Two were not:

* **G0.4 — model radar and pin.** Re-read the live MTEB(Code)/CoIR leaderboards, audit licence / remote-code /
  safetensors for each candidate ≤ ~1B parameters, pin commit SHAs and per-file SHA-256s, and check parity against
  sentence-transformers.
* **G0.5 — frozen zero-shot scores on all 5,000 dev queries** for the shortlist.

Both need encoder weights downloaded and a multi-hour CPU pass **per candidate** (`docs/spec/02` §7 projects
1.4–8 h for a single 0.6B encoder on the reference host; this development host is T-min and slower). Neither
produces an input to any Phase 1 acceptance row: Phase 1's criteria are about the harness, and the dense channel it
exercises is a model-free stand-in (`acis.embed.hashing`, `submission_capable=False`, refused by a strict run).

Both feed exactly one decision: **gate G-M**, the encoder choice, which `docs/spec/07` places in **Phase 2**.

## Decision

Close Phase 0 with G0.4 and G0.5 **deferred into Phase 2**, where they become the first work of the encoder
bake-off rather than a separate gate. Phase 2 cannot start without them, so nothing is skipped — the deferral moves
the work next to the decision it serves instead of blocking a harness phase on it.

## Consequences

* `configs/models/qwen3-embedding-0.6b.yaml` keeps `base_commit: null` until Phase 2 pins it. No model is loaded,
  downloaded or pinned before then, and no accuracy number for any encoder exists anywhere in the repository.
* `configs/targets.yaml` stays unset: the owner fills it after seeing G0.5's numbers, as `docs/STATUS.md` says.
* The declared host is **T-min** (`runs/hardware.json`: 8 cores, 9.7 GB, no GPU). G0.3 is closed *for this host*;
  the throughput and projected-cold-pass figures that G-M needs must be measured on whichever host will run the
  official pass, which is still an undeclared owner input.
* **Risk accepted:** the third-party figures in `docs/spec/01` R-02 (Qwen3-0.6B ≈ 75 NDCG@10 and the rest) remain
  unverified by us for longer. They are calibration only and `docs/reference/third_party_scores.md` already marks
  them unverified; no ACIS number depends on them. If the bake-off contradicts them, that is a finding, not a
  regression.
* **Not deferred:** the licence work G0.4 shares with ADR-0001 is already done for *dependencies*
  (`docs/reference/dependency_licences.md`: 108 packages, no GPL/AGPL, no CC-BY-NC). Only the *model* licence audit
  moves.

## Alternatives considered

* **Run them now.** Several multi-hour CPU passes on a T-min host, producing numbers that G-M would have to
  re-measure on the real reference host anyway, to unblock a phase whose acceptance rows do not use them.
* **Drop the gates.** Rejected: G-M is a scored decision (resource use is part of the evaluation, FAQ/D2) and
  needs the radar and the zero-shot table as its evidence. Deferring is not dropping.
