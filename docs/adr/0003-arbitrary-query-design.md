# ADR-0003 — Arbitrary-query design: INV-15, routing v1.1, per-route instruction, G-OOD
Status: **accepted 2026-09-20** (owner instructed the kit to be updated per the audit; revert by superseding ADR).

**Context.** Hands-on queries are unknown; the PDF's own example is a one-line question, not an APPS statement. The audit found four APPS-coupled spots (one global instruction, length-based LTR routing, hand-built bridge features, untuned short-query path) plus version-intent regexes, a fixed demo query set and adaptation over-specialisation.

**Decision.** Adopt `docs/spec/10`: **INV-15** (query-agnostic behaviour), per-route instruction (`statement_like` = G1 winner, else generic T3), routing v1.1 (OOD score + feature availability; the TRAIN-query bank is used for routing only and never offsets a document score), generic feature extractors with ≈ 25 % group dropout and LTR abstention, α-interpolated adaptation (WiSE-FT-style), NL version intent as a hint (`interpreted_intent`), gate **G-OOD** relative to the frozen base, `acis eval robustness`, REG for all query-side changes, no-gold behaviour (`no_strong_match`), blind-query protocol (≥ 50 queries, once), free-text-first demo. Innovation claim N8 records the proof obligation.

**Evidence.** [EXEC] `tests/robustness/`: a generic toy engine passes 37/37; a toy engine keyed on one benchmark-style heading is caught (re-verified at merge with `ACIS_ROBUSTNESS_K=3`; the 12-document toy corpus is too small for K=10); with no engine plugged in, 27 tests skip and 10 library tests run (re-run at merge: 10 pass, 27 skip). Toy engines validate the test logic, not the real engine — the real gate is `acis eval robustness` + REG once the engine exists.

**Consequences.** ≈ 1 day of extra work in Phases 1/3/4; slightly more conservative adaptation and LTR; honest failure on no-gold queries. **Rollback**: supersede with an ADR removing INV-15/G-OOD (not recommended).
