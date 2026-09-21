# What Claude Code hands to the owner (PPT and demo video are OWNER-made, not Claude Code tasks)
Claude Code's obligation stops at working software, reproducible numbers and evidence files. It does **not** build slides, record or script a video, or maintain a PPT outline.

## Files Claude Code produces (all generated from the ledger by `acis report`, never typed by hand)
| File | Use for the owner's slides/video |
|---|---|
| `docs/submission/facts_sheet.md` | ten one-line facts, each with a `[ledger:<id>]` |
| `docs/submission/scorecard.md/.json` | params, model MB, peak RSS, cold/warm time, latency, GPU = none |
| `docs/submission/ladder_table.md` | B0–B10 with CIs |
| `docs/submission/robustness_table.md` | G-OOD results (arbitrary-query behaviour) |
| `docs/submission/figures/*.png` | accuracy-vs-cold-time-vs-size Pareto, stage-latency bars, P1 update-time curve |
| `docs/submission/p1_report.md`, `bonus_report.md` | isolation, incremental ≡ scratch, lineage metrics |
| `docs/submission/known_limitations.md`, `licences_provenance.md` | honest limits, licences |
| `docs/submission/demo_runbook.md` + `make demo` | offline, pre-warmed, free-text queries first; commit-stream and rollback steps |
Applied in kit v1.1: `/evidence-pack` no longer generates a `PPT_OUTLINE.md`, `/preflight` skips C12, and the PPT items are gone from Phase 6 (docs/spec/07, 08 §5). Checks C1–C11 and C13 remain.

## Owner's rehearsal checklist
`make demo` on the machine you will present from, network off · type **your own** unseen queries (long statement, one-line question, an identifier, a code fragment) · show cold vs warm time · replay the commit stream, then roll back · have a recorded fallback clip · never quote a number that has no ledger id.
