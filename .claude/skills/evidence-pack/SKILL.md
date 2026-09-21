---
name: evidence-pack
description: Build the ledger-backed evidence files for the owner (facts sheet, scorecard, ladder/robustness tables, figures, P1/Bonus reports, limitations, licences) and the innovation-claim table N1–N8; delete any claim whose proof obligation is not met. No PPT outline.
disable-model-invocation: true
---
1. Read `docs/spec/08-novelty-evidence-submission.md` §2 and `docs/submission/OWNER_HANDOFF.md`, and the ledger (read-only).
2. Run `uv run acis report --claims --out docs/submission/evidence.md` for claims N1–N8 (proof metric, value, 95 % CI, threshold, `[ledger:<run_id>]`, verdict PROVEN / NOT PROVEN / NOT ADOPTED) and `uv run acis report` for the owner files: `facts_sheet.md`, `scorecard.md/.json`, `ladder_table.md`, `robustness_table.md`, `figures/*.png` (accuracy–cold-time–size Pareto, stage-latency bars, P1 update-time curve), `p1_report.md`, `bonus_report.md`, `known_limitations.md`, `licences_provenance.md`, `demo_runbook.md`.
3. For every NOT PROVEN claim, remove its wording from the README, `docs/submission/*` and demo scripts (grep); NOT ADOPTED claims (e.g. the agent) read "evaluated, not adopted". Give the owner the list of removed claims so slides and video match.
4. Scan README and `docs/submission/*` for any number without a ledger id and fail if found. Do not create slides, video scripts or a PPT outline (owner-made).
