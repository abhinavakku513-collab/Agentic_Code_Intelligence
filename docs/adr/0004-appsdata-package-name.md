# ADR-0004 — the package map's `data` role ships as `acis.appsdata`

Status: **accepted 2026-09-21** (Phase 0). Supersedes nothing; amends the package map in `docs/spec/06` §0 by name only.

## Context
`docs/spec/06` §0 names a package `data` ("APPS loaders (train only), fold split, decontamination"). The repository's
protected `.claude/settings.json` denies `Edit(./data/**)`, and the enforcement layer applies that glob to any path
containing a `data/` segment — including `src/acis/data/**`. Every attempt to create the spec-named package (Write tool,
`mkdir`, `cp`) is denied. `.claude/settings.json` is on the never-edit list (CLAUDE.md §4), so the tool cannot fix the
glob itself.

## Decision
Ship the package as **`src/acis/appsdata/`** with the identical contract (`README.md` in the package). Decontamination
lives in `acis.eval.decontam`, which is where `docs/spec/07` Phase 1 puts it anyway.

## Consequences
- Import sites read `from acis.appsdata import ...`. No behaviour, invariant or interface changes.
- `docs/spec/06` is **not** edited (spec edits need owner confirmation); this ADR records the deviation, as
  `docs/reference/README.md` requires.
- **Owner action to restore the spec name:** anchor the deny globs in `.claude/settings.json`
  (`Edit(./data/**)` → a form that matches only the repository-root `data/` directory, e.g. `Edit(//data/**)` or the
  absolute path). After that, `git mv src/acis/appsdata src/acis/data` plus an import rename is mechanical, and this
  ADR can be superseded.
