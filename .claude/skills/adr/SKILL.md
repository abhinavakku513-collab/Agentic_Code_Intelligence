---
name: adr
description: Record an architecture decision — a deviation from CLAUDE.md/spec, a dependency addition, or a gate override — with mandatory evidence.
argument-hint: [short-title]
disable-model-invocation: true
---
Create `docs/adr/NNNN-$ARGUMENTS.md` (next free number) with: **Status** (proposed/accepted/superseded), **Context** (what forced this; cite the spec section), **Decision**, **Evidence** (`[ledger:<id>]`, test id or measurement — required), **Consequences** (accuracy, resource use, risk), **Rollback**. Then list the exact CLAUDE.md/spec lines that must change and ask the owner to confirm before editing them. Dependency additions also need: licence, size, hash, and why nothing existing works.
