---
name: repro-check
description: Verify that a ledger run reproduces — config hash, lockfile/model/asset hashes, clean tree, and a rerun within tolerance.
argument-hint: [run_id]
disable-model-invocation: true
---
Run `uv run acis eval repro $ARGUMENTS` and report each check: config hash, lockfile, model fingerprint, dataset revision, dirty flag, metric delta vs the ledger, top-10 ranking overlap. For release candidates the rerun must be cold and strict. Fail loudly; never "fix" a mismatch by editing the ledger.
