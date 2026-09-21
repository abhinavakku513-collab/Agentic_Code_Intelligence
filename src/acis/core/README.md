# `acis.core` — contract

Spec: `docs/spec/06-contracts-testing-ops.md` §1 and §4. Rules file: none (this package has no ranking logic).

**Responsibility.** Value types (`Snippet`, `Unit`, `Hit`, `SearchRequest`, `SearchResponse`, `Snapshot`), frozen
configuration and its hash, deterministic hashing, repo-root/`ACIS_HOME` path resolution, numeric profiles and the
typed error hierarchy.

**Guarantees**

| # | Guarantee | Test |
|---|---|---|
| C-1 | No heavy imports: `acis.core` never imports torch, mteb, datasets or transformers | `tests/unit/test_core_imports.py` |
| C-2 | `config_hash` covers ranking-relevant keys only; `threads`/`port`/`batch_size`-style knobs never change it | `tests/unit/test_config.py`, `tests/property/test_config_hash.py` |
| C-3 | Hashing is stable across processes and insensitive to dict ordering; `fold_of` uses the query **text**, never an id (INV-4) | `tests/unit/test_hashing.py` |
| C-4 | Config and data paths resolve from the repo root (`ACIS_ROOT`), never the CWD (D16) | `tests/unit/test_paths.py` |
| C-5 | Every engine error subclasses `AcisError` and carries its HTTP status (docs/spec/06 §1) | `tests/unit/test_errors.py` |

**Non-goals.** No I/O beyond reading a YAML config; no ranking, no snapshot storage, no network.
