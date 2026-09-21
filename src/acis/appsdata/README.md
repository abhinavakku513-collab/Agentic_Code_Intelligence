# `acis.appsdata` — contract

Spec: `docs/spec/06-contracts-testing-ops.md` §0 (the package map's `data` role) and `docs/spec/03` §5 (seal, splits).
ADR: `docs/adr/0004-appsdata-package-name.md` explains why the directory is not called `data`.

**Responsibility.** The only network path in ACIS (`acis fetch`), the allow-list that keeps held-out labels out of the
dev environment, the dev-split loaders (corpus, queries, qrels), deterministic fold assignment, and the G0.2 dataset
audit.

| # | Guarantee | Test |
|---|---|---|
| D-1 | `acis fetch` downloads only allow-listed, non-sealed files; sealed patterns always lose | `tests/security/test_seal.py` |
| D-2 | After any fetch, `assert_seal()` proves no held-out label file exists under `ACIS_HOME` (G0.6) | `tests/security/test_seal.py` |
| D-3 | `load_qrels`/`load_queries` accept the dev split only; any other split raises `SealedDataAccess` (INV-8) | `tests/security/test_seal.py` |
| D-4 | The corpus is returned in corpus order, with all 8,765 documents (labelled and unlabelled) | `tests/integration/test_apps_loaders.py` |
| D-5 | `partition` metadata never leaves this package except through `dataset_audit()` | `tests/unit/test_no_partition_leak.py` |
| D-6 | Folds come from `sha256(query_text) mod k` — never from a query id (INV-4) | `tests/unit/test_hashing.py` |
| D-7 | Every fetched asset is checksummed and re-verifiable (`verify_manifest`) | `tests/integration/test_apps_loaders.py` |

**Verified layout** (pinned revision `f22508f96b7a…`): `corpus/corpus-*.parquet` (8,765 rows), `queries/queries-*.parquet`
(8,765 rows), `data/train-*.parquet` (5,000 qrels), and one held-out qrels file that only the owner fetches, into
`~/.acis-sealed/hf`.

**Non-goals.** No ranking, no embedding, no writes into the repository tree.
