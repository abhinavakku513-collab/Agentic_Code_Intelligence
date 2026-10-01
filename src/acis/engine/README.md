# `acis.cli` — contract

Spec: `docs/spec/06-contracts-testing-ops.md` §3. The CLI is a thin surface over the same library the tests call;
no command contains logic of its own.

| # | Guarantee | Test |
|---|---|---|
| L-1 | Every command resolves configs and data from the repo root (`ACIS_ROOT`), never the CWD (D16) | `tests/unit/test_core.py` |
| L-2 | Commands not yet built exit 2 naming the phase that builds them (docs/spec/07) | `tests/unit/test_cli.py` |
| L-3 | `acis fetch` is the only network path; it writes no sealed pattern into `ACIS_HOME` and re-checks the seal | `tests/security/test_seal.py` |
| L-4 | `acis doctor` writes `runs/hardware.json` and exits non-zero when the seal check fails | `tests/unit/test_cli.py` |
| L-5 | `acis eval official` refuses to run unless `HF_HOME` points at the physical seal (D19) | `tests/security/test_seal.py` |
| L-6 | Typed errors print as `code: message` and exit 1; they never raise a traceback at the user | `tests/unit/test_cli.py` |
| L-8 | `acis fetch --models` downloads safetensors only — never a pickle checkpoint — resolves a moving revision to a commit, and refuses a non-permissive model unless it is asked for as reference (D4, docs/DESIGN_RULES.md) | `tests/unit/test_model_fetch.py` |
| L-7 | `acis search` returns exactly `min(top_k, N)` ranked hits, prints evidence re-read from the content store (INV-1), and labels a stand-in encoder as one | `tests/integration/test_search_cli.py` |

| L-9 | `acis ingest/versions/diff/activate/rollback` drive the same engine methods the tests call, and `acis search --repo/--version` answers from one pinned snapshot | `tests/integration/test_p1_versions.py`, `tests/integration/test_search_cli.py` |

**Non-goals.** No evaluation endpoint is ever exposed over the network API (docs/spec/06 §1); `acis serve` is Track B3.
