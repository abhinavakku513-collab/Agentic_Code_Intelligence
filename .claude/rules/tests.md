---
paths:
  - "tests/**"
---
# Testing guardrails — read `docs/spec/06-contracts-testing-ops.md` §5 first
- Layout: `tests/{unit,property,metamorphic,contract,integration,security,chaos,perf}`; markers `slow`, `gpu`, `network` are excluded from `make test-fast`.
- Tests come first and must fail for the right reason before implementation. Use fixtures, never TEST data; dev data only through `acis.data` loaders.
- Property tests (Hypothesis) for INV-2/3/4 and score monotonicity; every failure-matrix row gets a fault-injection or fixture test; every parser gets an adversarial and a fuzz test.
- Robustness (INV-15): label-free properties in `tests/robustness` plug the real engine in through `acis.robust_hook:search`; a template-coupled engine must be caught. Guard/seal contract: `tests/security/test_guard_hook.py` + `test_guard_extra.py` (gap rows stay `xfail(strict)`; promote a row only when the hook truly catches it).
- Contract tests run on the mteb compatibility matrix (`uv run --with mteb==<v> pytest tests/contract -q`).
- Assertions on numbers use ledger-backed tolerances; no snapshot of accuracy figures in code.
