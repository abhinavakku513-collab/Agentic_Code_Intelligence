# `acis.obs` — contract

Spec: `docs/spec/06-contracts-testing-ops.md` §7.

**Responsibility.** Structured JSON-line logs, degradation/fallback counters, and the data behind `/metrics`,
`/healthz` and `acis report`.

| # | Guarantee | Test |
|---|---|---|
| O-1 | Logs carry hashes, lengths and timings — never query text or snippet bodies unless `ACIS_LOG_TEXT=1` | `tests/security/test_log_redaction.py` |
| O-2 | Every fallback increments a counter **and** appears in `diagnostics.degradations` (INV-7) | `tests/unit/test_counters.py` |
| O-3 | In strict mode a degradation raises `StrictViolation` instead of being recorded (official runs) | `tests/unit/test_counters.py` |

**Non-goals.** No Prometheus/Grafana server is required; metrics are exposed as text on the local API only.
