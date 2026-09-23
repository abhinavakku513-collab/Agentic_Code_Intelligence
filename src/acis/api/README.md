# `acis.api` — contract

Spec: `docs/spec/06-contracts-testing-ops.md` §2. Rules: `.claude/rules/security.md`.

**Responsibility.** An HTTP surface over the same engine the CLI and the tests use, plus the static demo UI.

| # | Guarantee | Test |
|---|---|---|
| A-1 | **No evaluation endpoint exists**: scoring reads labels, and nothing that reads labels is reachable over a socket (spec 06 §1) | `tests/integration/test_api.py` |
| A-2 | Every field is validated by pydantic before it reaches the engine — empty and over-long queries, out-of-range `top_k`, unknown fields — and a rejection is a 422 with a reason | `tests/integration/test_api.py` |
| A-3 | A typed `AcisError` becomes the status code spec 06 §1 assigns it, with its message and context; never a traceback | `tests/integration/test_api.py` |
| A-4 | A versioned search is answered from that version alone (INV-2), and every channel is reachable | `tests/integration/test_api.py` |
| A-5 | `/healthz` reports whether the encoder may ship, so a stand-in is never mistaken for a model | `tests/integration/test_api.py` |
| A-6 | `/metrics` is Prometheus text built from the engine's own counters — no second bookkeeping to drift | `tests/integration/test_api.py` |
| A-7 | The UI references no remote asset: no CDN, no web font, nothing that needs a network on demo day (D15) | `tests/integration/test_api.py` |
| A-8 | Loopback by default; a non-loopback bind without `ACIS_API_TOKEN` is refused rather than quietly exposed | `src/acis/api/app.py::serve` |

**Non-goals.** No ranking logic (the engine's), no label access of any kind, no background jobs.
