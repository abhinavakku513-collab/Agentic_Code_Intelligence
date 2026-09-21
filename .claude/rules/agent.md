---
paths:
  - "src/acis/agent/**"
---
# Agent-layer guardrails — read `docs/spec/05-agent-security-failure.md` §1 first
- Optional, interactive-only, never in the scored path (INV-13): the adapter asserts `agent_calls == 0`.
- Tools are read-only, allow-listed, snapshot-pinned and ID-based; there is no shell, network, write or arbitrary-file tool.
- Budgets (3 iterations / 12 tool calls / 30 units read / 3 s) are enforced by the controller, never by an LLM; on any failure return the first-pass ranking.
- Expansion adds candidates that re-enter the same ranker; the agent never ranks, generates code, or fabricates evidence (INV-1).
- Snippet text reaches an LLM only as delimited, length-capped data with schema-constrained output (INV-12). The system must work with no LLM.
