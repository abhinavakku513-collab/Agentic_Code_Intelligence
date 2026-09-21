---
paths:
  - "src/acis/{prep,embed,lexical,features,rank,engine,data}/**"
  - "tests/**/p0/**"
---
# P0 engine guardrails — read `docs/spec/02-p0-engine.md` before changing this code
- Batch invariance (INV-3): no per-batch statistics, no cross-query normalisation, never use other queries' results.
- IDs are opaque (INV-4): never a feature; only ordering among exact-content duplicates may use the corpus ordinal.
- No feature, filter or prior may encode TRAIN-solution membership; the PMI bridge table needs support from ≥ 8 distinct problems.
- Dense text = head 768 + tail 256 tokens; BM25 and features see the full text; window MaxP only if G1 accepted it.
- Instruction strings, pooling and truncation come from `configs/models/<name>.yaml`; never hard-code them.
- Modes: RC0 is Mode B (frozen dense); Mode A returns exactly `min(top_k, N)` entries.
- Every fallback increments a counter and appears in `degradations`; strict mode aborts. Never silently drop a channel.
- INV-15 (query-agnostic): per-route instruction (only `statement_like` uses the APPS-tuned one; everything else generic T3); no closed list of templates, phrases, headers, moduli or literals gates correctness; every pattern extractor returns NaN/neutral when it does not fire and is masked (≈25 % group dropout) in LTR training; LTR abstains when < ρ of feature groups fired; no lookup or cache keyed on query text (content-hash embedding caches are fine); the TRAIN-query bank may be used for routing only and never offsets a document score.
- Query-side, adaptation, LTR or prep changes need G-OOD rows (spec 10 §5) before they are called improvements.
- Anything new needs an ablation/gate row and a ledger id before it is described as an improvement.
