# ADR-0001 — Pre-approved dependency allowlist (so `uv add` does not stall every session)
Status: **accepted 2026-09-20** (owner instruction to apply the audit; Claude may add these without a per-package ADR). Licences are from memory **[MEM]** — Phase 0 runs `uv run pip-licenses` and records the real table here.

| Group | Packages | Licence [MEM] | Note |
|---|---|---|---|
| Core | numpy, scipy, pydantic, structlog, orjson | BSD/MIT/Apache | |
| P0 | torch (CPU wheel), transformers, safetensors, tokenizers, sentence-transformers (parity tests only), lightgbm, bm25s, mteb (pinned), datasets, pytrec-eval-terrier | BSD/Apache/MIT | `trust_remote_code=False` always |
| Adaptation (extra `train`, dev machine/GPU only) | peft, accelerate | Apache-2.0 | never imported at inference |
| Speed (optional extra) | onnxruntime | MIT | only after G6 |
| Data hygiene | datasketch (MinHash) | MIT | decontamination |
| Service | fastapi, uvicorn, prometheus-client | MIT/BSD/Apache | loopback by default |
| Dev/test | pytest, hypothesis, ruff, mypy, bandit, pip-audit, pip-licenses | MIT/MPL/Apache | Hypothesis is MPL-2.0 (test-only) |
Anything not listed → `/adr` with licence, size, hash and "why nothing existing works" (CLAUDE.md §4).

**Amendment (kit v1.1)** — also pre-approved: `pyyaml` (config), `zstandard` (CAS blobs), `rapidfuzz` (token similarity for lineage), `huggingface-hub` (allow-listed asset downloads and `list_repo_files`), `scikit-learn` (isotonic confidence; already a dependency of mteb). The CLI uses stdlib `argparse` (no `typer`). Anything else → `/adr`.
