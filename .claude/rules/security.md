---
paths:
  - "src/acis/ingest/**"
  - "src/acis/sec/**"
  - "src/acis/api/**"
  - "**/sandbox*"
---
# Security guardrails — read `docs/spec/05-agent-security-failure.md` §2 first
- Content is parsed, never executed (INV-5): no import/exec/eval/compile of snippets; parse only in sandboxed workers with rlimits, timeouts and no network.
- One `safe_path()` for every path; archives are streamed into the CAS, never extracted; enforce entry/ratio/size limits with streaming byte counters.
- Git via subprocess with hooks/fsmonitor/symlinks disabled and no checkout, submodules or LFS.
- No pickle/marshal, unsafe `yaml`, `torch.load` without `weights_only=True`, `shell=True`, or `trust_remote_code=True`. Safetensors only.
- API binds loopback; non-loopback requires a bearer token; validate every input with pydantic; logs carry hashes/lengths/timings only — never query text, code or secrets.
- Add an adversarial fixture with every new parser or endpoint.
