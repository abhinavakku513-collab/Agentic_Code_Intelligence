# 05 · Agent layer, security, failure engineering

Authority: subordinate to `CLAUDE.md` (D13, INV-5/7/12/13). The agent is optional and built last; security and failure handling are built in from Phase 0. **Cut-first items** when time is short (`docs/TRIAGE.md`; none is a Samsung criterion): S2 sandbox tier, cosign signing, reproducible container, ≥ 1 h fuzzing per parser, custom semgrep rules, LLM planner. Keep: parse-only workers, limits, no-exec, safe paths.

## 1. Agentic layer (D13)
**Definition**: a bounded, deterministic, read-only investigation controller. When first-pass evidence is weak it issues extra retrieval/inspection calls, re-ranks through the *same* ranker, verifies and stops. It never writes code, answers questions, or ranks by itself; it is absent from the scored path (`agent_calls == 0` is asserted by the adapter and `verify-submission`).
| Aspect | Specification |
|---|---|
| Triggers (interactive only) | confidence == low · `investigate=force` · identifier-looking query with zero exact hits · version-intent query needing a lineage lookup |
| Tools (allow-listed, snapshot-pinned, ID-based, schema-validated) | `search(query\|view, k)` · `search_exact(term)` · `neighbors(unit_id, k)` · `read_unit(unit_id)` · `lineage(unit_id)` · `compare_versions(unit_id, a, b)` · `diff_units(a, b)`. No shell, network, write or arbitrary-file tool |
| Actions (deterministic order) | (1) re-query with segment views (statement-only, io-only) (2) dense neighbours of the current top-1 (pseudo-relevance expansion) (3) lineage/version expansion when a version intent exists (4) optional local LLM decomposition into ≤ 3 sub-queries from a fixed JSON schema |
| State | original + normalised query, route, snapshot_id, `candidates{unit_id: score, provenance}`, inspected units, queries tried, evidence, confidence, remaining budget — no chat history |
| Observations | typed tool results; every added candidate carries `via` provenance and re-enters features → ranker |
| Budgets | 3 iterations · 12 tool calls · 30 units read · 50 new candidates · wall clock 3 s (deterministic) / 20 s (with LLM); enforced by the controller, never by the LLM |
| Stop | confidence high · no margin gain ≥ ε in an iteration · budget exhausted · top-1 verified. On timeout/exception return the first-pass ranking with `investigation.status="aborted"` |
| Verification | final ids ⊆ ids seen in tool results, re-read from CAS and hash-checked (INV-1); UI shows coarse progress only ("Searching… Inspecting… Verifying…"), never reasoning |
| LLM (optional) | local GGUF via llama.cpp (candidate Qwen3-4B-Instruct-2507, licence/throughput [U] → verify), lazy-loaded, CPU-only; snippet text only inside delimited, capped data blocks; schema-constrained output; unknown tools/args dropped (INV-12). System fully functional without it |
| Benefit gate | default-on only if Recovery@10 (first-pass misses recovered) on hard dev queries has paired CI > 0 within the latency budget and false-escalation rate is acceptable; else `investigate=force` only. Symbol/call-graph tools are **not** built (D14) |

## 2. Threat model (assets: host integrity, snapshot integrity, evaluation integrity, released artifacts)
Actors: malicious snippet/archive/git author · malicious API client · compromised dependency or model publisher · curious co-tenant. Boundaries: network→API · API→ingest workers · repository content→parsers/LLM · model & dependency artifacts→runtime · release artifacts→users.
| STRIDE | Threat | Control (test in CI) |
|---|---|---|
| Spoofing | unauthenticated remote use | bind `127.0.0.1`; non-loopback requires a bearer token (`hmac.compare_digest`) or the server refuses to start |
| Tampering | modified weights/index/cache | per-file SHA-256 manifests, `VALID` sentinels, checksum-on-read, signed release |
| Repudiation | untraceable evaluation runs | hash-chained ledger, manifests with git SHA + dirty flag |
| Information disclosure | cross-repo leakage, secrets/queries in logs | repo namespacing, 0700 dirs, worker sees only its job dir, secret-file exclusion, logs = hashes/lengths/timings only |
| Denial of service | zip bombs, parser DoS, request floods | limits below, worker rlimits/timeouts, bounded queues, 429 + Retry-After |
| Elevation | code execution via content or deserialisation | INV-5 (parse-only), sandboxed workers, banned pickle/`eval`/`exec`/`shell=True`/unsafe `yaml`/`torch.load`, safetensors only |
**Sandbox tiers** (capability-detected at startup): **S2** Linux user/mount/pid/net namespaces (bubblewrap-style), read-only input bind, tmpfs work dir, seccomp allow-list, rlimits (AS, CPU, NOFILE, FSIZE, NPROC), no network · **S1** portable spawn-ed subprocess, scrubbed env, `python -I -S`, POSIX rlimits (Windows Job Object), wall-clock timeout, temp cwd · **S0** in-process only for the public MTEB corpus with size caps. Untrusted sources require ≥ S1, else ingestion refuses (`--allow-unsandboxed` for local dev, with a warning).
**Ingestion limits**: archives streamed into CAS, never extracted; entries ≤ 50k, declared total ≤ 1 GiB, ratio ≤ 100:1, per-file ≤ 50 MB **plus** streaming counters of actual bytes (declared sizes can lie); no nested archives; one `safe_path()` everywhere (rejects `..`, absolute, drive letters, NUL, backslash tricks, reserved names; NFC-normalises; detects case collisions); symlinks/devices never followed; max file 1 MB (head/tail sample flagged); parser worker 5 s + memory rlimit.
**Git hardening**: subprocess `git -c core.hooksPath=/dev/null -c core.fsmonitor=false -c core.symlinks=false`, `GIT_TERMINAL_PROMPT=0`, `GIT_CONFIG_NOSYSTEM=1`; read objects (`ls-tree`, `cat-file --batch`), no checkout, no submodule recursion, no LFS smudge; clone only over HTTPS from an allow-list, no private/loopback IPs.
**Supply chain**: hash-pinned `uv.lock` + `--require-hashes`; minimal dependency set (no tree-sitter, no JS toolchain); model = pinned HF commit SHA + per-file SHA-256, safetensors, `trust_remote_code=False`; offline at runtime; ONNX/int8 conversions done offline and hash-recorded; `pip-audit` + `osv-scanner` + licence scan + CycloneDX SBOM in CI; digest-pinned non-root read-only container with `--network none` for serving; SOURCE_DATE_EPOCH reproducible builds; release checksums + signature (cosign/minisign). **Secrets**: none required anywhere; gitleaks in CI; API token from file/env, never logged. **Secure-SDLC gates (all block merge/release)**: ruff, mypy `--strict` on core, bandit, semgrep (bans listed constructs), unit/property/fuzz (archive, path, selector, JSONL parsers), injection corpus, fault injection, dependency and secret scans, SBOM, signed artifacts.
**Prompt/code injection**: snippet text is data (INV-12); the deterministic ranking path has no instruction channel; keyword-stuffing is damped by BM25 saturation + length normalisation + a `stuffing_score` feature + token caps; an injection corpus ("ignore previous instructions" in comments/ids) must not change tool calls, rankings' provenance or evidence.

## 3. Failure and degradation matrix — Detection → Recovery/Fallback → User-visible → Metric
Every failure is detected, contained (never corrupts a snapshot or crashes the service), counted, and listed in `diagnostics.degradations`; `strict` turns any degradation into a hard failure.
| Failure | Detection | Recovery / fallback | User-visible | Metric |
|---|---|---|---|---|
| Invalid input / malformed JSON/UTF-8 | schema validation, NUL/entropy sniff | 400 problem+json; per-file skip/replace+flag | error body with code | `acis_input_rejected_total` |
| Empty / ambiguous query | length check; router uncertainty | empty ⇒ `InvalidInput`; ambiguous ⇒ all channels on, grouped alternatives | 400 / normal results | `acis_query_ambiguous_total` |
| Corrupted repo/archive, ZIP bomb, path traversal, symlink | reader error, pre-scan + streaming counters, `safe_path` | job FAILED, nothing extracted, temp removed | job error reason | `acis_ingest_reject_total{reason}` |
| Huge file / repo | size/count limits | head+tail sample flagged / streaming ingest, bounded workers, `allow_partial` | `coverage` in job report | `acis_limit_hit_total{kind}` |
| Unsupported / binary file | extension + sniff | skipped, counted | in report | `acis_skipped_total` |
| Parser failure / malformed code / Py2 | `ast` exception, timeout | tolerant tokeniser → raw text; `parse_ok=false`; dense+lexical still work | none | `acis_parse_fail_total` |
| Duplicate / near-duplicate code | exact hash; MinHash | exact ⇒ adjacent (P0 keeps all docs); near ⇒ cluster feature, grouped in lineage view | `also_at[]` | `acis_dup_total` |
| Embedding failure / NaN | exception, `isfinite` | retry at half batch → per-unit isolation → `dense_missing` (stays lexical); strict ⇒ abort | degradation flag | `acis_embed_fail_total` |
| Model unavailable / corrupt weights | startup checksum | fail closed; non-strict serving falls back to lexical | `/readyz` false or banner | `acis_model_unavailable` |
| Out of memory | allocation error, RSS watchdog | halve token budget → shorter max length → int8 (only if G6 passed) → lexical-only | degradation flag | `acis_oom_total` |
| Corrupted index / segment | hash/size mismatch on open | quarantine, serve PREV, rebuild from CAS | banner | `acis_index_corrupt_total` |
| Stale index (model/config/tokenizer changed) | fingerprint mismatch in manifest | snapshot ineligible, rebuild scheduled; `index_stale` unless `allow_stale` | error/banner | `acis_stale_total` |
| Interrupted indexing / partial update | `.tmp-*`, journal, no `VALID` | startup recovery; nothing partial visible | none | `acis_recovery_total` |
| Version conflict | single-writer lock, `expected_active` | 409, client retries | 409 body | `acis_conflict_total` |
| Cache corruption/poisoning | checksum on read, keys include snapshot+config | drop entry, recompute | none | `acis_cache_corrupt_total` |
| Disk full | pre-flight estimate + write errors | abort build cleanly, active untouched | job error | `acis_disk_full_total` |
| No relevant result / low confidence | low margin/scores | return best evidence with `no_strong_match=true`; interactive may investigate; scored path never truncates | flag | `acis_low_conf_total` |
| Ranker (LTR) failure | exception | weighted fusion → dense-only → lexical-only | degradation flag | `acis_fallback_total{stage}` |
| Agent failure/timeout | budget/exception | return first-pass result, `status="aborted"` | status field | `acis_agent_abort_total` |
| Resource exhaustion by clients | queue depth, latency | bounded queues, per-request timeouts, 429 + Retry-After, payload caps | 429 | `acis_shed_total` |
| Weak host (no AVX-512 / <8 GB) | ISA + RAM probe at startup | fp32 reference everywhere; bf16/int8 only when supported+gated; tiers (spec 02 §7) | tier in `/v1/diagnostics` | `acis_tier` gauge |
| Clock skew / ambiguous `as_of` | commit timestamps | resolve in UTC, report resolved version, ambiguous ⇒ 409 with candidates | 409 | – |
| Adversarial content / secrets | stuffing detector, injection corpus, filename+entropy patterns | data-only handling, caps, exclusion by default, never logged/returned | flag | `acis_adversarial_total` |
