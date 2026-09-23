# ACIS Makefile — thin wrappers; logic lives in the `acis` CLI and scripts/. Phase-0 SEED: targets that need code from a later phase fail with a clear message.
PY ?= uv run
FAST_DIRS := $(wildcard tests/unit tests/security tests/contract tests/robustness)
.PHONY: setup fetch fetch-models doctor lint typecheck test test-fast test-security robustness bench eval-dev gate rc-official reproduce reproduce-cache demo preflight evidence

setup:            ; uv sync
fetch:            ; $(PY) acis fetch
# OWNER, once, on a networked machine: downloads safetensors + tokeniser for the carded models and pins them (G0.4).
fetch-models:     ; @test -n "$(MODELS)" || { echo "usage: make fetch-models MODELS=qwen3-embedding-0.6b[,...]"; exit 64; }; $(PY) acis fetch --models $(MODELS) --pin
doctor:           ; $(PY) acis doctor
lint:             ; $(PY) ruff format --check . && $(PY) ruff check .
typecheck:
	@dirs="$$(ls -d src/acis/core src/acis/eval src/acis/engine src/acis/store 2>/dev/null)"; \
	if [ -n "$$dirs" ]; then $(PY) mypy --strict $$dirs; else echo "typecheck: no strict packages yet"; fi
test:             ; $(PY) pytest -q
test-fast:        ; $(PY) pytest -q -m "not slow and not gpu and not network" $(FAST_DIRS)
test-security:    ; $(PY) pytest -q tests/security
robustness:       ; $(PY) pytest -q tests/robustness
bench:            ; @test -d scripts/bench && $(PY) python scripts/bench/run.py || { echo "bench: built in Phase 2 (scripts/bench/)"; exit 2; }
eval-dev:         ; @test -n "$(CONFIG)" || { echo "usage: make eval-dev CONFIG=configs/dev.yaml"; exit 64; }; $(PY) acis eval dev --config $(CONFIG)
gate:             ; @test -n "$(GATE)" || { echo "usage: make gate GATE=G1|G2|G3|..."; exit 64; }; $(PY) acis eval gate --gate $(GATE)

# OWNER-RUN (docs/spec/03 s5): the official run alone uses the sealed HF cache; cold, strict, uninterrupted.
rc-official:
	@test -n "$(RC)" || { echo "usage: make rc-official RC=RC0|RC1|RC2 [MODE=A|B|AB]"; exit 64; }
	SEAL=$${ACIS_SEALED_HOME:-$$HOME/.acis-sealed}; \
	HF_HOME=$$SEAL/hf HF_DATASETS_CACHE=$$SEAL/hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
	$(PY) acis eval official --rc $(RC) --mode $(or $(MODE),AB) --config configs/official.yaml --cold --strict

# JUDGE QUICK START (README opens with expected runtimes per hardware tier)
reproduce:        ; $(PY) acis eval official --config configs/official.yaml --cold --strict --reproduce --force --out runs/reproduce
reproduce-cache:  ; $(PY) acis eval official --config configs/official.yaml --cache-verify --strict --force --out runs/reproduce-cache
demo:             ; @test -x scripts/demo/run_demo.sh && scripts/demo/run_demo.sh || { echo "demo: built in Track B3 (scripts/demo/)"; exit 2; }
preflight:        ; @test -x scripts/preflight.sh && scripts/preflight.sh || { echo "preflight: use the /preflight skill (judge-simulator subagent); scripted parts arrive in Phase 6"; exit 2; }
evidence:         ; $(PY) acis report --claims --out docs/submission/evidence.md
