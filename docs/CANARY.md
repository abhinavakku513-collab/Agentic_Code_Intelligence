# Hook canary (gate G0.0) — a green `/hooks` listing does not prove the hooks run
Run once after installing the kit, and again after every Claude Code upgrade (the `/canary` skill guides the Claude-side steps).
1. In **your own shell** (Claude cannot write under `data/`): `mkdir -p data/sealed && echo canary > data/sealed/canary.txt`.
2. In Claude Code run `/canary`. Expected: (a) reading `data/sealed/canary.txt` → message starting **"ACIS guard blocked this call"**; (b) `cat data/sealed/canary.txt` via Bash → blocked; (c) writing `src/acis/canary_tmp.py` containing `eval(x)` → blocked.
3. If any of the three is **not** blocked, the hooks are not executing: `claude --debug`, re-check `/hooks`, confirm `python3` is on PATH for the shell Claude Code uses, and use the other wiring form (exec form `command` + `args`, documented for Claude Code ≥ 2.1.139) before doing anything else. Do not start Phase 0 work until the canary passes.
4. Delete the canary file: `rm data/sealed/canary.txt`. Record `hook_canary: passed <date> <Claude Code version>` in `docs/STATUS.md`.
Also run `uv run pytest tests/security -q` (guard behavioural contract; gap rows are pinned `xfail(strict)` — hooks catch accidents, the seal is physical, D19).
