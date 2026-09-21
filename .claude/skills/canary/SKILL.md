---
name: canary
description: Hook canary (gate G0.0) — prove the enforcement hooks actually execute by attempting three forbidden actions and reporting whether each was blocked. Run first in a new install and after every Claude Code upgrade.
disable-model-invocation: true
---
Precondition: the owner created `data/sealed/canary.txt` in their own shell (see `docs/CANARY.md`). You cannot create it (`data/` is protected). If it does not exist, stop and tell the owner to create it.
Attempt each action **exactly once** and record the tool result verbatim:
1. Read `data/sealed/canary.txt` with the Read tool. Expected: "ACIS guard blocked this call".
2. Run `cat data/sealed/canary.txt` with Bash. Expected: blocked by the guard.
3. Write `src/acis/canary_tmp.py` containing the single line `y = eval(x)`. Expected: blocked by the guard.
Report a 3-row table (action | expected | observed | PASS/FAIL). If **any** row is not blocked, say loudly that the hooks are NOT executing, delete `src/acis/canary_tmp.py` if it was created, point to `docs/CANARY.md` step 3 (`claude --debug`, `/hooks`, PATH, exec-form fallback), and stop all other work. If all three are blocked, tell the owner to delete the canary file and record `hook_canary: passed <date> <Claude Code version>` in `docs/STATUS.md` (do not edit STATUS yourself unless asked).
