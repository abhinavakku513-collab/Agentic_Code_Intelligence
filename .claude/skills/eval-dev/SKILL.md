---
name: eval-dev
description: Run a dev-split evaluation (never TEST) for a config and record it in the ledger with paired-bootstrap deltas against the current best.
argument-hint: [config-path]
disable-model-invocation: true
---
1. Verify `$ARGUMENTS` exists, targets dev/gate data (`kind` is not `rc`), and names its decision set: all 5,000 TRAIN queries (non-fit component) or K-fold OOF (trained component). DEV-H (F4) only as a counted once-per-milestone confirmation. Refuse anything that could load TEST labels.
2. Delegate to `retrieval-experimenter`: `uv run acis eval dev --config $ARGUMENTS` (appends a ledger row; jobs > ~10 min through `scripts/run_detached.sh`).
3. Report NDCG@10, MRR@10, Recall@{1,10,100,1000}, latency (p50/p95), cold time, peak RSS, paired ΔNDCG@10 with 95 % CI vs the current best, and G-OOD rows when the change touches query-side, adaptation, LTR or prep — each cited as `[ledger:<run_id>]`.
