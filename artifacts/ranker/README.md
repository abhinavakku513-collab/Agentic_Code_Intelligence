# Shipped ranking models

**`p0-gte-qwen-mix-pool500.txt` is the model the P0 configuration serves** (`configs/dev.yaml`, `configs/official.yaml`):
LightGBM LambdaRank trained on the 500-candidate pools of all 5,000 dev (APPS train) queries, built by the engine's own
`candidate_pool` + `pool_features` with both dense encoders (gte-modernbert-base and Qwen3-Embedding-0.6B), BM25 and
exact symbols — 34 features including the gte/Qwen agreement group; hyper-parameters frozen in
`acis.rank.ltr.PARAMS`, 400 rounds, seed 0.

* **What it was measured at**: the served pipeline, out of fold, NDCG@10 87.07 · MRR@10 84.72 against the dense
  baseline's 71.03 · 67.46 — `[ledger:dev-839fba9f81a6]`, `[ledger:dev-5786ffd6b30a]`.
* **What "out of fold" means here**: the number above comes from five models, each scoring only the fold it was not
  trained on. **This file** is the refit on all five folds with the same hyper-parameters, which is what serves; its
  own score on its training queries would be higher and would mean nothing.
* **How to rebuild it**: `bash scripts/bench/p0_qwen_chain.sh` (refit, then the out-of-fold evaluation).

It lives in the repository rather than under `runs/` because the submission has to be reproducible from a clone.

Older models, kept so their ledger rows stay reproducible: `p0-gte-qwen-pool500.txt` (without the agreement
features, 86.70 `[ledger:dev-f93aeb265be7]`), `p0-gte-pool500-symbols.txt` (the previous served pipeline,
74.13 `[ledger:dev-299a3010be5a]`) and `g5-gte-modernbert-code.txt` (gate G5, 72.97 `[ledger:gate-9bdbad663084]`).
