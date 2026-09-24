# Shipped ranking model

`g5-gte-modernbert-code.txt` is the LightGBM LambdaRank model gate G5 accepted: trained on all 5,000 dev queries
with the G-M encoder (`gte-modernbert-base`) and the code-aware lexical channel, hyper-parameters frozen in
`acis.rank.ltr.PARAMS`.

It lives in the repository rather than under `runs/` because the submission has to be reproducible from a clone:
a model referenced by a configuration but absent from the tree is a configuration that does not work.

* **What it was measured at**: NDCG@10 72.97 out-of-fold against a frozen-dense baseline of 71.03,
  Δ +1.93 pt, CI [+1.52, +2.35] — `[ledger:gate-9bdbad663084]`.
* **What "out-of-fold" means here**: the number above comes from five models, each scoring only the fold it was
  not trained on. **This file** is the refit on all five folds with the same hyper-parameters, which is what
  serves; its own score on the training queries would be higher and would mean nothing.
* **How to rebuild it**: `uv run python scripts/bench/ltr_build.py --model gte-modernbert-base --tokenizer code`.
