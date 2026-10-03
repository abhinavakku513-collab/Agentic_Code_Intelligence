# P0 development evidence (APPS **train** split = dev; the TEST labels are never used here)

These files are the measurements behind the P0 configuration. All of them use the 5,000 queries of the
CoIR AppsRetrieval **train** split against the full 8,765-document corpus; anything trained (the ranker) is scored
out of fold (5 folds, each fold ranked by a model that never saw it).

| File | What it is | Produced by |
|---|---|---|
| `served_pipeline_summary.json` | the shipped pipeline end to end through `AcisEngine._rank_one` (the function behind the API, the UI and MTEB Mode A): NDCG@10 87.07, MRR@10 84.72, R@100 98.76 vs dense 71.03 — ledger rows `dev-839fba9f81a6` / `dev-5786ffd6b30a` | `scripts/bench/p0_qwen_chain.sh` → `scripts/bench/eval_pipeline.py --no-route-analysis` |
| `served_pipeline_per_query.jsonl` | per-query record of that run (route, ordering stage, gold rank, top 10); SHA-256 `f010320a9ec0…`, pinned in the ledger row | same |
| `encoder_diagnostics_gte_base.json` | each encoder alone over the full corpus (NDCG@10, MRR@10, Recall@1…1000), marginal recall over gte, z-score fusion sweeps | `scripts/bench/encoder_diag.py` |
| `encoder_diagnostics_qwen_base.json` | the same with Qwen3-Embedding-0.6B as the base encoder | same |
| `feature_lab_results.json` | every ranker / feature / pool / fusion variant tried, out of fold, with paired-bootstrap deltas | `scripts/bench/feature_lab.py` |
| `union_marginal_recall.json` | which retrieval channel's list contains the gold, and which channel alone found it | inline analysis of the engine-built pools (`ranker_lab.py build`) |
| `failure_buckets.json`, `failure_buckets_per_query.jsonl` | every dev query bucketed: top1 / top10 / ranking failure / union failure / missing from all | `scripts/bench/failure_buckets.py` |
| `qwen_subcorpus_probe.json` | the first sizing probe (400 queries, 2,873 cached documents) that justified embedding the full corpus with Qwen | `scripts/bench/qwen_probe.py` |
| `parity_mteb_vs_engine.json` | 300 DEV queries ranked by `mteb.evaluate` with the adapter (Mode A) and by a separately built engine: identical top-100 lists | `scripts/bench/parity_mteb_engine.py` |
| G-OOD (ledger rows `dev-88f75c495660` … `dev-6804437b5a4b`) | robustness of the frozen pipeline to 9 query perturbations, 200 DEV queries per family, out of fold: 6 of 9 pass; format_noise (+0.24, limit 0.0), sentence_dropout (+1.88) and truncate (+1.03, limit 1.0) exceed their limits with CIs spanning zero; under every perturbation the pipeline stays 16–19 NDCG@10 points above dense retrieval | `scripts/bench/g_ood_engine.py --sample 200 --record` |
| `generic_fusion_csn_python.json` | the generic route's Qwen weight on human-written CodeSearchNet-Python queries (w = 0.75: +3.58 NDCG@10) | `scripts/bench/reg_aux_weight.py` |

The hash-chained ledger (`runs/ledger.jsonl`) holds the recorded rows; `docs/adr/0009-qwen-second-encoder.md`
explains the decision these numbers support.
