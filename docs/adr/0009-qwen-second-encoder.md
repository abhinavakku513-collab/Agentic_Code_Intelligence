# ADR-0009 — Qwen3-Embedding-0.6B joins the P0 pipeline as a second dense encoder

- **Status:** accepted (2026-10-02, owner instruction: P0 retrieval accuracy is the top priority)
- **Supersedes in part:** G-M (`configs/gates/G-M.yaml`), which excluded Qwen3-Embedding-0.6B on its *projected cold
  pass* (26.1 h, measured from a contended throughput probe) before any accuracy was measured.
- **Context:** D4 asks for the smallest encoder within 1.0 NDCG@10 point of the best (3.0 if the best's cold pass
  exceeds 2 h) and a 4 h cold-pass SLO. Qwen was never measured on accuracy, so "the best" in that rule was unknown.

## Evidence (dev = APPS train split, all 5,000 queries, full 8,765-document corpus; TEST never read)

| Measurement | Result | Source |
|---|---|---|
| gte-modernbert-base dense | NDCG@10 71.04 · MRR@10 67.47 · R@100 93.78 · R@1000 98.88 | `scripts/bench/encoder_diag.py` |
| Qwen3-Embedding-0.6B dense (statement instruction T1) | NDCG@10 84.41 · MRR@10 81.63 · R@100 98.50 · R@1000 99.54 | same |
| Gold in Qwen's top 100 but not gte's / the reverse | 256 / 20 queries | same |
| Qwen as ranker features over the previous pools | NDCG@10 85.89 vs 74.14, Δ +11.75 [+11.00, +12.49] | `scripts/bench/feature_lab.py` |
| + Qwen top 300 in the 500-candidate union | pool recall 97.16 → 99.38 %, NDCG@10 86.32 | `scripts/bench/ranker_lab.py build`, `feature_lab.py` |
| Generic-route fusion with Qwen (w = 0.75) | APPS +0.30 [+0.17, +0.45]; CodeSearchNet-Python (human queries) +3.58 [+2.69, +4.48] | `feature_lab.py`, `scripts/bench/reg_aux_weight.py` |
| Rejected alternatives | granite-small-r2 as a third encoder +0.11 (fails the +0.5 gate); union cap 700 ranks worse (86.33 vs 86.62); PRF +0.06; query view V1 +0.04; ranker on every route +0.37 (fails the gate) | `feature_lab.py` |

The end-to-end number for the shipped configuration — through `AcisEngine._rank_one`, out of fold, with a ledger row
— is the one quoted in the README; the rows above are the lab measurements that chose it.

## Decision

- gte-modernbert-base stays the **primary** encoder (routing bank, confidence, P1/Bonus snapshots, dense tail).
- Qwen3-Embedding-0.6B (Apache-2.0, safetensors, no remote code, pinned commit `97b0c614…`) is the second encoder
  (`model.aux_encoder`): its top 300 join the candidate union, its cosine and whole-corpus rank are ranker features
  (`cos2`, `rank_dense2`, `z_cos2`), and it carries 0.75 of the generic route's dense term.
- The ranker is refit on the new pipeline's own pools (`artifacts/ranker/p0-gte-qwen-mix-pool500.txt` after the
  addendum below), hyper-parameters unchanged (capacity variants measured within noise or worse).
- Mode A (the full pipeline) is the primary submission surface.

## Addendum (2026-10-02) — encoder-agreement features, then freeze

With the gate rule's remaining headroom in *ranking* (gold in the pool for 99.38 % of DEV queries, NDCG@10 86.70),
five deterministic features comparing the two encoders inside each query's own pool were added to the ranker
(`mix` group in `acis.rank.candidates`: z_cos + z_cos2, z_cos − z_cos2, log rank ratio, log min rank, pool rank of the
z-sum). End to end, out of fold: NDCG@10 86.70 → **87.07**, MRR@10 84.19 → **84.72** `[ledger:dev-839fba9f81a6]`;
paired Δ +0.37 [+0.11, +0.64] NDCG@10 and +0.53 [+0.19, +0.87] MRR@10 — below the +0.5 NDCG@10 bar of the gate rule
but with a CI clear of zero on both metrics and no measurable cost, so adopted. Measured and not adopted: pool-context
features (document–document similarity to the pool's top results, near-duplicate counts; +0.25, nothing on top of
the agreement features), more ranker capacity (31 leaves +0.17, 600 rounds +0.22, both CIs spanning zero), the
`rank_xendcg` objective (−0.95), truncation 10 (−0.06), the ranker on every route (+0.08 on top). A cross-encoder
reranker was not pursued: on this CPU it would cost tens of hours on the TEST queries.

**The architecture is frozen here.** The official TEST run is the final evaluation; nothing is tuned on its result.

## Costs, stated rather than hidden

- Parameters: 149 M (gte) + 596 M (Qwen) ≈ 0.75 B, inside D4's ~1 B envelope.
- CPU time on the 8-core dev host: Qwen embeds a corpus document in ~1.8–2.3 s and a full problem statement in
  ~1.5–2 s (fp32). The cold official pass therefore exceeds D4's 4 h SLO on this host; the SLO is relaxed for P0
  by this ADR and the measured cold time is reported, never replaced by a warm one (D17).
- Measured: the cold official pass took **27,849 s (7.7 h)** on the 8-core dev host `[ledger:rc-bd2285335a86]`, with
  Qwen batches capped at 4,096 tokens (`token_budget` in its model card) to stay inside 10 GB of RAM.
- bf16 was ~4× *slower* on this CPU and dynamic int8 broke the vectors (cosine 0.33–0.56 to fp32), so fp32 stays.
- Interactive queries pay one extra Qwen forward pass (~1–2 s for a long statement, less for a short question).
