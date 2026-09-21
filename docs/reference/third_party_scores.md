# Third-party scores used for calibration only [R] — none is a measurement of ACIS
Rule: a number here may guide ordering of work (dense first, adaptation early) but may never appear in a deliverable as our result (INV-14). Status "unverified" until the owner or Claude re-reads the cited source; G0.4/G0.5 replace them with our own measurements.
| Claim (AppsRetrieval NDCG@10 unless noted) | Value | Cited source (as named in earlier blueprints) | Status |
|---|---|---|---|
| BM25 | 4.76 | minishlab `potion-code-16M` README (mteb ≥ 2.10 run) | unverified |
| UniXcoder / BGE-Base / BGE-M3 | 1.36 / 4.05 / 7.37 | CodeXEmbed paper (arXiv 2411.12644) Table 1, CoIR leaderboard baselines | unverified |
| E5-base / E5-Mistral | 11.52 / 21.33 | same | unverified |
| Voyage-Code-002 (API, excluded) | 26.52 | same | unverified |
| CodeSage-large-v2 | 50.45 | same | unverified |
| CodeXEmbed 400M / 2B / 7B (general training) | 48.57 / 74.99 / 85.22 | same | unverified |
| Qwen3-Embedding-0.6B (fp16, 8k tokens) | 75.22 | Jina, "Efficient Code Embeddings from Code Generation Models", Table 2 | unverified — one earlier blueprint called it "verified", another "unverifiable"; treat as unmeasured |
| jina-code-embeddings 0.5B / 1.5B (trained on APPS-train; CC-BY-NC) | 84.17 / 86.63 | same paper, Tables 2–3 | unverified; reference only |
| voyage-code-3 / gemini-embedding-001 (APIs, excluded) | 93.77 / 95.70 | same paper | unverified |
| gte-modernbert-base 149M / granite-embedding-english-r2 149M / granite small-r2 47M (CoIR **10-task average**, not APPS) | 71.5 / 54.8 / 53.4 | IBM granite-embedding model-card comparison table | unverified; candidate signal only |
How to verify: re-read the cited paper/model card, or reproduce with `mteb` on the same task and settings; record the outcome and date in this table and the ledger.
