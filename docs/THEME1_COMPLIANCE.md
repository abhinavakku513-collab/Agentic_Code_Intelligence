# Theme 1 compliance checklist

Source of truth: the Theme 1 guidelines and FAQ (`docs/official/theme1_guidelines.md`, `docs/official/FAQ.md`).

| Requirement (guidelines) | How ACIS meets it | Where |
|---|---|---|
| Code retrieval: rank code snippets for a natural-language query; generation out of scope | Retrieval and ranking only; every result is existing code re-read by hash; no LLM in the ranking path | `src/acis/engine/core.py`, README §1, §3 |
| Improvements may include query categorisation, pre-processing, snippet processing, **multiple retrieval passes** | Query routing (statement vs. generic vs. identifier), normalisation, head+tail truncation, four independent retrieval passes, a 500-candidate union and a learned ranker | README §3–4 |
| Runs on CPU with minimal GPU | CPU-only inference (`device: cpu`, fp32), CPU-only PyTorch in the lock; no GPU used anywhere | `configs/official.yaml`, `pyproject.toml` |
| Resource use (running time, GPU, model size) is evaluated (FAQ) | Two encoders, 0.75 B parameters total, no GPU; costs stated in ADR-0009; cold `evaluation_time` reported honestly | `docs/adr/0009-qwen-second-encoder.md` |
| APPS (Python) only (FAQ) | Python only; JS ignored as instructed | README §2 |
| **P0** screening: CoIR AppsRetrieval **test** split, **NDCG@10** and **MRR** | Official run via MTEB on the test split; both metrics are in MTEB's JSON | README §10–11 |
| Use the **MTEB** library: `AbsEncoder` subclass → `mteb.get_task("AppsRetrieval")` → `mteb.evaluate(..., encode_kwargs={"batch_size": 64})` → JSON of `task_result.to_dict()` | `PrePostPipelineEncoder(AbsEncoder)` in `src/acis/mteb_adapter.py`; `make rc-official` / `make reproduce` run exactly that recipe; `write_official_json` = `json.dump(task_result.to_dict())` with a date-safe encoder | README §10 |
| JSON with the inference on the test split, uploaded as a **GitHub Release** asset | `appsretrieval_results.json` (NDCG@10 78.283, MRR@10 74.393 on `test`) attached to Release v1.0.0; copy in `docs/evidence/test/` | README §6, §11 |
| GitHub repo with instructions on how to run the submission | Beginner install / run / search / evaluate instructions; judge quick start | README §7–10, §15 |
| Attach any files needed as Release artifacts | Release also carries the prebuilt vector packs (optional speed-up) | README §7 |
| **P1**: retrieval on different versions; rebuild indexes/caches in reasonable time | Immutable snapshot per version, incremental embedding, atomic activation, rollback; ~2 s p95 for a 1–10 unit change `[ledger:bench-9153e49050de]` | README §12 |
| **Bonus**: retrieval across all versions; near-identical snippets | Lineage alignment, one answer per lineage with timeline; grouped and flat tie on Evolution-NDCG@10 (no gain claimed) | README §13 |
| Hands-on: queries similar to the dataset; show responses and speed | `acis search` / `acis serve` show results, per-channel evidence and per-stage timings for any free-text query | README §8–9, §14 |
| PPT and demo video | Prepared by the team separately (not part of this repository) | — |
