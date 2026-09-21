# 03 · MTEB integration, evaluation integrity, gates

Authority: subordinate to `CLAUDE.md` (D8, D9, D16, D17, INV-3/4/8/10/11/14). Facts about mteb are [V] in `01` §2.

## 1. Boundary: engine vs adapter
`acis.engine` knows nothing about mteb: it takes `Snippet(handle, text)` lists and returns `{query_handle: [(doc_handle, score), …]}`. `acis.mteb_adapter` only (i) converts `corpus["id"|"title"|"text"]` / `queries["id"|"text"]`, (ii) owns ModelMeta/revision identity, (iii) enforces strict mode, (iv) writes the run manifest + artifacts. Handles are opaque (INV-4). No ranking logic in the adapter (INV-11).

### P0 execution path (exact)
```
APPS CoIR-Retrieval/apps@f22508f96b7a… (test) ──mteb──▶ corpus(id,title,text) · queries(id,text) · qrels
Mode A  PrePostPipelineEncoder.index(corpus)  → snapshot: d1 prep · vectors via CAS · BM25 (per snapshot)
        .search(queries, top_k=1000)          → per query: q1 normalise → instruction+view → encode → exact dense over ALL docs ∥ BM25
                                                → union ≤100 → features → LightGBM LTR → dense tail → rank-derived scores (1000/query)
Mode B  encode() only: same model + prep, mteb SearchEncoderWrapper (honest embeddings; separate JSON)
mteb    pytrec_eval ndcg_cut/map/recall/P/success + mteb MRR (sort key (score, doc_id) desc) → TaskResult → JSON
Output  appsretrieval_results.json (+ .modeB.json) · run.csv + run.trec · predictions/ · manifest.json · SHA256SUMS · resource scorecard
Rules   cache=None · overwrite_strategy="always" · unique ModelMeta(name "acis/acis-apps", revision <git12>+<cfg12>) · strict · datetime-safe writer · no predict()
```
Verified in mteb 2.21.0 source (latest on PyPI, 2026-09-20): dispatch is `EncoderProtocol ∧ ¬SearchProtocol → SearchEncoderWrapper`, `CrossEncoderProtocol → cross-encoder wrapper`, `SearchProtocol → used directly`; `evaluate()` defaults to a persistent cache with `only-missing`; `top_k`=max(k_values)=1000. Re-verify commands: `docs/spec/01` §2.

## 2. Adapter contract
1. Class `PrePostPipelineEncoder(AbsEncoder)`, zero-argument constructor, models loaded lazily. 2. Mode A defines `index()` **and** `search()` (keyword-only, `num_proc=None`, `**_` absorbs later arguments) → mteb uses the object directly. 3. **Never define `predict()`.** 4. Mode B = the same class without `index/search` (returned by `__new__` when `ACIS_MODE=B`), with an honest `encode()` (same model + prep) so mteb's `SearchEncoderWrapper` runs. 5. `mteb_model_meta` is a real `ModelMeta` (`name="acis/acis-apps"`, `revision="<git12>+<config12>"`); build with `create_empty` when present, else an introspecting fallback. 6. `search()` returns exactly `min(top_k, N)` finite, strictly decreasing scores per query id, ids ⊆ corpus ids; strict mode aborts on any degradation. 7. `search()` writes the run manifest (proof our code ran); adapter invocation counter must be 1. 8. Official script: `cache=None`, `overwrite_strategy="always"`, `prediction_folder=…`, `encode_kwargs={"batch_size":64}`.

```python
# src/acis/mteb_adapter.py  (skeleton; final code is tested against the compatibility matrix)
from __future__ import annotations
import os
import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType
from acis.core.config import load_frozen_config
from acis.core.types import Snippet
from acis.engine import AcisEngine
from acis.mteb_meta import make_model_meta          # create_empty(overwrites=…) if present, else introspect required fields
from acis.rank.compose import rank_derived_scores   # {doc_id: (top_k+1-rank)/top_k}

def _doc_text(title, text): return f"{title} {text}".strip() if title else text   # mteb convention

class _EncoderSurface(AbsEncoder):                  # Mode B surface: honest dense encoder
    def __init__(self, config_path=None):
        self.cfg = load_frozen_config(config_path or os.environ.get("ACIS_CONFIG", "configs/official.yaml"))
        self._engine, self._meta = None, make_model_meta(self.cfg)
    @property
    def mteb_model_meta(self) -> ModelMeta: return self._meta
    @property
    def engine(self) -> AcisEngine:
        if self._engine is None: self._engine = AcisEngine.from_config(self.cfg)
        return self._engine
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kw):
        texts = [t for batch in inputs for t in batch["text"]]
        return np.asarray(self.engine.dense.encode(texts, is_query=(prompt_type == PromptType.query),
                          batch_size=int(kw.get("batch_size", 64))), dtype=np.float32)

class PrePostPipelineEncoder(_EncoderSurface):      # Mode A surface = SearchProtocol
    def __new__(cls, *a, **k):
        if os.environ.get("ACIS_MODE", "A").upper() == "B":
            return _EncoderSurface(*a, **k)         # no index/search attributes -> SearchEncoderWrapper path
        return super().__new__(cls)
    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None,
              encode_kwargs=None, num_proc=None, **_):
        titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus["id"])
        docs = [Snippet(handle=str(i), text=_doc_text(t, x)) for i, t, x in zip(corpus["id"], titles, corpus["text"])]
        self._snap = self.engine.build_snapshot(docs, source=f"mteb:{task_metadata.name}:{hf_split}")
    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None, top_k=1000,
               encode_kwargs=None, top_ranked=None, num_proc=None, **_):
        ranked = self.engine.search_batch(self._snap, [str(i) for i in queries["id"]], list(queries["text"]),
                                          top_k=top_k, restrict_to=top_ranked, strict=self.cfg.strict)
        self.engine.write_run_manifest(task_metadata, hf_split)
        return {qid: rank_derived_scores(hits, top_k) for qid, hits in ranked.items()}
```
**Official writer** (`acis.mteb_adapter.write_official_json`): `json.dump(task_result.to_dict(), f, indent=2, default=_json_default)` with `datetime→isoformat`, `Path→str`, `np.generic→item()`; additionally `task_result.to_disk(...)` as the mteb-native artifact. The PDF sample's bare `json.dump` is not used.
**Why rank-derived scores** (D9, V-08): pytrec_eval merges near-equal scores; mteb MRR does not; raw model scores or ε-ramps can make NDCG and MRR disagree. Exact-content duplicates get distinct adjacent ranks by corpus ordinal (the only place order is used).
**Forbidden**: basis/one-hot "embeddings", score-vector encoders, `similarity()` overrides that read the corpus, precomputed test rankings, any use of qrels, IDs, doc order or batch statistics in ranking.

## 3. Modes A/B and the primary JSON (G-AB)
Both modes run from the same frozen RC (same model, prep, config hash + mode flag). Mode B may use everything an encoder can express (adapted model, prep, truncation, an exact weighted ensemble by concatenating L2-normalised embeddings); it cannot express rank fusion, LTR or PRF. **Primary = A iff ΔNDCG@10(A−B) ≥ +0.5 pt with paired-bootstrap CI lower bound > 0 on the OOF pool over all 5,000 TRAIN queries (spec 09 §2); else B** (simpler = cheaper). RC0 (Phase 2) is Mode B with the best frozen dense encoder; the A-vs-B decision is taken at Phase 4/5 on the final encoder. The other JSON is attached and both are disclosed in the README (and in the owner's slides). P1, Bonus and the interactive demo always use the search-level engine.

## 4. Artifacts of one official run (`acis eval official`, cold, strict)
`appsretrieval_results.json` (primary) · `appsretrieval_results.modeA.json` / `.modeB.json` · `run.csv` (`query_id,corpus_id,rank,score`, top-1000, explicit rank) · `run.trec` · `predictions/` (mteb native) · `manifest.json` · `evaluation.log` · `SHA256SUMS` (+ signature) · resource scorecard. The JSON's `evaluation_time` is the honest cold time (D17). A separate `--cache-verify` run (shipped vectors, 1 % spot recompute, cosine ≥ 0.9999) lets judges reproduce rankings quickly; it is labelled and never replaces the cold JSON. Never ship test-query embeddings or rankings as a "cache".
**`acis verify-submission`** checks: JSON parses and has `scores.test[0].ndcg_at_10`; `dataset_revision == f22508f96b7a…`; `mteb_version` recorded; model revision equals the manifest's `<git12>+<config12>`; manifest shows `adapter_invocations==1`, `fallbacks==0`, `strict==true`, `agent_calls==0`, `test_touch_count` consistent with the ledger; re-scoring `run.trec` with pytrec_eval + mteb's MRR reproduces NDCG@10/MRR within 1e-9; both modes present; `evaluation_time` above a floor (proves code ran); checksums valid.
**Compatibility CI**: adapter contract tests on synthetic retrieval tasks against mteb 2.21.0 (pinned), 2.12.x, 2.5.x (+2.1.x smoke) — dispatch path, `encode` never called in Mode A, `top_k` entries, JSON writer, stale-cache hazard reproduced then defeated.

## 5. Data, splits, physical seal, leakage controls
| Set | Content | Rule |
|---|---|---|
| TEST | AppsRetrieval test qrels (3,765) | kept **outside the working tree** (`~/.acis-sealed/`, D19); read only by mteb inside `acis eval official` and by `acis.eval.final`; budget **6** (RC0 ×1, RC1 ×2, contingency ×2, post-hoc clean-pool ×1) |
| TRAIN | CoIR-APPS train qrels (5,000) [R] | folds `sha256(query_text) mod 5` → F0…F4; the dev environment loads `split="train"` only |
| OOF pool | all 5,000 TRAIN queries | **decision set**: frozen/non-fit components are scored on all 5,000; trained components (LoRA, LTR, PMI) only through K-fold out-of-fold predictions |
| DEV-H | F4 (≈1,000) | once-per-milestone **confirmation** with a touch counter; not the decision set |
| REG | mteb code tasks with human-written queries (CosQA, StackOverflowQA, SyntheticText2SQL, CodeSearchNet, CodeFeedback-ST, …; exact class names via `mteb.get_tasks` in G0) | guards generality; part of G-OOD |
Dev tasks use the full 8,765-doc corpus (unlabelled docs as distractors — mirrors test difficulty) through an `AbsTaskRetrieval` subclass, so dev exercises the same adapter and metric code as the official run. **Why not DEV-H alone**: at n ≈ 1,000 the paired-bootstrap CI half-width is ≈ 1 pt, so the +0.5 pt rule effectively demands ≳ 1 pt (simulation in spec 09 §2 [E]).
**Physical seal (D19)**: (1) `acis fetch` lists the dataset repo files and downloads only allow-listed non-test patterns into `ACIS_HOME`; the TEST qrels are fetched once, by the owner, into `~/.acis-sealed/hf`; if the layout stores splits inside one file, dev extracts train rows in a throwaway `HF_HOME` and deletes it [layout U → verified in G0.6]. (2) The official run alone sets `HF_HOME=~/.acis-sealed/hf` and `HF_DATASETS_CACHE=~/.acis-sealed/hf/datasets`. (3) `settings.json` denies Read/Edit on the sealed paths; the guard hook blocks accidental reads/writes; `xfail(strict)` gap tests document what a regex cannot see. (4) The owner launches `make rc-official` (or confirms the ask-gated command); Claude Code reads only the ledger row and `verify-submission` output. (5) Wording everywhere: “TEST labels are kept outside the working tree and touched only at ledgered release candidates; hooks catch accidental access” — never “provably sealed”.
**Controls**: `configs/splits.lock.json` stores SHA-256 of the sorted ID lists of every set; `acis.eval.guard` raises if a TEST id reaches any training batch, feature table or tuning call; CI greps for reads of sealed paths outside `acis.eval.final`; decontamination and the exposure diagnostic (on held-out folds) per spec 02 §6; permutation test (shuffled IDs/order ⇒ identical metrics except duplicate ordinal); batch-invariance test (single query ≡ inside a 3,765-batch, scores within 1e-5).

## 6. Ledger and reproducibility
Append-only, hash-chained `runs/ledger.jsonl` (written only by `acis.eval.ledger`): `{prev_hash, run_id, ts, git_sha, dirty, dataset_revision, split_lock_hash, mteb_version, python/torch/transformers/lightgbm/bm25s versions, model_fingerprint, adapter_hash, prep_hash, feature_ver, ltr_hash, config_hash, numeric_profile, seeds, thread_count, hardware{cpu, cores, isa, ram}, kind{dev|gate|rc}, mode, test_touch_count, metrics, latency{p50,p95,p99}, resource{peak_rss, cpu_seconds, evaluation_time}, fallback_counters, artifact_sha256s}`. `acis eval repro <run_id>` re-derives the config hash, verifies lockfile/model/asset hashes, refuses dirty trees for RCs and compares the rerun to the ledger within tolerance. Determinism claim = same machine, config, thread count; rankings compared by tolerance test (identical top-10 except swaps within 1e-6).

## 7. Ladder and gates
**Statistics** (spec 09 §2): paired bootstrap, 10,000 resamples over queries; practical threshold ΔNDCG@10 ≥ +0.5 pt **and** CI lower bound > 0; decision sets as in §5 (all 5,000 TRAIN queries for non-fit components, K-fold OOF for trained ones, DEV-H confirmation once per milestone); cap the number of variants, log every trial; ties → the cheaper option (G-R). Simulation [E]: at n = 5,000 a true +0.75 pt gain is accepted ≈ 87 % of the time vs ≈ 34 % at n = 1,000.
| Gate | Question | Accept if | Else → frozen default |
|---|---|---|---|
| G0.0 | do the hooks actually execute? | `/canary`: sealed read blocked, sealed `cat` blocked, `eval(` write blocked (docs/CANARY.md) | fix wiring first; repeat after every Claude Code upgrade |
| G0.1–G0.5 | mteb toy contract · data audit · hardware · model radar + pin · zero-shot on TRAIN queries | all pass | stop, fix |
| G0.6 | guard tests + physical seal verified (dev cache holds no TEST label file) | `tests/security` green; seal check green | stop, fix |
| G-M | encoder | D4 rule (smallest within 1.0 pt of best; 3.0 pt if the best's cold pass > 2 h; under SLO) | seed Qwen3-Embedding-0.6B |
| G1 | task string, view, lengths — **per route** | best by rule; ties → cheaper | `statement_like`: T1 + V0, 1024/1024; `generic`: T3 + V0 |
| G2 | BM25 + bridge features help | thresholds met | dense-only channel set |
| G3 | LoRA adaptation (+ α) | threshold on OOF, REG drop ≤ 1.0 pt/task, D1 clean on a held-out fold, **G-OOD passes** | frozen base |
| G4 | PRF second pass | threshold | disabled |
| G5 | LTR vs weighted fusion vs dense-only (OOD-guarded, group dropout) | threshold over the next-simpler ranker | weighted fusion if it beats dense-only, else dense-only |
| G-AB | Mode A vs B primary | §3 rule on the OOF pool | B primary |
| G6 | bf16/int8 profile | overlap ≥ 99 %, ΔNDCG ≥ −0.3 | cpu-fp32 |
| G7 | determinism, batch invariance, ID permutation, strict-mode tests | all pass | block release |
| G-OOD | not more brittle than the frozen base (spec 10 §5) | per perturbation family `drop(system) ≤ drop(base) + 1.0 pt`; REG ≤ 1.0 pt/task; format noise strict no-op | reject the adaptation/LTR/prep change |
| G-R | resource tie-break | inside the threshold → fewer params / lower cold time | – |
Ladder: B0 metric parity (random/oracle rankings through our code vs mteb) → B1 stock `bm25s` BM25 (parity with `mteb/baseline-bm25s`, ≥ 95 % top-10 identical with matched tokenisation — harness sanity, not a submission) → B2 frozen dense variants (bake-off; reference-only models measured) → B3 + per-route prep → B4 adapted (G3) → B5 + BM25 fusion → B6 + bridge features / LTR → B7 + PRF → B8 numeric profile (G6). Report NDCG@10, MRR@10/@1000, Recall@{1,10,100,1000}, p50/p95 latency, cold and warm time, peak RSS per rung. Experiments that must exist: truncation sweep, per-route instruction/view sweep, hard-negative ablation (with/without false-negative filter), feature-group ablations and dropout, D1 exposure report, REG + perturbation (G-OOD), 5-seed LTR variance, duplicate-handling audit.
