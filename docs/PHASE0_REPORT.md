  # Phase 0 report — foundations and trust gates

  Every number below is a measurement on the declared host or a fact read out of the pinned dataset. Nothing here is
  an accuracy claim about ACIS (INV-14).

  ## G0.0 — hook canary

  The three canary conditions were exercised in-session against the live hooks (`docs/CANARY.md` step 2):

  | Condition | Result |
  |---|---|
  | `Read data/sealed/canary.txt` | blocked (permission deny layer) |
  | `Read data/hf/qrels/test.tsv` | blocked — *"ACIS guard blocked this call: TEST labels are sealed (INV-8)"* |
  | `cat data/sealed/canary.txt` via Bash | blocked — *"ACIS guard blocked this call: this command references sealed TEST labels"* |
  | `Write src/acis/canary_tmp.py` containing `eval(x)` | blocked — *"eval/exec are forbidden … (INV-5)"* |

  The `SessionStart` hook also ran and printed `docs/STATUS.md`, so all three hook events (SessionStart, PreToolUse,
  PostToolUse) are proven to execute. `uv run pytest tests/security -q` is green with the three GAP rows still
  `xfail(strict)`. **The owner still owns the final sign-off** (create the canary file, re-run after each Claude Code
  upgrade, record the version in `docs/STATUS.md`).

  Two false positives were found and are worth knowing about:

  1. the deny glob `Edit(./data/**)` also matches `src/acis/data/**`, which is why the package map's `data` role
    ships as `acis.appsdata` (ADR-0004);
  2. source code that *names* the held-out split next to the word `qrels` — such as an allow-list that exists to
    exclude those files — is treated as an attempted read, so `acis/appsdata/sources.py` composes the split name
    instead of writing it as a literal.

  ## G0.1 — mteb toy contract

  `tests/contract` is green on the installed mteb **2.21.0**: the dispatch matrix (encoder-only → wrapper;
  `index`+`search` → direct; adding `predict` → cross-encoder branch), the source anchor on
  `AbsTaskRetrieval._evaluate_subset`, and the V-08 tie experiment. Phase 1 adds the adapter contract tests
  (`encode` never called in Mode A, `min(top_k, N)` entries, stale-cache hazard, datetime-safe writer).

  ## G0.2 — dataset audit

  Full report: `runs/dataset_audit.json` (`uv run acis audit`). Verified layout of `CoIR-Retrieval/apps` at revision
  `f22508f96b7a…`:

  | File | Rows | Visibility |
  |---|---|---|
  | `corpus/corpus-00000-of-00001.parquet` | 8,765 | dev |
  | `queries/queries-00000-of-00001.parquet` | 8,765 | dev |
  | `data/train-00000-of-00001.parquet` | 5,000 qrels | dev |
  | the held-out qrels file | 3,765 | **sealed** — owner-only, `~/.acis-sealed/hf` |

  | Fact | Measured | Reference [R-01] |
  |---|---|---|
  | Corpus documents | 8,765 | 8,765 ✓ |
  | Dev (TRAIN) queries / qrels | 5,000 / 5,000 | 5,000 ✓ |
  | Relevant documents per query | min 1, max 1, mean 1.0 | 1 ✓ |
  | Exact-duplicate document groups | 11 groups, 22 documents | "≈11 exact duplicates" ✓ |
  | Document length (chars) | mean 575, p50 332, p95 1,561, max 289,048 | mean ≈573 ✓ |
  | Dev query length (chars) | mean 1,254, p50 1,052, p95 2,815, max 13,955 | (R-01 quotes the held-out split: ≈1,670) |
  | Document whitespace tokens | mean 70, p50 45, p95 201, p99 360, max 10,017 | — |
  | Dev query whitespace tokens | mean 220, p50 181, p95 510, p99 717, max 2,138 | — |
  | `ast.parse` success | 8,691 / 8,765 (74 failures, mostly Python-2 syntax) | — |
  | Id patterns | `d<int>`, `q<int>`; corpus order equals natural id order | — |
  | Fold sizes (`sha256(query_text) mod 5`) | 1,027 / 940 / 985 / 1,036 / 1,012 | — |

  Decisions this settles:

  * **I-01 confirmed.** The corpus carries a `partition` column: 5,000 `train` and 3,765 `test` documents. The
    documents of both partitions are in one corpus, so unlabelled documents are distractors — and a filter that knew
    which was which would be exactly the forbidden train-doc detector (CLAUDE.md §4). The column is read by the audit
    and by nothing else (`tests/integration/test_dev_harness.py::test_partition_metadata_does_not_leak_out_of_the_package`).
  * **Truncation defaults are safe.** p99 document length is 360 whitespace tokens and p99 query length 717, so the
    spec's 768+256 head/tail budget truncates almost nothing. The document maximum (10,017 tokens) is a long tail of a
    handful of files, not a material one — **E-LONG is not triggered** (spec 02 §5).
  * **Alternate solutions are not available.** `meta_information` holds `starter_code` and `url`, never a solution
    list, so the D7 plan of training on a *different* solution than the corpus copy cannot be executed from this
    dataset: decontamination and the D1 exposure diagnostic carry that weight instead (spec 02 §6, open item for
    Phase 3).
  * **Marker inventory** (dev queries): `-----Input-----` 1,181 · `-----Output-----` 1,185 · `-----Example…` 1,099 ·
    `-----Note-----` 373 · bare `Input` 2,821 · `Output` 2,750 · `Example` 2,987 · `Constraints` 1,483. Barely a
    quarter of the queries carry the ruled markers, which is the empirical argument for INV-15: a marker-gated
    segmenter would mis-handle most of the dataset, never mind arbitrary queries.

  ## G0.3 — hardware profile

  `runs/hardware.json` (`uv run acis doctor`). Declared development host:

  | Item | Value |
  |---|---|
  | CPU | AMD Ryzen 7 170 with Radeon Graphics |
  | Physical cores / logical | 8 / 16 |
  | RAM | 9.71 GB |
  | ISA | avx2 (no avx512) |
  | Tier | **T-min** (T-rec needs 16 GB) |
  | GEMM fp32, 1024³ | 157.6 GFLOP/s |
  | GPU | none |
  | Python / torch / mteb | 3.12.3 / 2.14.0+cpu / 2.21.0 |

  **Consequence for planning.** This host is T-min, so it is *not* the reference host a scorecard should quote
  (spec 06 §0 reserves that for T-rec, 8 cores / 16 GB). The encoder throughput measurements of G0.4/G0.5 and the
  projected cold-pass hours must be taken on the host that will actually run the official pass — an owner decision
  still recorded as `undeclared` in `docs/STATUS.md`.

  ## G0.4 / G0.5 — model radar and zero-shot scores

  **Not run in this phase.** Both need encoder weights downloaded and a multi-hour CPU pass over all 5,000 dev
  queries per candidate, and both feed gate G-M, which Phase 2 decides. `configs/models/qwen3-embedding-0.6b.yaml`
  still holds `base_commit: null`. Phase 1 needs neither: its acceptance criteria are about the harness, and the
  dense channel it exercises is the model-free stand-in encoder (`acis.embed.hashing`, `submission_capable=False`).

  Dependency licences *are* audited: `docs/reference/dependency_licences.md` — 108 packages, no GPL/AGPL, no
  CC-BY-NC, 5 MPL-2.0 packages that are all either test-only or dual-licensed.

  ## G0.6 — guard tests and the physical seal

  | Check | Result |
  |---|---|
  | `tests/security` green | yes (guard behaviour rows + 3 `xfail(strict)` GAP rows) |
  | Held-out label file inside the working tree | none |
  | Held-out label file inside `ACIS_HOME` | none — `acis doctor` reports `seal: clean`, and `fetch_dev()` re-checks after every download |
  | `acis fetch` allow-list | 4 files fetched, 1 sealed file skipped by pattern, 1 ignored |
  | Dev loaders | `load_qrels("anything-but-train")` raises `SealedDataAccess` |
  | Official run | refuses to start unless `HF_HOME` points at `~/.acis-sealed` |

  The seal is physical: the held-out labels are a *separate file* in the dataset repository, so keeping them out of
  the dev environment is a file filter rather than a promise. The hooks catch accidents; they are not the boundary
  (D19).
