# Official P0 evaluation — CoIR AppsRetrieval TEST split (MTEB)

The submission run, produced once with the frozen configuration (`configs/official.yaml`, Mode A, strict, CPU, fp32,
cold) by `make rc-official RC=RC1 MODE=A`. Ledger row `rc-bd2285335a86`; model revision `280454985c1b+68b5a3b4e3a0`.

| File | What it is |
|---|---|
| `appsretrieval_results.json` | MTEB's `TaskResult.to_dict()`, written unmodified — NDCG@10 0.78283, MRR@10 0.74393 on `test` |
| `manifest.json` | the adapter's run manifest: Mode A dispatch (`adapter_invocations=1`, `encode_calls=0`), strict, 0 fallbacks, config hash, model revision, dataset revision |
| `verify_report.txt` | `verify-submission` written at the end of the run (PASS; the re-score against labels is skipped there) |
| `verify_report_heldout.txt` | `make rc-verify`: the same checks plus re-scoring `run.trec` against the held-out labels — NDCG@10 within 5.8e-7 of the JSON's 5-decimal value, MRR@10 exact; PASS |
| `SHA256SUMS` | checksums of the run directory's files |

The ranked lists themselves (`run.trec`, `run.csv`, MTEB's predictions file) are in `acis-rc1-official-run.zip` on
the GitHub Release v1.0.0.
