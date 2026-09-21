---
paths:
  - "src/acis/mteb_adapter.py"
  - "src/acis/mteb_adapter/**"
  - "src/acis/mteb_meta.py"
  - "src/acis/eval/**"
---
# MTEB adapter and evaluation-integrity guardrails — read `docs/spec/03-mteb-and-integrity.md` first
- The adapter only translates (INV-11). Never define `predict()`. `index/search` are keyword-only with `num_proc=None` and `**_`.
- ModelMeta: `name="acis/acis-apps"`, `revision="<git12>+<config12>"`. Official run: `cache=None`, `overwrite_strategy="always"`, strict, batch size 64.
- Mode A scores are rank-derived `(top_k+1−rank)/top_k`, exactly `min(top_k, N)` entries per query; JSON only through the datetime-safe writer.
- TEST labels live outside the working tree (`~/.acis-sealed/`, D19) and are readable only by `acis.eval.final` / `acis eval official`; `acis.eval.guard` raises anywhere else; dev tasks load `split="train"` only. Never launch the official run yourself — the owner does. The ledger is append-only through `acis.eval.ledger`.
- Gate decisions use the decision sets of spec 03 §5 (all 5,000 TRAIN queries for non-fit components, K-fold OOF for trained ones); DEV-H is a counted once-per-milestone confirmation.
- Dev tasks use the same adapter and mteb metric functions as the official run; parity tests P1–P7 stay green.
- Gate decisions are written once to `configs/gates/<id>.yaml`; supersede via `/adr`, never edit.
