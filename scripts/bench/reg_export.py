#!/usr/bin/env python
"""Export REG code-retrieval tasks (human-written queries) to plain JSONL under the REG home (docs/spec/10 §5).

The generic route — short questions, identifiers, anything that is not an APPS problem statement — is tuned on
REG data, never on APPS-derived stubs (spec 02 §6b, spec 10 Q4). These are other benchmarks' labels, so they live
under `acis.core.paths.reg_home()`, outside `ACIS_HOME` and far from the sealed APPS labels.

* **CosQA** (CoIR, MIT): real web queries ("sort by a token in string python") against 20,604 Python functions.
  Its query splits are kept apart: `valid` tunes, `test` reports — a weight chosen on the queries it is scored on
  would report its own fit.
* **CodeSearchNet (Python)** (mteb, MIT): docstring → function, evaluation only.

Needs the network once (like `make fetch`); afterwards everything reads the export.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from acis.core.paths import reg_home


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_cosqa(out: Path) -> dict[str, int]:
    from datasets import load_dataset

    corpus = load_dataset("CoIR-Retrieval/cosqa", "corpus")["corpus"]
    queries = load_dataset("CoIR-Retrieval/cosqa", "queries")["queries"]
    pairs = load_dataset("CoIR-Retrieval/cosqa", "default")
    _write(out / "corpus.jsonl", [{"id": r["_id"], "text": r["text"]} for r in corpus])
    text = {r["_id"]: r["text"] for r in queries}
    counts = {"corpus": len(corpus)}
    for split in ("valid", "test"):
        rel: dict[str, dict[str, int]] = {}
        for r in pairs[split]:
            if int(r["score"]) > 0:
                rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
        rows = [{"id": q, "text": text[q], "relevant": docs} for q, docs in sorted(rel.items()) if q in text]
        _write(out / f"queries.{split}.jsonl", rows)
        counts[split] = len(rows)
    return counts


def export_csn_python(out: Path) -> dict[str, int]:
    import mteb

    task = mteb.get_task("CodeSearchNetRetrieval", languages=["python"])
    task.load_data()
    subsets = task.dataset
    key = next(k for k in subsets if "python" in k)
    split = subsets[key]["test"]
    corpus, queries, relevant = split["corpus"], split["queries"], split["relevant_docs"]
    _write(out / "corpus.jsonl", [{"id": r["id"], "text": r["text"]} for r in corpus])
    rows = [
        {"id": r["id"], "text": r["text"], "relevant": {d: int(s) for d, s in relevant[r["id"]].items() if int(s) > 0}}
        for r in queries
        if r["id"] in relevant
    ]
    _write(out / "queries.test.jsonl", rows)
    return {"corpus": len(corpus), "test": len(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="cosqa,csn-python")
    args = parser.parse_args(argv)
    home = reg_home()
    os.environ["HF_HOME"] = str(home)
    for task in args.tasks.split(","):
        out = home / "export" / task
        counts = export_cosqa(out) if task == "cosqa" else export_csn_python(out)
        print(task, counts, "->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
