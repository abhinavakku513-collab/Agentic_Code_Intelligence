#!/usr/bin/env python
"""Validate the adapter on a **public, non-APPS** mteb code task (Phase 1 acceptance, docs/spec/07).

Why this exists: an adapter that only ever meets one task can be wrong in ways nobody notices — a task-specific
column name, a split name, an assumption about corpus size. Running the same object through an unrelated public
retrieval task, in both modes, is the cheapest way to find that out. It is also the first row of the REG suite that
gate G-OOD will need from Phase 3 (docs/spec/03 §5).

The REG cache lives **outside** `ACIS_HOME` (`acis.core.paths.reg_home`): these benchmarks ship their own
`qrels/test-*` files, which are other people's labels and must not sit next to ours.

Run (needs network the first time, like `make fetch`):

    uv run python scripts/validate_reg_task.py --task StackOverflowQA --mode B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acis.core.paths import reg_home  # noqa: E402

DEFAULT_TASK = "StackOverflowQA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--mode", default="B", choices=["A", "B"])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--offline", action="store_true", help="fail rather than download")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cache = reg_home()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1" if args.offline else "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1" if args.offline else "0"
    os.environ["ACIS_MODE"] = args.mode
    os.environ["ACIS_CONFIG"] = args.config
    os.environ.setdefault("ACIS_RUN_DIR", str(cache / "runs" / args.task))

    import mteb

    from acis.mteb_adapter import PrePostPipelineEncoder

    model = PrePostPipelineEncoder()
    task = mteb.get_task(args.task)
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
        cache=None,
        overwrite_strategy="always",
        show_progress_bar=False,
    )
    task_result = list(result.task_results)[0]
    split = next(iter(task_result.scores))
    scores = task_result.scores[split][0]

    report = {
        "task": task_result.task_name,
        "dataset": task.metadata.dataset["path"],
        "split": split,
        "mode": args.mode,
        "dispatch": type(model).__name__,
        "defines_index_and_search": hasattr(model, "index") and hasattr(model, "search"),
        "encode_calls": model.encode_calls,
        "adapter_invocations": model.adapter_invocations,
        "ndcg_at_10": round(float(scores["ndcg_at_10"]), 6),
        "mrr_at_10": round(float(scores["mrr_at_10"]), 6),
        "evaluation_time_s": round(float(task_result.evaluation_time or 0.0), 3),
        "note": "harness validation with the model-free stand-in encoder; not an accuracy claim",
        "reg_cache": str(cache),
    }
    # INV-14: a number about our system exists only if a ledger row backs it.
    from acis.eval import ledger

    run_id = ledger.append(
        ledger.LedgerRowBuilder(kind="audit")
        .with_metrics({"ndcg_at_10": report["ndcg_at_10"], "mrr_at_10": report["mrr_at_10"]})
        .with_fields(
            rung=f"reg:{args.task}",
            dataset=report["dataset"],
            decision_set=f"reg_{args.task.lower()}",
            mode=args.mode,
            dispatch=report["dispatch"],
            encode_calls=report["encode_calls"],
            adapter_invocations=report["adapter_invocations"],
            seconds=report["evaluation_time_s"],
            notes=report["note"],
        )
        .build()
    ).run_id
    report["ledger"] = run_id

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")

    # The point of the exercise: the right dispatch branch for the requested mode.
    if args.mode == "A" and model.adapter_invocations != 1:
        print("FAIL: Mode A did not run our search()", file=sys.stderr)
        return 1
    if args.mode == "B" and model.encode_calls == 0:
        print("FAIL: Mode B did not go through mteb's encoder wrapper", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
