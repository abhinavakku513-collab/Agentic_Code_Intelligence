"""The **only** sanctioned reader of the held-out labels (INV-8, docs/spec/03 §5).

Nothing in ACIS imports this module except `acis eval official` and `acis eval verify-submission`. It exists so
that "who may read the labels" is a single, greppable place rather than a convention, and so that the guard hook's
allow-list has exactly one target.

Claude Code never runs this. The owner does, in their own terminal, through `make rc-official`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from acis.core.errors import SealedDataAccess
from acis.core.paths import sealed_root
from acis.eval.guard import official_run_env_ok


def _require_sealed_environment() -> None:
    ok, hf_home = official_run_env_ok()
    if not ok:
        raise SealedDataAccess(
            "held-out labels may only be read with HF_HOME pointing at the physical seal (D19)",
            hf_home=hf_home or "unset",
        )
    if not sealed_root().exists():
        raise SealedDataAccess(f"the sealed area does not exist: {sealed_root()}")


def load_holdout_qrels() -> Mapping[str, Mapping[str, int]]:
    """Read the held-out labels from the sealed area. Raises unless the official environment is in force."""
    _require_sealed_environment()
    import pyarrow.parquet as pq  # noqa: PLC0415

    # A *positive* match on the sealed patterns, not "anything that is not train": once the sealed cache also
    # holds the corpus and the queries (it must, or the official run cannot load the task offline), a negative
    # filter would read a corpus shard as qrels and die on a missing `query-id` column.
    from acis.appsdata.sources import is_sealed_file  # noqa: PLC0415

    files = sorted(p for p in (sealed_root() / "hf").rglob("*.parquet") if is_sealed_file(str(p)))
    if not files:
        raise SealedDataAccess(
            f"no held-out label file under {sealed_root() / 'hf'}; the owner runs `acis fetch --sealed`"
        )
    out: dict[str, dict[str, int]] = {}
    for path in files:
        table = pq.read_table(path)
        qids = [str(v) for v in table.column("query-id").to_pylist()]
        dids = [str(v) for v in table.column("corpus-id").to_pylist()]
        scores = [int(v) for v in table.column("score").to_pylist()]
        for qid, did, score in zip(qids, dids, scores, strict=True):
            out.setdefault(qid, {})[did] = score
    return out


def verify_official(run_dir: str | Path) -> Any:
    """`verify-submission` with the re-scoring check enabled — the one place that needs the held-out labels."""
    from acis.eval.verify import verify_submission  # noqa: PLC0415

    return verify_submission(run_dir, qrels=load_holdout_qrels())


def smoke_test_offline_load() -> dict[str, Any]:
    """OWNER, BEFORE RC DAY. Prove the sealed cache can load the task offline.

    The official run is the one thing that cannot be debugged while it happens, and its very first step is a data
    load that only succeeds if the sealed area holds a *datasets* cache rather than only hub blobs. Running this
    once, after `acis fetch --sealed` and before `make rc-official`, converts that from a hope into a fact. It
    reads the held-out labels, which is why it lives here.
    """
    _require_sealed_environment()
    import os  # noqa: PLC0415

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import mteb  # noqa: PLC0415

    from acis.appsdata.sources import APPS  # noqa: PLC0415

    task: Any = mteb.get_task(APPS.task)
    task.load_data()
    loaded: Any = task.dataset["default"]
    split = next(iter(loaded))
    data = loaded[split]
    return {
        "task": APPS.task,
        "split": split,
        "n_corpus": len(data["corpus"]),
        "n_queries": len(data["queries"]),
        "n_qrels": len(data["relevant_docs"]),
        "offline": True,
    }


def main(argv: list[str] | None = None) -> int:
    """`python -m acis.eval.final <run_dir>` — the ask-gated entry point in `.claude/settings.json`."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="acis.eval.final", description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="")
    parser.add_argument("--smoke", action="store_true", help="prove the sealed cache loads the task offline")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke_test_offline_load(), indent=2, sort_keys=True))
        return 0
    if not args.run_dir:
        parser.error("a run directory is required unless --smoke is given")
    report = verify_official(args.run_dir)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
