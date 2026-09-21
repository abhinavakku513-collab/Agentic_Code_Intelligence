"""The official run (docs/spec/03 §1, §4; D17). **The owner runs this**, in their own terminal.

Everything in here is about honesty rather than cleverness:

* the process refuses to start unless `HF_HOME` points at the physical seal, so the held-out labels can only be
  loaded where they actually live (D19);
* `cache=None, overwrite_strategy="always"` — mteb's default result cache would happily hand back a *previous*
  run's numbers for an unchanged model identity (docs/spec/01 V-04);
* the JSON is written by the datetime-safe writer, because `TaskResult.to_dict()` carries a `date` that plain
  `json.dump` cannot serialise (V-07);
* `evaluation_time` is the honest cold time and is never replaced by a warm one (D17);
* the run spends one held-out touch, recorded in the ledger before anyone reads the number.

`acis.eval.final` is the sanctioned reader of the held-out labels (INV-8); this module is its entry point.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acis.appsdata.sources import APPS
from acis.core.config import FrozenConfig, load_frozen_config
from acis.core.errors import SealedDataAccess, StrictViolation
from acis.core.paths import acis_root
from acis.eval import ledger
from acis.eval.guard import official_run_env_ok
from acis.eval.runfile import write_run
from acis.eval.verify import MODE_A_JSON, MODE_B_JSON, RESULT_JSON, verify_submission, write_checksums
from acis.mteb_adapter import MODE_ENV, RUN_DIR_ENV, write_official_json
from acis.mteb_meta import revision_for

BATCH_SIZE = 64
TOP_K = 1000


@dataclass(slots=True)
class OfficialRun:
    rc: str
    mode: str
    run_dir: Path
    metrics: dict[str, float]
    evaluation_time: float
    run_id: str


def default_run_dir(rc: str) -> Path:
    return acis_root() / "runs" / f"{rc.lower()}-{time.strftime('%Y%m%dT%H%M%S')}"


def assert_official_environment(config: FrozenConfig) -> None:
    """Fail loudly before anything is loaded, rather than quietly producing a number that means nothing."""
    sealed_ok, hf_home = official_run_env_ok()
    if not sealed_ok:
        raise SealedDataAccess(
            "the official run must point HF_HOME at the physical seal "
            "(`make rc-official` sets HF_HOME=~/.acis-sealed/hf); refusing to run",
            hf_home=hf_home or "unset",
        )
    if not config.strict:
        raise StrictViolation("the official run requires run.strict: true")
    if str(config.get("run.device", "cpu")) != "cpu":
        raise StrictViolation("the official run is CPU-only (D15)")


def run_mode(config: FrozenConfig, mode: str, run_dir: Path) -> tuple[Any, float]:
    """One mode through the real mteb pipeline. Returns `(task_result, evaluation_time)`."""
    import mteb  # noqa: PLC0415

    os.environ[MODE_ENV] = mode
    os.environ[RUN_DIR_ENV] = str(run_dir)

    from acis.mteb_adapter import PrePostPipelineEncoder  # noqa: PLC0415 — after ACIS_MODE is set

    model = PrePostPipelineEncoder()
    task = mteb.get_task(APPS.task)
    started = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": BATCH_SIZE},
        cache=None,  # V-04: the default persistent cache can return a stale result
        overwrite_strategy="always",
        prediction_folder=str(run_dir / "predictions"),
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started
    task_result = _first_task_result(result)
    return task_result, elapsed


def _first_task_result(model_result: Any) -> Any:
    for attr in ("task_results", "results"):
        value = getattr(model_result, attr, None)
        if value:
            return list(value)[0]
    if isinstance(model_result, list) and model_result:
        return model_result[0]
    return model_result


def _scores_of(task_result: Any) -> dict[str, float]:
    scores = getattr(task_result, "scores", {}) or {}
    for entries in scores.values():
        if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
            return {k: float(v) for k, v in entries[0].items() if isinstance(v, (int, float))}
    return {}


def run_official(
    *,
    rc: str = "RC0",
    mode: str = "AB",
    config_path: str = "configs/official.yaml",
    out: str | None = None,
    cold: bool = True,
) -> OfficialRun:
    """Execute the official run and write every artifact of docs/spec/03 §4."""
    config = load_frozen_config(config_path)
    assert_official_environment(config)

    run_dir = Path(out) if out else default_run_dir(rc)
    run_dir.mkdir(parents=True, exist_ok=True)

    modes = ["A", "B"] if mode.upper() == "AB" else [mode.upper()]
    primary = modes[0]
    metrics: dict[str, float] = {}
    evaluation_time = 0.0

    for m in modes:
        task_result, elapsed = run_mode(config, m, run_dir)
        evaluation_time += elapsed
        write_official_json(task_result, run_dir / (MODE_A_JSON if m == "A" else MODE_B_JSON))
        if m == primary:
            write_official_json(task_result, run_dir / RESULT_JSON)
            metrics = _scores_of(task_result)

    run_id = ledger.append(
        ledger.LedgerRowBuilder(kind="rc")
        .with_metrics(metrics)
        .with_fields(
            rc=rc,
            mode=mode.upper(),
            dataset="held-out",
            config_hash=config.config_hash,
            model_revision=revision_for(config),
            cold=cold,
            evaluation_time=round(evaluation_time, 3),
            run_dir=str(run_dir),
            test_touch_count=1,
        )
        .build()
    ).run_id

    write_checksums(run_dir)
    report = verify_submission(run_dir)
    (run_dir / "verify_report.txt").write_text(report.render() + "\n", encoding="utf-8")
    return OfficialRun(
        rc=rc,
        mode=mode.upper(),
        run_dir=run_dir,
        metrics=metrics,
        evaluation_time=evaluation_time,
        run_id=run_id,
    )


def write_run_files(run: Mapping[str, Mapping[str, float]], run_dir: str | Path) -> dict[str, str]:
    return write_run(run, run_dir, top_k=TOP_K)


__all__ = ["BATCH_SIZE", "TOP_K", "OfficialRun", "assert_official_environment", "run_official", "write_run_files"]
