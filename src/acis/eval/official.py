"""The official run (docs/spec/03 §1, §4; D17). **The owner runs this**, in their own terminal.

Everything in here is about honesty rather than cleverness:

* on a machine that has a sealed area the process refuses to start unless `HF_HOME` points at it, so the held-out
  labels can only be loaded where they actually live (D19);
* a release candidate refuses to run from a dirty working tree: its `git_sha` has to rebuild the artifact;
* the held-out budget is checked **before** the expensive pass, not after it — an exhausted budget must not cost a
  real touch;
* `cache=None, overwrite_strategy="always"` — mteb's default result cache would hand back a *previous* run's
  numbers for an unchanged model identity (docs/spec/01 V-04);
* the JSON is written by the datetime-safe writer, because `TaskResult.to_dict()` carries a `date` that plain
  `json.dump` cannot serialise (V-07);
* `evaluation_time` is the honest cold time **of the primary mode**, never a sum and never a warm time (D17);
* the run books one held-out touch **per mode**, and the manifest declares the same figure so `verify-submission`
  can cross-check it.

This path is the one thing that cannot be debugged on the day, so `run_pipeline` takes the task and the touch
accounting as parameters: `tests/integration/test_official_dry_run.py` drives the whole thing — config loading,
both modes, run files, manifest, checksums, verification — over a synthetic task with no held-out data in sight.

`acis.eval.final` is the sanctioned reader of the held-out labels (INV-8); this module is its entry point.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acis.appsdata.sources import APPS
from acis.core.config import FrozenConfig, load_frozen_config
from acis.core.errors import InvalidInput, SealedDataAccess, StrictViolation
from acis.core.paths import acis_root, sealed_root
from acis.eval import ledger
from acis.eval.guard import official_run_env_ok
from acis.eval.runfile import write_run
from acis.eval.verify import MANIFEST_JSON, MODE_A_JSON, MODE_B_JSON, RESULT_JSON, verify_submission, write_checksums
from acis.mteb_adapter import CONFIG_ENV, MODE_ENV, RUN_DIR_ENV, TOUCH_ENV, write_official_json
from acis.mteb_meta import revision_for

BATCH_SIZE = 64
TOP_K = 1000
PREDICTIONS_DIR = "predictions"


@dataclass(slots=True)
class ModeResult:
    mode: str
    task_result: Any
    evaluation_time: float
    run: dict[str, dict[str, float]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OfficialRun:
    rc: str
    mode: str
    primary_mode: str
    run_dir: Path
    metrics: dict[str, float]
    evaluation_time: float
    per_mode_seconds: Mapping[str, float]
    run_id: str
    verified: bool


def default_run_dir(rc: str) -> Path:
    return acis_root() / "runs" / f"{rc.lower()}-{time.strftime('%Y%m%dT%H%M%S')}"


# -- preconditions -------------------------------------------------------------------------------------------------
def assert_official_environment(config: FrozenConfig, *, ledgered: bool = True, touches: int = 1) -> None:
    """Fail loudly before anything is loaded, rather than quietly producing a number that means nothing.

    The seal requirement applies wherever the seal **exists**. On the owner's machine the held-out labels live in
    the sealed area, so a run reading them from anywhere else is reading something it should not. A judge's machine
    has no sealed area and the labels come from an ordinary cache, so `make reproduce` works there — that path is a
    reproduction, not a ledgered release candidate.
    """
    sealed_ok, hf_home = official_run_env_ok()
    if sealed_root().exists() and not sealed_ok:
        raise SealedDataAccess(
            "this machine has a sealed area, so the official run must point HF_HOME at it "
            "(`make rc-official` does); refusing to run",
            hf_home=hf_home or "unset",
        )
    if not config.strict:
        raise StrictViolation("the official run requires run.strict: true")
    if str(config.get("run.device", "cpu")) != "cpu":
        raise StrictViolation("the official run is CPU-only (D15)")
    if not ledgered:
        return
    if _working_tree_is_dirty():
        raise StrictViolation(
            "the working tree has uncommitted changes; a release candidate must be rebuildable from its git_sha "
            "(docs/spec/03 §6). Commit first, or set ACIS_ALLOW_DIRTY_RC=1 to override deliberately."
        )
    used = ledger.test_touches_used()
    if used + touches > ledger.TEST_TOUCH_BUDGET:
        raise InvalidInput(
            "this run would exceed the held-out touch budget — refusing before it is spent, not after",
            used=used,
            requested=touches,
            budget=ledger.TEST_TOUCH_BUDGET,
        )


def _working_tree_is_dirty() -> bool:
    import subprocess  # noqa: PLC0415

    if os.environ.get("ACIS_ALLOW_DIRTY_RC") == "1":
        return False
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=acis_root(), capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


# -- one mode ----------------------------------------------------------------------------------------------------
def run_mode(
    config_path: str,
    mode: str,
    run_dir: Path,
    *,
    touches: int,
    task_factory: Callable[[], Any] | None = None,
) -> ModeResult:
    """One mode through the real mteb pipeline, with the ranking captured so run files can be written.

    The config path is passed to the model **explicitly**. Relying on the ambient `ACIS_CONFIG` default would mean
    the environment check validated `configs/official.yaml` while the model quietly ran on `configs/dev.yaml` —
    different strictness, different encoder, different config hash in the manifest.
    """
    import mteb  # noqa: PLC0415

    os.environ[MODE_ENV] = mode
    os.environ[CONFIG_ENV] = config_path
    os.environ[RUN_DIR_ENV] = str(run_dir)
    os.environ[TOUCH_ENV] = str(touches)

    from acis.mteb_adapter import PrePostPipelineEncoder  # noqa: PLC0415 — after ACIS_MODE is set

    model = PrePostPipelineEncoder(config_path)
    task = task_factory() if task_factory is not None else mteb.get_task(APPS.task)

    started = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": BATCH_SIZE},
        cache=None,  # V-04: the default persistent cache can return a stale result
        overwrite_strategy="always",
        prediction_folder=str(run_dir / PREDICTIONS_DIR),
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started

    task_result = _first_task_result(result)
    # Mode B is graded by mteb's own wrapper, which never calls our `search()`, so its manifest is written here.
    manifest_path = model.write_run_manifest(task.metadata, _split_of(task_result), suffix=mode)
    return ModeResult(
        mode=mode,
        task_result=task_result,
        evaluation_time=elapsed,
        run=_load_predictions(run_dir, task),
        manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8")),
    )


def _split_of(task_result: Any) -> str:
    scores = getattr(task_result, "scores", {}) or {}
    return str(next(iter(scores), "test"))


def _first_task_result(model_result: Any) -> Any:
    for attr in ("task_results", "results"):
        value = getattr(model_result, attr, None)
        if value:
            return list(value)[0]
    if isinstance(model_result, list) and model_result:
        return model_result[0]
    raise InvalidInput(
        "mteb returned no task result — the evaluation failed before scoring; read the log before rerunning",
        got=type(model_result).__name__,
    )


def _load_predictions(run_dir: Path, task: Any) -> dict[str, dict[str, float]]:
    """Read back the rankings mteb saved, so `run.trec` is the ranking that was actually graded."""
    folder = run_dir / PREDICTIONS_DIR
    if not folder.is_dir():
        return {}
    name = str(getattr(task.metadata, "name", ""))
    candidates = sorted(folder.rglob(f"{name}*predictions.json")) or sorted(folder.rglob("*predictions.json"))
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload.get("predictions", payload) if isinstance(payload, Mapping) else payload
        if isinstance(run, Mapping):
            flat = _flatten_predictions(run)
            if flat:
                return flat
    return {}


def _flatten_predictions(run: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """mteb nests predictions per split/subset in some versions and not in others; accept both shapes."""
    out: dict[str, dict[str, float]] = {}
    for key, value in run.items():
        if not isinstance(value, Mapping) or not value:
            continue
        first = next(iter(value.values()))
        if isinstance(first, Mapping):  # {split: {qid: {doc: score}}}
            out.update(_flatten_predictions(value))
        elif isinstance(first, (int, float)):  # {qid: {doc: score}}
            out[str(key)] = {str(d): float(s) for d, s in value.items()}
    return out


def _scores_of(task_result: Any) -> dict[str, float]:
    scores = getattr(task_result, "scores", {}) or {}
    for entries in scores.values():
        if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
            return {k: float(v) for k, v in entries[0].items() if isinstance(v, (int, float))}
    return {}


# -- the whole run -------------------------------------------------------------------------------------------------
def primary_mode_for(config: FrozenConfig, modes: list[str]) -> str:
    """Which mode's result becomes `appsretrieval_results.json`.

    Not simply "the first one we ran": D8 makes RC0 Mode B, and gate G-AB decides the primary at RC1. The frozen
    config states it (`run.primary_mode`, falling back to `run.mode`), and a single-mode run is its own primary.
    """
    if len(modes) == 1:
        return modes[0]
    declared = str(config.get("run.primary_mode", "")).upper()
    if declared in modes:
        return declared
    configured = config.mode
    return configured if configured in modes else modes[-1]


def run_pipeline(
    *,
    rc: str,
    mode: str,
    config_path: str,
    run_dir: Path,
    cold: bool = True,
    ledgered: bool = True,
    task_factory: Callable[[], Any] | None = None,
) -> OfficialRun:
    """Execute the run and write every artifact of docs/spec/03 §4."""
    config = load_frozen_config(config_path)
    modes = ["A", "B"] if mode.upper() == "AB" else [mode.upper()]
    assert_official_environment(config, ledgered=ledgered, touches=len(modes))

    run_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_mode_for(config, modes)

    results: dict[str, ModeResult] = {}
    for m in modes:
        results[m] = run_mode(config_path, m, run_dir, touches=len(modes), task_factory=task_factory)
        write_official_json(results[m].task_result, run_dir / (MODE_A_JSON if m == "A" else MODE_B_JSON))

    primary_result = results[primary]
    write_official_json(primary_result.task_result, run_dir / RESULT_JSON)
    metrics = _scores_of(primary_result.task_result)

    # The run files hold the primary mode's ranking: `verify-submission` re-scores them against the JSON (P3).
    if primary_result.run:
        write_run(primary_result.run, run_dir, top_k=TOP_K)
    (run_dir / MANIFEST_JSON).write_text(json.dumps(primary_result.manifest, indent=2), encoding="utf-8")

    per_mode = {m: round(r.evaluation_time, 3) for m, r in results.items()}
    run_id = ""
    if ledgered:
        run_id = ledger.append(
            ledger.LedgerRowBuilder(kind="rc")
            .with_metrics(metrics)
            .with_fields(
                rc=rc,
                mode=mode.upper(),
                primary_mode=primary,
                dataset="held-out",
                decision_set="held_out",
                config_hash=config.config_hash,
                model_revision=revision_for(config),
                cold=cold,
                evaluation_time=per_mode[primary],  # the primary mode's own cold time, never a sum (D17)
                evaluation_time_per_mode=per_mode,
                run_dir=str(run_dir),
                # One held-out touch per mode: an A+B run spends two of the six (CLAUDE.md §4).
                test_touch_count=len(modes),
            )
            .build()
        ).run_id

    report = verify_submission(run_dir)
    (run_dir / "verify_report.txt").write_text(report.render() + "\n", encoding="utf-8")
    write_checksums(run_dir)  # after the report, so every artifact is covered
    return OfficialRun(
        rc=rc,
        mode=mode.upper(),
        primary_mode=primary,
        run_dir=run_dir,
        metrics=metrics,
        evaluation_time=per_mode[primary],
        per_mode_seconds=per_mode,
        run_id=run_id,
        verified=report.passed,
    )


def run_official(
    *,
    rc: str = "RC0",
    mode: str = "AB",
    config_path: str = "configs/official.yaml",
    out: str | None = None,
    cold: bool = True,
    cache_verify: bool = False,
) -> OfficialRun:
    """The owner-facing entry point. `cache_verify` reproduces rankings from shipped caches and books no touch."""
    run_dir = Path(out) if out else default_run_dir("cache-verify" if cache_verify else rc)
    return run_pipeline(
        rc=rc,
        mode=mode,
        config_path=config_path,
        run_dir=run_dir,
        cold=cold and not cache_verify,
        ledgered=not cache_verify,
    )


__all__ = [
    "BATCH_SIZE",
    "PREDICTIONS_DIR",
    "TOP_K",
    "ModeResult",
    "OfficialRun",
    "assert_official_environment",
    "default_run_dir",
    "primary_mode_for",
    "run_mode",
    "run_official",
    "run_pipeline",
]
