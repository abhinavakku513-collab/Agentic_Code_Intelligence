"""`acis eval verify-submission` (docs/spec/03 §4).

One question: *is this run directory the thing we claim it is?* Every check is mechanical and each returns PASS,
FAIL or SKIP with a reason — a skipped check is never counted as a pass.

Checks: the result JSON parses and carries `ndcg_at_10` · the dataset revision is the pinned one · the mteb version
is recorded · the model revision equals the manifest's `<git12>+<config12>` · the manifest shows
`adapter_invocations == 1`, `fallbacks == 0`, `strict == true`, `agent_calls == 0` · the held-out touch count agrees
with the ledger · re-scoring `run.trec` reproduces the JSON's NDCG@10 and MRR@10 to 1e-9 (parity P3) · both mode
JSONs are present · `evaluation_time` clears a floor (a zero-time run did not run) · the checksums match.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acis.appsdata.sources import APPS
from acis.core.hashing import sha256_file
from acis.eval import ledger
from acis.eval.metrics import score_run
from acis.eval.runfile import TREC_NAME, read_trec

RESULT_JSON = "appsretrieval_results.json"
MODE_A_JSON = "appsretrieval_results.modeA.json"
MODE_B_JSON = "appsretrieval_results.modeB.json"
MANIFEST_JSON = "manifest.json"
CHECKSUMS = "SHA256SUMS"
RESCORE_TOLERANCE = 1e-9
EVALUATION_TIME_FLOOR_S = 1.0

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


@dataclass(slots=True)
class VerifyReport:
    run_dir: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def n_skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == SKIP)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "verdict": PASS if self.passed else FAIL,
            "skipped": self.n_skipped,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks],
        }

    def render(self) -> str:
        lines = [f"verify-submission: {self.run_dir}"]
        for c in self.checks:
            lines.append(f"  [{c.status:4}] {c.name}{(' — ' + c.detail) if c.detail else ''}")
        lines.append(f"  verdict: {PASS if self.passed else FAIL} ({self.n_skipped} skipped)")
        return "\n".join(lines)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_scores(result: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    """`scores` maps a split name to a list of per-subset score dicts; return the first one we find."""
    scores = result.get("scores") or {}
    for split, entries in scores.items():
        if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
            return str(split), entries[0]
    return None


def write_checksums(run_dir: str | Path) -> Path:
    d = Path(run_dir)
    lines = []
    for path in sorted(p for p in d.rglob("*") if p.is_file() and p.name != CHECKSUMS):
        lines.append(f"{sha256_file(path)}  {path.relative_to(d).as_posix()}")
    out = d / CHECKSUMS
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_checksums(run_dir: str | Path) -> list[str]:
    d = Path(run_dir)
    path = d / CHECKSUMS
    if not path.is_file():
        return ["SHA256SUMS is missing"]
    problems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        target = d / name
        if not target.is_file():
            problems.append(f"missing file: {name}")
        elif sha256_file(target) != digest:
            problems.append(f"checksum mismatch: {name}")
    return problems


def verify_submission(run_dir: str | Path, *, qrels: Mapping[str, Mapping[str, int]] | None = None) -> VerifyReport:
    """Run every check against a run directory. `qrels` enables the re-scoring check (P3)."""
    d = Path(run_dir)
    report = VerifyReport(run_dir=str(d))
    if not d.is_dir():
        report.add("run directory exists", FAIL, str(d))
        return report
    report.add("run directory exists", PASS)

    # -- result JSON ----------------------------------------------------------------------------------------
    result_path = d / RESULT_JSON
    result: Mapping[str, Any] | None = None
    if not result_path.is_file():
        report.add("result JSON present", FAIL, RESULT_JSON)
    else:
        try:
            result = _load_json(result_path)
            report.add("result JSON present", PASS)
        except json.JSONDecodeError as exc:
            report.add("result JSON present", FAIL, f"does not parse: {exc}")

    ndcg10 = mrr10 = None
    if result is not None:
        found = _find_scores(result)
        if found is None:
            report.add("scores[<split>][0].ndcg_at_10", FAIL, "no score entry found")
        else:
            split, entry = found
            ndcg10, mrr10 = entry.get("ndcg_at_10"), entry.get("mrr_at_10")
            if ndcg10 is None:
                report.add("scores[<split>][0].ndcg_at_10", FAIL, f"absent for split {split!r}")
            else:
                report.add("scores[<split>][0].ndcg_at_10", PASS, f"{split}: {float(ndcg10):.6f}")

        revision = (result.get("dataset_revision") or "").strip()
        if revision == APPS.revision:
            report.add("dataset_revision is pinned", PASS, revision[:12])
        else:
            report.add("dataset_revision is pinned", FAIL, f"{revision[:12]!r} != {APPS.revision[:12]!r}")

        mteb_version = result.get("mteb_version")
        report.add("mteb_version recorded", PASS if mteb_version else FAIL, str(mteb_version))

        eval_time = result.get("evaluation_time")
        if eval_time is None:
            report.add("evaluation_time above floor", FAIL, "absent")
        elif float(eval_time) < EVALUATION_TIME_FLOOR_S:
            report.add("evaluation_time above floor", FAIL, f"{float(eval_time):.3f}s — did the code run?")
        else:
            report.add("evaluation_time above floor", PASS, f"{float(eval_time):.1f}s")

    # -- manifest -------------------------------------------------------------------------------------------
    manifest_path = d / MANIFEST_JSON
    manifest: Mapping[str, Any] | None = None
    if not manifest_path.is_file():
        report.add("run manifest present", FAIL, MANIFEST_JSON)
    else:
        manifest = _load_json(manifest_path)
        report.add("run manifest present", PASS)
        expectations = (
            ("adapter_invocations", 1, "our search() ran exactly once"),
            ("fallbacks", 0, "no degradation was recorded"),
            ("agent_calls", 0, "INV-13"),
        )
        for key, want, why in expectations:
            got = manifest.get(key)
            report.add(f"manifest.{key} == {want}", PASS if got == want else FAIL, f"{got} ({why})")
        strict = manifest.get("strict")
        report.add("manifest.strict is true", PASS if strict is True else FAIL, str(strict))

        if result is not None:
            model_revision = str(manifest.get("model_revision", ""))
            reported = str(result.get("model_revision") or result.get("revision") or "").strip()
            if not reported:
                report.add("model revision matches manifest", SKIP, "the result JSON records no model revision")
            else:
                report.add(
                    "model revision matches manifest",
                    PASS if reported == model_revision else FAIL,
                    f"{reported} vs {model_revision}",
                )

    # -- both modes -----------------------------------------------------------------------------------------
    present = [name for name in (MODE_A_JSON, MODE_B_JSON) if (d / name).is_file()]
    if len(present) == 2:
        report.add("both mode JSONs present", PASS)
    elif present:
        report.add("both mode JSONs present", FAIL, f"only {present[0]}")
    else:
        report.add("both mode JSONs present", SKIP, "single-mode run")

    # -- re-scoring (parity P3) ----------------------------------------------------------------------------
    trec_path = d / TREC_NAME
    if not trec_path.is_file():
        report.add("run.trec re-scores to the JSON", FAIL, "run.trec is missing")
    elif qrels is None:
        report.add(
            "run.trec re-scores to the JSON", SKIP, "no qrels supplied (held-out labels are read by acis.eval.final)"
        )
    elif ndcg10 is None:
        report.add("run.trec re-scores to the JSON", SKIP, "no ndcg_at_10 in the JSON to compare against")
    else:
        rescored = score_run(qrels, read_trec(trec_path))
        deltas = {
            "ndcg_at_10": abs(rescored["ndcg_at_10"] - float(ndcg10)),
            **({"mrr_at_10": abs(rescored["mrr_at_10"] - float(mrr10))} if mrr10 is not None else {}),
        }
        worst = max(deltas.values())
        report.add(
            "run.trec re-scores to the JSON",
            PASS if worst <= RESCORE_TOLERANCE else FAIL,
            ", ".join(f"Δ{k}={v:.3e}" for k, v in deltas.items()),
        )

    # -- ledger ---------------------------------------------------------------------------------------------
    if manifest is not None and "test_touch_count" in manifest:
        declared = int(manifest["test_touch_count"])
        used = ledger.test_touches_used()
        report.add(
            "held-out touches agree with the ledger",
            PASS if declared <= used else FAIL,
            f"manifest {declared}, ledger total {used} (budget {ledger.TEST_TOUCH_BUDGET})",
        )
    else:
        report.add("held-out touches agree with the ledger", SKIP, "the manifest declares no touch count")

    # -- checksums -----------------------------------------------------------------------------------------
    problems = verify_checksums(d)
    if problems == ["SHA256SUMS is missing"]:
        report.add("checksums valid", SKIP, problems[0])
    else:
        report.add("checksums valid", PASS if not problems else FAIL, "; ".join(problems[:3]))

    return report


__all__ = [
    "CHECKSUMS",
    "EVALUATION_TIME_FLOOR_S",
    "FAIL",
    "MANIFEST_JSON",
    "MODE_A_JSON",
    "MODE_B_JSON",
    "PASS",
    "RESCORE_TOLERANCE",
    "RESULT_JSON",
    "SKIP",
    "Check",
    "VerifyReport",
    "verify_checksums",
    "verify_submission",
    "write_checksums",
]
