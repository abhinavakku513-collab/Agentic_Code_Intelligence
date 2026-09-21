"""`acis doctor` — hardware profile, tier classification and environment checks (Phase 0, G0.3/G0.6).

Writes `runs/hardware.json`: the declared reference host behind every scorecard number. Nothing here is an accuracy
claim; it records facts about the machine plus a small, honest GEMM throughput measurement.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from acis.appsdata.fetch import manifest_path, scan_for_sealed
from acis.core.numeric import physical_cores, resolve_threads
from acis.core.paths import acis_home, acis_root, sealed_root

T_MIN = {"cores": 4, "ram_gb": 8}
T_REC = {"cores": 8, "ram_gb": 16}


def _ram_gb() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 2)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _isa_flags() -> list[str]:
    wanted = ("avx2", "avx512f", "avx512_vnni", "amx_bf16", "neon", "asimd", "sve")
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            text = fh.read().lower()
    except OSError:
        return []
    return [flag for flag in wanted if f" {flag} " in text or f" {flag}\n" in text]


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=acis_root(), capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=acis_root(), capture_output=True, text=True, timeout=10
            ).stdout.strip()
        )
        return sha or "unknown", dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def gemm_gflops(n: int = 1024, repeats: int = 3) -> float:
    """Sustained fp32 GEMM throughput — the single number that projects the cold official pass (G0.3)."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    a @ b  # warm up BLAS
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return round(2 * n**3 / best / 1e9, 2)


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out: dict[str, str] = {"python": platform.python_version()}
    for pkg in (
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "mteb",
        "datasets",
        "lightgbm",
        "bm25s",
        "pytrec-eval-terrier",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "absent"
    return out


def tier(cores: int, ram_gb: float) -> str:
    if cores >= T_REC["cores"] and ram_gb >= T_REC["ram_gb"]:
        return "T-rec"
    if cores >= T_MIN["cores"] and ram_gb >= T_MIN["ram_gb"]:
        return "T-min"
    return "below-T-min"


def environment_checks() -> dict[str, Any]:
    """Facts the seal and the offline policy depend on (G0.6)."""
    sealed_leaks = scan_for_sealed()
    return {
        "acis_root": str(acis_root()),
        "acis_home": str(acis_home()),
        "sealed_root": str(sealed_root()),
        "sealed_root_exists": sealed_root().exists(),
        "sealed_files_in_dev_env": sealed_leaks,
        "seal_ok": not sealed_leaks,
        "dataset_manifest": str(manifest_path()) if manifest_path().is_file() else None,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "unset"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE", "unset"),
        "disk_free_gb": round(shutil.disk_usage(acis_root()).free / 1024**3, 2),
    }


def collect(*, with_gemm: bool = True) -> dict[str, Any]:
    cores = physical_cores()
    ram = _ram_gb()
    sha, dirty = _git_sha()
    report: dict[str, Any] = {
        "ts": time.time(),
        "git_sha": sha,
        "dirty": dirty,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "hardware": {
            "cpu": _cpu_model(),
            "physical_cores": cores,
            "logical_cores": os.cpu_count() or cores,
            "ram_gb": ram,
            "isa": _isa_flags(),
        },
        "threads": resolve_threads("auto_physical"),
        "tier": tier(cores, ram),
        "gpu": "none",
        "packages": package_versions(),
        "environment": environment_checks(),
    }
    if with_gemm:
        report["throughput"] = {"gemm_fp32_gflops_1024": gemm_gflops()}
    return report


def write_report(report: dict[str, Any] | None = None) -> str:
    report = report if report is not None else collect()
    out = acis_root() / "runs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hardware.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


__all__ = ["T_MIN", "T_REC", "collect", "environment_checks", "gemm_gflops", "package_versions", "tier", "write_report"]
