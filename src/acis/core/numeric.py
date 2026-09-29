"""Numeric profiles and thread control (docs/spec/02 §3, INV-6).

`cpu-fp32` is the reference profile and the only one the official JSON may use before gate G6. Thread counts are an
operational knob (they do not change `config_hash`) but they *are* recorded in every ledger row, because the
determinism claim is "same machine, same config, same thread count".
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

PROFILES = ("cpu-fp32", "cpu-bf16", "cpu-int8", "gpu-fp16")
REFERENCE_PROFILE = "cpu-fp32"
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True, slots=True)
class NumericProfile:
    name: str
    dtype: str
    device: str
    is_reference: bool

    @property
    def needs_gate_g6(self) -> bool:
        return not self.is_reference


def get_profile(name: str = REFERENCE_PROFILE) -> NumericProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown numeric profile {name!r}; known: {PROFILES}")
    device, _, dtype = name.partition("-")
    return NumericProfile(name=name, dtype=dtype, device=device, is_reference=name == REFERENCE_PROFILE)


def physical_cores() -> int:
    """Physical core count, falling back to logical cores then 1. Never raises."""
    try:
        import subprocess  # noqa: PLC0415 — only used on the slow path

        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            cores = {ln for ln in out.stdout.splitlines() if ln and not ln.startswith("#")}
            if cores:
                return len(cores)
    except Exception:  # noqa: BLE001 — hardware probing is best-effort by design
        return os.cpu_count() or 1
    return os.cpu_count() or 1


def resolve_threads(setting: object = "auto_physical") -> int:
    if isinstance(setting, int) and setting > 0:
        return setting
    if isinstance(setting, str) and setting.isdigit() and int(setting) > 0:
        return int(setting)
    return max(1, physical_cores())


def apply_threads(threads: int) -> int:
    """Pin BLAS/OpenMP (and torch, if already imported) to `threads`. Returns the value actually applied."""
    threads = max(1, int(threads))
    for var in _THREAD_ENV:
        os.environ[var] = str(threads)
    import sys  # noqa: PLC0415

    torch = sys.modules.get("torch")
    if torch is not None:  # never import torch just to set threads (keeps the CLI fast and offline)
        with contextlib.suppress(Exception):
            torch.set_num_threads(threads)
    return threads


#: numpy/scipy BLAS threads for the retrieval core: **one**. A query's dense scores are two matrix-vector products
#: (8,765 and 5,000 rows): memory-bound, ~5 ms on one thread. On the 16 hyper-threads OpenBLAS starts with, waking
#: its pool after an idle second cost ~60 ms — more than the work — and its spinning threads slowed the next torch
#: forward pass by ~70 ms (measured on the dev host, `scripts/bench/run.py`). One thread is also the only count
#: whose scores do not depend on the host: OpenBLAS splits a product differently per thread count, which moves a
#: score by up to ~5e-7 and can flip an exact near-tie (INV-6, `tests/metamorphic/test_blas_threads.py`).
MAX_BLAS_THREADS = 1


def apply_blas_threads(threads: int) -> int | None:
    """Cap numpy/scipy BLAS at `min(threads, MAX_BLAS_THREADS)`. Returns the cap, or `None` if it cannot be set.

    Fails soft: without `threadpoolctl` (a scikit-learn dependency) the process keeps its default pool, which is
    slower and host-dependent at the ~5e-7 level; the engine records the cap it got in `blas_threads`.
    """
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return None
    cap = max(1, min(int(threads), MAX_BLAS_THREADS))
    threadpool_limits(limits=cap, user_api="blas")
    return cap


__all__ = [
    "MAX_BLAS_THREADS",
    "PROFILES",
    "REFERENCE_PROFILE",
    "NumericProfile",
    "apply_blas_threads",
    "apply_threads",
    "get_profile",
    "physical_cores",
    "resolve_threads",
]
