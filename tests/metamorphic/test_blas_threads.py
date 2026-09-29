"""The retrieval core's dense scores do not depend on the host's core count (INV-6).

OpenBLAS splits a matrix-vector product differently for each thread count, and the float32 results differ by up
to ~5e-7 — enough to flip an exact near-tie. The engine therefore pins numpy's BLAS to one thread
(`acis.core.numeric.MAX_BLAS_THREADS`), which is also the fastest setting for one query at a time. These tests pin
both halves: that the difference is real (so the cap is not decoration) and that the engine applies the cap.
"""

from __future__ import annotations

import numpy as np
import pytest

threadpoolctl = pytest.importorskip("threadpoolctl")

from acis.core.config import freeze_config  # noqa: E402
from acis.core.numeric import MAX_BLAS_THREADS, apply_blas_threads  # noqa: E402
from acis.embed.base import exact_search  # noqa: E402


def _scores(threads: int, matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
        return exact_search(query, matrix)


def _problem(rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rows)
    matrix = rng.standard_normal((rows, 768)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, rng.standard_normal((1, 768)).astype(np.float32)


def test_one_thread_is_reproducible():
    matrix, query = _problem(8765)
    assert np.array_equal(_scores(1, matrix, query), _scores(1, matrix, query))


def test_the_thread_count_can_move_a_score_which_is_why_it_is_pinned():
    matrix, query = _problem(8765)
    one = _scores(1, matrix, query)
    diffs = [float(np.abs(one - _scores(t, matrix, query)).max()) for t in (2, 4, 8)]
    assert max(diffs) < 1e-5  # rounding, never a different answer
    # Not asserted to be non-zero: a BLAS build may be deterministic across threads, and that is fine too.


def test_the_engine_applies_the_cap():
    from acis.engine import AcisEngine
    from acis.engine.core import DEFAULT_CONFIG

    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG))
    assert engine.blas_threads == MAX_BLAS_THREADS == 1
    matrix, query = _problem(400)
    assert np.array_equal(exact_search(query, matrix), _scores(1, matrix, query))
    assert apply_blas_threads(64) == 1
