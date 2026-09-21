"""Arbitrary-query robustness contract (docs/spec/10). The engine is plugged in through a tiny hook:

    ACIS_ROBUSTNESS_ENGINE="module:function"      (default: acis.robust_hook:search)
    def search(query: str, top_k: int = 10) -> list[str]     # ranked ids over a FIXED fixture corpus (>= 200 docs for real runs)
    raises an exception whose text contains 'InvalidInput' for empty/whitespace-only queries

Env knobs: ACIS_ROBUSTNESS_K (default 10), ACIS_ROBUSTNESS_MILD_MIN (0.8), ACIS_ROBUSTNESS_MODERATE_MIN (0.5),
ACIS_ROBUSTNESS_QUERIES (jsonl of {"q": ...} overriding the built-in seeds; use DEV-CV/REG samples for real runs).
These tests check PROPERTIES that need no labels. Accuracy under perturbation is measured by `acis eval robustness` (gate G-OOD).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perturb  # noqa: E402

K = int(os.environ.get("ACIS_ROBUSTNESS_K", "10"))
MILD_MIN = float(os.environ.get("ACIS_ROBUSTNESS_MILD_MIN", "0.8"))
MODERATE_MIN = float(os.environ.get("ACIS_ROBUSTNESS_MODERATE_MIN", "0.5"))

# Deliberately DIVERSE seeds (no single template). Real runs replace them via ACIS_ROBUSTNESS_QUERIES.
SEEDS = [
    "Given a string s, print YES if it is a palindrome and NO otherwise.\n\n-----Input-----\nThe first line contains t, the number of test cases.\nEach of the next t lines contains a string s.\n\n-----Output-----\nFor each test case print YES or NO.\n\n-----Examples-----\nInput\n2\nabba\nabc\nOutput\nYES\nNO",
    "How is the input preprocessed before going to the main function?",
    "normalize",
    "def gcd(a, b): while b: a, b = b, a % b",
    "reads a number of test cases and prints YES or NO for each string",
    "## Task\n\nCompute n! modulo a large prime.\n\n## Constraints:\n\nn can be up to one million.\n\nThe answer must be printed on a single line.",
    "find code taht computes the factoral modulo 1000000007",
    "Implement a breadth first search over an adjacency list and return the visited vertices in order.",
]
if os.environ.get("ACIS_ROBUSTNESS_QUERIES"):
    SEEDS = [
        json.loads(ln)["q"] for ln in Path(os.environ["ACIS_ROBUSTNESS_QUERIES"]).read_text().splitlines() if ln.strip()
    ]


@pytest.fixture(scope="module")
def search():
    spec = os.environ.get("ACIS_ROBUSTNESS_ENGINE", "acis.robust_hook:search")
    mod, _, fn = spec.partition(":")
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"engine hook {spec!r} not available yet ({e})")


def overlap(a: list[str], b: list[str], k: int = K) -> float:
    return len(set(a[:k]) & set(b[:k])) / max(1, min(k, len(a), len(b)))


# ---- 1. the perturbation library itself (no engine needed) --------------------------------------------------------------
@pytest.mark.parametrize("name,op", {**perturb.MILD, **perturb.MODERATE}.items())
def test_operators_are_deterministic_and_nonempty(name, op):
    for q in SEEDS:
        assert op(q, seed=3) == op(q, seed=3)
        assert isinstance(op(q, seed=3), str) and op(q, seed=3).strip()


def test_operators_contain_no_benchmark_or_template_vocabulary():
    src = (HERE / "perturb.py").read_text()
    assert not re.search(r"\b(Input|Output|Examples?|Constraints?|APPS|problem statement)\b", src.split('"""', 2)[2])


# ---- 2. engine properties -------------------------------------------------------------------------------------------------
def test_determinism(search):
    for q in SEEDS:
        assert search(q, top_k=K) == search(q, top_k=K)


def test_format_noise_is_a_strict_no_op(search):
    for q in SEEDS:
        for seed in range(3):
            assert search(perturb.format_noise(q, seed), top_k=K) == search(q, top_k=K), (
                f"format noise changed ranking for {q[:40]!r}"
            )


@pytest.mark.parametrize("name", ["strip_headings", "lower"])
def test_mild_structure_changes_keep_topk(search, name):
    """A generic engine must not flip because heading lines are absent or capitalisation differs."""
    for q in SEEDS:
        p = perturb.MILD[name](q, seed=0)
        if p != q:
            assert overlap(search(q, top_k=K), search(p, top_k=K)) >= MILD_MIN, (
                f"{name} flipped the ranking for {q[:50]!r}"
            )


@pytest.mark.parametrize("name", list(perturb.MODERATE))
def test_moderate_perturbations_degrade_gracefully(search, name):
    vals = []
    for q in SEEDS:
        for seed in range(3):
            p = perturb.MODERATE[name](q, seed=seed)
            if p != q and p.strip():
                vals.append(overlap(search(q, top_k=K), search(p, top_k=K)))
    if vals:
        assert sum(vals) / len(vals) >= MODERATE_MIN, f"{name}: mean top-{K} overlap {sum(vals) / len(vals):.2f}"


@pytest.mark.parametrize("name,text", list(perturb.hostile_queries().items()))
def test_hostile_queries_never_crash(search, name, text):
    if not text.strip():
        with pytest.raises(Exception, match="InvalidInput"):
            search(text, top_k=K)
        return
    out = search(text, top_k=K)
    assert isinstance(out, list) and len(out) <= K and len(set(out)) == len(out)
