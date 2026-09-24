"""Document, query and bridge features (docs/spec/02 Appendix A, INV-5, INV-15).

The bridge is the point: a statement and its solution share almost no vocabulary but they share shape, and these
are the only features that can see it. So the tests are pairs — a statement that says "print YES or NO" against a
solution that prints it — plus the two rules that keep the features honest: nothing is executed, and an absent
signal is `NaN` rather than a confident zero.
"""

from __future__ import annotations

import math

from acis.features import doc as docf
from acis.features import query as qf

SOLUTION = """
import sys
from collections import deque

MOD = 10**9 + 7

def solve():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        if n % k == 0:
            print("YES")
        else:
            print("NO")

solve()
"""


def test_parsing_reads_the_shape_of_a_solution():
    f = docf.extract(SOLUTION)
    assert f.parse_ok
    assert {"sys", "collections"} <= f.imports
    assert f.reads_input >= 2 and f.splits_input and f.maps_int
    assert f.has_testcase_loop
    assert {"yes", "no"} <= f.output_literals
    assert 1000000007 in f.numeric_constants


def test_an_unparseable_file_is_still_described_and_never_raises():
    f = docf.extract("def broken(:\n  print('YES')\n  x = 1000000007\n")
    assert not f.parse_ok
    assert "yes" in f.output_literals and 1000000007 in f.numeric_constants


def test_extraction_never_executes_the_code(tmp_path):
    """INV-5: the file would write a sentinel if anything imported or executed it."""
    sentinel = tmp_path / "EXECUTED"
    hostile = f"import pathlib\npathlib.Path({str(sentinel)!r}).write_text('x')\n"
    docf.extract(hostile)
    assert not sentinel.exists()


def test_recursion_and_memoisation_are_recognised():
    f = docf.extract("from functools import lru_cache\n\n@lru_cache\ndef fib(n):\n    return fib(n-1)+fib(n-2)\n")
    assert f.recursive and f.memoised and "lru_cache" in f.idioms


def test_a_query_reports_what_it_implies():
    q = qf.extract('For each of t test cases print "YES" or "NO". Output the answer modulo 10^9+7.')
    assert q.mentions_testcases and q.mentions_output
    assert {"yes", "no"} <= q.expected_outputs
    assert 1000000007 in q.numeric_literals


def test_the_bridge_connects_a_statement_to_its_solution():
    q = qf.extract('Given t test cases, read n and k from input and print "YES" or "NO". Use modulo 10^9+7.')
    right = qf.bridge(q, docf.extract(SOLUTION))
    assert right["out_literal_recall"] == 1.0
    assert right["io_shape_compat"] == 1.0
    assert right["tc_loop_expected"] == 1.0
    assert right["numeric_literal_overlap"] > 0.0


def test_the_bridge_separates_a_wrong_solution_from_a_right_one():
    q = qf.extract('Print "YES" or "NO" for each of t test cases.')
    wrong = qf.bridge(q, docf.extract("def add(a, b):\n    return a + b\n"))
    right = qf.bridge(q, docf.extract(SOLUTION))
    assert wrong["out_literal_recall"] == 0.0 < right["out_literal_recall"]
    assert wrong["tc_loop_expected"] == 0.0 < right["tc_loop_expected"]


def test_an_absent_signal_is_nan_not_a_confident_zero():
    """A query with no quoted output has nothing to compare; saying 0.0 would be a claim it cannot support."""
    q = qf.extract("Compute the shortest path length.")
    features = qf.bridge(q, docf.extract(SOLUTION))
    assert math.isnan(features["out_literal_recall"])
    assert math.isnan(features["tc_loop_expected"])
    assert math.isnan(features["io_shape_compat"]) or features["io_shape_compat"] in (0.0, 1.0)


def test_availability_counts_the_features_that_fired():
    rich = qf.bridge(qf.extract('t test cases, read input, print "YES", modulo 10^9+7'), docf.extract(SOLUTION))
    bare = qf.bridge(qf.extract("Solve it."), docf.extract(SOLUTION))
    assert qf.availability(rich) > qf.availability(bare)
    assert 0.0 <= qf.availability(bare) <= 1.0


def test_features_are_generic_rather_than_a_list_of_known_constants():
    """INV-15: an unfamiliar modulus behaves exactly like the famous one."""
    q = qf.extract("print the result modulo 998244353")
    f = docf.extract("MOD = 998244353\nprint(1 % MOD)\n")
    assert 998244353 in q.numeric_literals and 998244353 in f.numeric_constants
    assert qf.bridge(q, f)["numeric_literal_overlap"] > 0.0


def test_a_long_constant_expression_does_not_take_quadratic_time():
    """Found in the corpus: an APPS document that is one arithmetic expression tens of thousands of terms long.

    Folding constants naively visits every node and re-descends the whole chain beneath it, so extraction on that
    file did not finish in an hour. The depth bound makes it linear, and the bound costs nothing real: `10**9 + 7`
    is depth two.
    """
    import time

    for terms in (2_000, 20_000):
        chain = "x = " + " + ".join(str(i % 97) for i in range(terms)) + "\n"
        started = time.perf_counter()
        features = docf.extract(chain)
        elapsed = time.perf_counter() - started
        assert elapsed < 5.0, f"extraction took {elapsed:.1f}s on a {terms}-term constant expression"
        # Either answer is correct and both are useful: a chain deep enough to exhaust Python's own recursion
        # limit is reported as unparseable and still described, which is the contract (spec 02 §2).
        assert features.n_tokens > 0


def test_the_depth_bound_does_not_cost_the_constant_that_matters():
    """The fold exists for `10**9 + 7`; bounding it must leave that untouched."""
    assert 1000000007 in docf.extract("MOD = 10**9 + 7\n").numeric_constants
    assert 998244353 in docf.extract("m = 998244353\n").numeric_constants
