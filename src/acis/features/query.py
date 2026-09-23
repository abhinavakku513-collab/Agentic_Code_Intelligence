"""Query-side features and the query↔document bridge (docs/spec/02 Appendix A, INV-15).

The dense channel sees meaning and the lexical channel sees words. Neither sees that a statement saying *"print
YES or NO"* implies a solution that prints `"YES"`, or that *"for each of t test cases"* implies a loop whose
count is read from input. Those are the bridge features, and they are the cheapest real signal available on this
task — a statement and its solution share almost no vocabulary, but they share *shape*.

Two rules govern everything here, both from INV-15:

* **Generic, never a list.** `numeric_literals` folds and returns whatever numbers a query contains; it has no
  idea which constants are interesting. `expected_outputs` returns whatever short quoted strings a query asks to
  be printed. No closed list of phrases, headers or moduli decides anything.
* **Absence is `NaN`, not zero.** A query with no numbers has no numeric overlap to report, and reporting `0.0`
  would tell the ranker "these do not match" when the truth is "there is nothing to compare". LightGBM handles
  `NaN` natively, and training masks whole feature groups so no extractor is load-bearing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from acis.features.doc import DocFeatures
from acis.prep.normalize import fold_numeric_literals

NAN = float("nan")
MAX_QUERY_CHARS = 20_000
_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,18})(?!\.?\d)"
)  # trailing "." is punctuation, not a decimal point: "modulo 10^9+7." ends in a number
_QUOTED = re.compile(r"""["'`]([^"'`\n]{1,30})["'`]""")
#: Words that *suggest* a shape. They raise a prior; they never gate anything, and a query using none of them
#: still gets every feature (the value is simply absent, not zero).
_INPUT_HINT = re.compile(r"\b(input|stdin|read|given|receive)\w*\b", re.IGNORECASE)
_OUTPUT_HINT = re.compile(r"\b(output|print|return|display)\w*\b", re.IGNORECASE)
_TESTCASE_HINT = re.compile(r"\b(test\s*cases?|queries|t\s+lines|each\s+case)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """What a query implies about the shape of its answer."""

    n_chars: int
    n_tokens: int
    numeric_literals: frozenset[int]
    expected_outputs: frozenset[str]
    mentions_input: bool
    mentions_output: bool
    mentions_testcases: bool

    def as_row(self) -> dict[str, Any]:
        return {
            "n_chars": self.n_chars,
            "n_tokens": self.n_tokens,
            "numeric_literals": sorted(self.numeric_literals),
            "expected_outputs": sorted(self.expected_outputs),
            "mentions_input": self.mentions_input,
            "mentions_output": self.mentions_output,
            "mentions_testcases": self.mentions_testcases,
        }


def extract(text: str) -> QueryFeatures:
    """Features for one query. Never raises; an empty query yields empty sets rather than an error."""
    source = (text or "")[:MAX_QUERY_CHARS]
    folded = fold_numeric_literals(source)
    numbers = {int(m.group(1)) for m in _NUMBER.finditer(folded) if len(m.group(1)) <= 18}
    quoted = {m.group(1).strip().lower() for m in _QUOTED.finditer(source) if m.group(1).strip()}
    return QueryFeatures(
        n_chars=len(source),
        n_tokens=len(source.split()),
        numeric_literals=frozenset(numbers),
        expected_outputs=frozenset(q for q in quoted if len(q) <= 30),
        mentions_input=bool(_INPUT_HINT.search(source)),
        mentions_output=bool(_OUTPUT_HINT.search(source)),
        mentions_testcases=bool(_TESTCASE_HINT.search(source)),
    )


# -- the bridge ----------------------------------------------------------------------------------------------
def _jaccard(a: frozenset[Any], b: frozenset[Any]) -> float:
    """Absent on either side is `NaN`: "nothing to compare" is not the same as "no overlap"."""
    if not a or not b:
        return NAN
    union = a | b
    return len(a & b) / len(union) if union else NAN


def bridge(query: QueryFeatures, doc: DocFeatures) -> dict[str, float]:
    """Features that only exist for a (query, document) pair. Every one of them may be `NaN`.

    `out_literal_recall` is the sharpest of them on this task: a statement that says to print `"YES"` and a
    solution that prints `"YES"` agree on something no embedding of either one can see.
    """
    out_recall = NAN
    if query.expected_outputs:
        out_recall = (
            len(query.expected_outputs & doc.output_literals) / len(query.expected_outputs)
            if doc.output_literals
            else 0.0
        )

    io_shape = NAN
    if query.mentions_input:
        io_shape = 1.0 if (doc.reads_input or doc.reads_stdin) else 0.0

    tc_expected = NAN
    if query.mentions_testcases:
        tc_expected = 1.0 if doc.has_testcase_loop else 0.0

    return {
        "out_literal_recall": out_recall,
        "numeric_literal_overlap": _jaccard(query.numeric_literals, doc.numeric_constants),
        "const_jaccard": _jaccard(
            frozenset(n for n in query.numeric_literals if n >= 10),
            frozenset(n for n in doc.numeric_constants if n >= 10),
        ),
        "io_shape_compat": io_shape,
        "tc_loop_expected": tc_expected,
    }


#: The bridge group, named so training can mask it as a unit (≈25 % group dropout, spec 02 §4).
BRIDGE_FEATURES = (
    "out_literal_recall",
    "numeric_literal_overlap",
    "const_jaccard",
    "io_shape_compat",
    "tc_loop_expected",
)


def availability(features: dict[str, float]) -> float:
    """Share of bridge features that actually fired — the ρ the router and the abstaining ranker read."""
    values = [features.get(name, NAN) for name in BRIDGE_FEATURES]
    fired = sum(1 for v in values if v == v)  # NaN != NaN
    return fired / len(values) if values else 0.0


__all__ = ["BRIDGE_FEATURES", "MAX_QUERY_CHARS", "NAN", "QueryFeatures", "availability", "bridge", "extract"]
