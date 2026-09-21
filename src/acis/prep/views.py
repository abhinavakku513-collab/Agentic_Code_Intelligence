"""Query segmentation and views (docs/spec/02 §2, §5; INV-15).

Segmentation is a **hint**, never a requirement. The markers below are the common shapes a problem statement can
take, but no list of markers gates correctness: when nothing matches — a one-line question, an identifier, a code
fragment, a language we have never seen — the whole text is the statement and every view still exists. A failed
segmentation is not an error and never removes the dense+lexical baseline (spec 02 §6b).

Views: **V0** whole text (default) · **V1** statement + I/O spec without examples or notes (cheaper) · **V2** the
mean of the V0 and V1 embeddings, which only the dense channel can build, so it lives in `acis.embed`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from acis.prep.normalize import q1

VIEWS = ("V0", "V1", "V2")

# Heading-ish lines. Deliberately generic: any short label between rules, any markdown heading, any short
# "Something:" line. Nothing here names a benchmark, a dataset or a problem source.
_RULED = re.compile(r"^\s*[-=#*~_]{3,}\s*(?P<label>[^\n]{1,40}?)\s*[-=#*~_]{3,}\s*$")
_MARKDOWN = re.compile(r"^\s*#{1,6}\s+(?P<label>\S[^\n]{0,60})\s*$")
_LABELLED = re.compile(r"^\s*(?P<label>[A-Za-z][\w /-]{0,30}):\s*$")

_INPUT_WORDS = ("input", "stdin", "arguments", "parameters")
_OUTPUT_WORDS = ("output", "stdout", "returns", "return")
_EXAMPLE_WORDS = ("example", "examples", "sample", "samples", "demo")
_NOTE_WORDS = ("note", "notes", "explanation", "clarification")


@dataclass(frozen=True, slots=True)
class Segments:
    """A soft decomposition of a query. Every field defaults to the whole text when nothing was detected."""

    statement: str
    io_spec: str = ""
    examples: str = ""
    note: str = ""
    markers_found: int = 0

    @property
    def segmented(self) -> bool:
        return self.markers_found > 0


def _heading_label(line: str) -> str | None:
    for pattern in (_RULED, _MARKDOWN, _LABELLED):
        m = pattern.match(line)
        if m:
            return m.group("label").strip().lower()
    return None


def _bucket(label: str) -> str | None:
    words = set(re.findall(r"[a-z]+", label))
    if words & set(_EXAMPLE_WORDS):
        return "examples"
    if words & set(_NOTE_WORDS):
        return "note"
    if words & set(_INPUT_WORDS) or words & set(_OUTPUT_WORDS):
        return "io_spec"
    return None


def segment(text: str) -> Segments:
    """Split a query into `{statement, io_spec, examples, note}`. Never raises; never returns nothing."""
    normalised = q1(text)
    if not normalised:
        return Segments(statement="")

    buckets: dict[str, list[str]] = {"statement": [], "io_spec": [], "examples": [], "note": []}
    current = "statement"
    markers = 0
    for line in normalised.split("\n"):
        label = _heading_label(line)
        target = _bucket(label) if label else None
        if target is not None:
            current = target
            markers += 1
            continue
        buckets[current].append(line)

    statement = "\n".join(buckets["statement"]).strip()
    if not statement:  # a query that begins with a heading still has a statement: the whole text
        statement = normalised
    return Segments(
        statement=statement,
        io_spec="\n".join(buckets["io_spec"]).strip(),
        examples="\n".join(buckets["examples"]).strip(),
        note="\n".join(buckets["note"]).strip(),
        markers_found=markers,
    )


def build_view(text: str, view: str = "V0") -> str:
    """Render a view. Unknown views and failed segmentation both fall back to V0 (the whole text)."""
    normalised = q1(text)
    if view != "V1":
        return normalised
    seg = segment(text)
    if not seg.segmented:
        return normalised
    parts = [p for p in (seg.statement, seg.io_spec) if p]
    rendered = "\n\n".join(parts).strip()
    return rendered or normalised


def weak_marker_count(text: str) -> int:
    """A routing signal (spec 10 §4), never a filter: how many heading-ish lines the query has."""
    return segment(text).markers_found


__all__ = ["VIEWS", "Segments", "build_view", "segment", "weak_marker_count"]
