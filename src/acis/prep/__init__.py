"""`acis.prep` — deterministic query/document normalisation, segmentation, views and truncation."""

from __future__ import annotations

from acis.prep.normalize import d1, fold_numeric_literals, is_empty_query, lexical_view, q1
from acis.prep.truncate import Truncation, head_tail, truncate_text
from acis.prep.views import Segments, build_view, segment, weak_marker_count

__all__ = [
    "Segments",
    "Truncation",
    "build_view",
    "d1",
    "fold_numeric_literals",
    "head_tail",
    "is_empty_query",
    "lexical_view",
    "q1",
    "segment",
    "truncate_text",
    "weak_marker_count",
]
