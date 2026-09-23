"""Deterministic text normalisation (docs/spec/02 §2).

`q1` (queries) and `d1` (documents) are pure functions of their input: same text in, same text out, on any machine.
They are the reason format noise is a *strict* no-op for the whole engine (INV-15, `tests/robustness`): NFKC folds
non-breaking spaces, CRLF becomes LF, trailing spaces disappear and blank-line runs collapse — none of which a user
meant to change.

Nothing else is done to a query for the dense view: raw problem statements are what encoders were trained on.
"""

from __future__ import annotations

import re
import unicodedata

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
BLANK_RUN = re.compile(r"\n{3,}")
TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
BOM = "\ufeff"

QUERY_VERSION = "q1"
DOC_VERSION = "d1"


def q1(text: str) -> str:
    """Query normalisation: NFKC, `\\r\\n -> \\n`, strip trailing spaces, cap blank-line runs at one, drop controls."""
    if not isinstance(text, str):
        raise TypeError("query text must be a string")
    out = unicodedata.normalize("NFKC", text.replace(BOM, ""))
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = CONTROL.sub("", out)
    out = TRAILING_WS.sub("", out)
    out = BLANK_RUN.sub("\n\n", out)
    return out.strip()


def d1(text: str) -> str:
    """Document normalisation: strip BOM, `\\r`, trailing whitespace. **Tabs are kept** — they are Python syntax."""
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    out = text.replace(BOM, "").replace("\r\n", "\n").replace("\r", "\n")
    out = CONTROL.sub("", out)
    out = TRAILING_WS.sub("", out)
    return out.strip("\n")


_LATEX_MACRO = re.compile(r"\\[a-zA-Z]+\s*")
_MATH_DELIM = re.compile(r"[$]+")
# **Bounded** digit runs, and possessive quantifiers, and both are load-bearing.
#
# The readable version — `(\d+)\s*\^` — is quadratic on a long run of digits with no `^` after it: every start
# position rescans the whole run before failing. A 40,000-character document of digits took 34 seconds, and this
# function runs on **every query**, so that is a denial of service rather than a slow test. Possessive
# quantifiers stop the engine giving characters back *within* an attempt; bounding each run to `_MAX_DIGITS`
# stops the attempts themselves from being long. The bound costs nothing real: `_MAX_EXPONENT` is already 18, and
# a base of more than 18 digits raised to a power is not a constant any statement is talking about.
_MAX_DIGITS = 18
_OFFSET = rf"(?:\s*+([+-])\s*+(\d{{1,{_MAX_DIGITS}}}+))?"
_POWER = re.compile(rf"(\d{{1,{_MAX_DIGITS}}}+)\s*+\^\s*+\{{?\s*+(\d{{1,{_MAX_DIGITS}}}+)\s*+\}}?" + _OFFSET)
_SCIENTIFIC = re.compile(rf"\b(\d{{1,{_MAX_DIGITS}}}+)[eE](\d{{1,{_MAX_DIGITS}}}+)\b" + _OFFSET)
_MAX_EXPONENT = 18
#: Folding is linear, but a pathological input is still bounded: beyond this, the text is folded in one pass and
#: the remainder is left as it is rather than scanned indefinitely.
MAX_FOLD_CHARS = 200_000


def _apply_offset(value: int, sign: str | None, offset: str | None) -> str:
    if sign is None or offset is None:
        return str(value)
    return str(value + int(offset) if sign == "+" else value - int(offset))


def fold_numeric_literals(text: str) -> str:
    """Evaluate numeric literal expressions so one constant written three ways matches itself.

    `10^9+7`, `10^{9}+7` and `1e9+7` all become `1000000007` (docs/spec/02 §2). This is a *generic* arithmetic
    fold over any base, exponent and offset — INV-15 forbids a closed list of interesting moduli, and this
    function contains none. Absurd exponents are left untouched rather than materialised.
    """

    def _pow(m: re.Match[str]) -> str:
        base, exp = int(m.group(1)), int(m.group(2))
        if exp > _MAX_EXPONENT:
            return m.group(0)
        return _apply_offset(base**exp, m.group(3), m.group(4))

    def _sci(m: re.Match[str]) -> str:
        mantissa, exp = int(m.group(1)), int(m.group(2))
        if exp > _MAX_EXPONENT:
            return m.group(0)
        return _apply_offset(mantissa * 10**exp, m.group(3), m.group(4))

    if len(text) > MAX_FOLD_CHARS:
        head, tail = text[:MAX_FOLD_CHARS], text[MAX_FOLD_CHARS:]
        return _SCIENTIFIC.sub(_sci, _POWER.sub(_pow, head)) + tail
    return _SCIENTIFIC.sub(_sci, _POWER.sub(_pow, text))


def lexical_view(text: str) -> str:
    """Prose for the lexical channel: lowercase, LaTeX macros dropped, numeric literals folded, numbers kept."""
    out = fold_numeric_literals(q1(text))
    out = _LATEX_MACRO.sub(" ", out)
    out = _MATH_DELIM.sub(" ", out)
    return " ".join(out.lower().split())


def is_empty_query(text: str) -> bool:
    return not q1(text).strip()


__all__ = [
    "DOC_VERSION",
    "QUERY_VERSION",
    "d1",
    "fold_numeric_literals",
    "is_empty_query",
    "lexical_view",
    "q1",
]
