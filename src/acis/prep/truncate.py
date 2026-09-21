"""Head + tail truncation (docs/spec/02 §2, §5, D3).

Dense text is `head 768 + tail 256` tokens: the beginning of a problem statement carries the task, the end carries
the constraints and the examples, and the middle is the cheapest thing to lose. BM25 and the feature extractors
always see the full text, so truncation costs the dense channel only.

The token counter is injected. Phase 1 uses a whitespace counter (no model is loaded in the harness phase); Phase 2
passes the encoder's real tokenizer, and the `is_truncated` flag it returns becomes an LTR feature.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 1024
DEFAULT_HEAD = 768
DEFAULT_TAIL = 256

Tokenizer = Callable[[str], Sequence[str]]


def whitespace_tokens(text: str) -> list[str]:
    """Model-independent fallback counter. Deliberately crude — it only has to be deterministic."""
    return text.split()


@dataclass(frozen=True, slots=True)
class Truncation:
    text: str
    n_tokens: int
    truncated: bool
    head: int
    tail: int


def head_tail(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
    tokenizer: Tokenizer = whitespace_tokens,
    joiner: str = " ",
) -> Truncation:
    """Keep the first `head` and last `tail` tokens when a text exceeds `max_tokens`."""
    if head < 0 or tail < 0 or max_tokens <= 0:
        raise ValueError("max_tokens must be positive and head/tail non-negative")
    tokens = list(tokenizer(text))
    if len(tokens) <= max_tokens:
        return Truncation(text=text, n_tokens=len(tokens), truncated=False, head=head, tail=tail)
    budget_head = min(head, max_tokens)
    budget_tail = min(tail, max(0, max_tokens - budget_head))
    kept = tokens[:budget_head] + (tokens[-budget_tail:] if budget_tail else [])
    return Truncation(text=joiner.join(kept), n_tokens=len(tokens), truncated=True, head=budget_head, tail=budget_tail)


def truncate_text(text: str, **kwargs: object) -> str:
    return head_tail(text, **kwargs).text  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_HEAD",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TAIL",
    "Tokenizer",
    "Truncation",
    "head_tail",
    "truncate_text",
    "whitespace_tokens",
]
