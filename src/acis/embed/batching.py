"""Token-budget batching (docs/spec/02 §3, §7; docs/spec/06 §6).

Encoding dominates the cold pass — 8,765 documents plus 3,765 queries, roughly 3M tokens — so this is the first
place profiling points at, and the first optimisation the spec sanctions.

Two ideas, both of which have to hold exactly:

* **Batch by token budget, not by row count.** A fixed batch size is sized for the worst row in the corpus: APPS
  documents run from 5 characters to 289k, so a 64-row batch is either tiny for short rows or an out-of-memory
  risk for long ones. Batching to ~16k tokens keeps the work per batch roughly constant.
* **Sort by length first.** Padding is wasted compute, and a batch of similar lengths barely pads at all. The
  original order is restored afterwards, because **the caller's order is part of the contract**: `encode()`
  returns row *i* for input *i*, and a reordering bug here would silently mislabel every vector.

Both are pure functions of the lengths, so they are tested without a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_TOKEN_BUDGET = 16_384
DEFAULT_MAX_ROWS = 256


@dataclass(frozen=True, slots=True)
class Batch:
    """One unit of work: the original indices, so results can be scattered back where they belong."""

    indices: tuple[int, ...]
    max_length: int
    total_tokens: int

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def padded_tokens(self) -> int:
        """What the model actually computes: every row padded to the longest in the batch."""
        return self.size * self.max_length


def plan_batches(
    lengths: Sequence[int],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_rows: int = DEFAULT_MAX_ROWS,
    sort_by_length: bool = True,
) -> list[Batch]:
    """Group row indices into batches whose *padded* size stays within `token_budget`.

    A single row longer than the budget gets a batch of its own rather than being dropped or truncated here:
    truncation is `acis.prep`'s decision and has already happened by this point.
    """
    if token_budget <= 0 or max_rows <= 0:
        raise ValueError("token_budget and max_rows must be positive")
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i)) if sort_by_length else list(range(len(lengths)))

    batches: list[Batch] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        length = max(1, int(lengths[index]))
        candidate_max = max(current_max, length)
        if current and (len(current) + 1 > max_rows or (len(current) + 1) * candidate_max > token_budget):
            batches.append(Batch(tuple(current), current_max, sum(lengths[i] for i in current)))
            current, current_max = [], 0
            candidate_max = length
        current.append(index)
        current_max = candidate_max
    if current:
        batches.append(Batch(tuple(current), current_max, sum(lengths[i] for i in current)))
    return batches


def restore_order(results: Sequence[tuple[int, object]], n: int) -> list[object]:
    """Scatter `(original_index, value)` pairs back into input order, refusing gaps and duplicates."""
    out: list[object | None] = [None] * n
    for index, value in results:
        if not 0 <= index < n:
            raise IndexError(f"result index {index} outside 0..{n - 1}")
        if out[index] is not None:
            raise ValueError(f"two results for input {index}")
        out[index] = value
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise ValueError(f"no result for inputs {missing[:5]}{'…' if len(missing) > 5 else ''}")
    return [v for v in out if v is not None]


def padding_waste(batches: Sequence[Batch]) -> float:
    """Fraction of computed tokens that are padding — the number length-sorting exists to reduce."""
    padded = sum(b.padded_tokens for b in batches)
    real = sum(b.total_tokens for b in batches)
    return 0.0 if padded == 0 else 1.0 - (real / padded)


def plan_for_texts(
    texts: Sequence[str],
    count_tokens: Callable[[str], int],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[Batch]:
    return plan_batches(
        [count_tokens(t) for t in texts], token_budget=token_budget, max_rows=max_rows, sort_by_length=True
    )


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TOKEN_BUDGET",
    "Batch",
    "padding_waste",
    "plan_batches",
    "plan_for_texts",
    "restore_order",
]
