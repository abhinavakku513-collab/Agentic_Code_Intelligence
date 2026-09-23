"""Pooling a transformer's hidden states into one vector (docs/spec/02 §3).

This is short, and it is the easiest place in the dense engine to be quietly wrong. Every strategy here is a
different answer to "which positions count", and picking the wrong one — or ignoring which side the batch was
padded on — produces vectors that are not garbage, just subtly worse. Nothing crashes; the model simply scores
lower than it should and the bake-off blames the model.

Two rules that are easy to get wrong and are therefore tested directly against real tensors:

* **Padding never contributes.** A mean over the padded positions makes a short document's vector depend on how
  long the longest document in its batch happened to be — which would also break batch invariance (INV-3).
* **"Last token" means the last *real* token.** With left padding that is index −1 for every row; with right
  padding it is `mask.sum() − 1`, and using −1 there pools a pad.
"""

from __future__ import annotations

from typing import Any

STRATEGIES = ("last_token", "mean", "cls", "max")


def pool(hidden_states: Any, attention_mask: Any, *, strategy: str, padding_side: str = "right") -> Any:
    """Reduce `(batch, seq, dim)` hidden states to `(batch, dim)`, honouring the mask and the padding side."""
    import torch  # noqa: PLC0415

    if strategy not in STRATEGIES:
        raise ValueError(f"unknown pooling strategy {strategy!r}; known: {STRATEGIES}")
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', not {padding_side!r}")

    mask = attention_mask.to(dtype=hidden_states.dtype)
    if strategy == "last_token":
        if padding_side == "left":
            return hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1).long() - 1
        lengths = lengths.clamp(min=0)
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), lengths]

    if strategy == "cls":
        if padding_side == "right":
            return hidden_states[:, 0]
        # Left padding puts the real first token after the pads; find it rather than pooling a pad.
        first = attention_mask.long().argmax(dim=1)
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), first]

    expanded = mask.unsqueeze(-1)
    if strategy == "mean":
        summed = (hidden_states * expanded).sum(dim=1)
        counts = expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    # max: masked positions must not win the maximum
    neutral = torch.finfo(hidden_states.dtype).min
    masked = hidden_states.masked_fill(expanded == 0, neutral)
    return masked.max(dim=1).values


__all__ = ["STRATEGIES", "pool"]
