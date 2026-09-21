"""G0.1 seed: how the INSTALLED mteb picks the search model for a retrieval task (docs/spec/01 V-02, docs/spec/03 s2).

Run it on every cell of the compatibility matrix:  uv run --with mteb==<v> pytest tests/contract -q
The source-anchor test is a drift detector: if mteb rewrites its dispatch, it fails before a release candidate does.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mteb")
import numpy as np  # noqa: E402
from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.models_protocols import CrossEncoderProtocol, EncoderProtocol, SearchProtocol  # noqa: E402


def dispatch(model) -> str:
    """The rule from mteb/abstasks/retrieval.py (identical in mteb 2.0.5, 2.12.30, 2.21.0)."""
    if isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol):
        return "encoder-wrapper"
    if isinstance(model, CrossEncoderProtocol):
        return "cross-encoder-wrapper"
    if isinstance(model, SearchProtocol):
        return "direct"
    return "rejected"


class _Base(AbsEncoder):
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kw):
        return np.zeros((1, 4), dtype=np.float32)


def _index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None, encode_kwargs=None, num_proc=None, **_):
    return None


def _search(
    self,
    queries,
    *,
    task_metadata=None,
    hf_split=None,
    hf_subset=None,
    top_k=1000,
    encode_kwargs=None,
    top_ranked=None,
    num_proc=None,
    **_,
):
    return {}


def _predict(self, inputs1, inputs2, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kw):
    return np.zeros(1)


def _mk(**methods):
    return type("M", (_Base,), methods)()


CASES = [
    ("encoder only", {}, "encoder-wrapper"),
    ("index + search (Mode A)", {"index": _index, "search": _search}, "direct"),
    (
        "index + search + predict  <- NEVER",
        {"index": _index, "search": _search, "predict": _predict},
        "cross-encoder-wrapper",
    ),
    ("index only", {"index": _index}, "encoder-wrapper"),  # silently ignored
    ("search only", {"search": _search}, "encoder-wrapper"),  # silently ignored
]


@pytest.mark.parametrize("name,methods,expected", CASES, ids=[c[0] for c in CASES])
def test_dispatch(name, methods, expected):
    assert dispatch(_mk(**methods)) == expected


def test_dispatch_source_anchor():
    src = inspect.getsource(AbsTaskRetrieval._evaluate_subset)
    assert "isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol)" in src
    assert "isinstance(model, CrossEncoderProtocol)" in src
    assert "isinstance(model, SearchProtocol)" in src
