"""The mteb adapter contract, executed against the installed mteb (docs/spec/03 §2, G0.1).

Every row here is a way the official run could silently produce a meaningless number:

* the object is dispatched down the wrong branch, so our pipeline never runs;
* `encode()` is called in Mode A, meaning mteb wrapped us instead of using us;
* fewer than `min(top_k, N)` entries come back, so NDCG is computed over a truncated ranking;
* `ModelMeta` identity never changes, so mteb's persistent cache returns a **previous** run's result;
* `TaskResult.to_dict()` carries a `date` and the sample's bare `json.dump` crashes at the very end.

The toy task is synthetic and offline: it exercises the adapter, not the dataset.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("mteb")

import mteb  # noqa: E402
from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from mteb.abstasks.task_metadata import TaskMetadata  # noqa: E402
from mteb.models.models_protocols import CrossEncoderProtocol, EncoderProtocol, SearchProtocol  # noqa: E402

from acis.core.config import freeze_config  # noqa: E402
from acis.mteb_adapter import MODE_ENV, PrePostPipelineEncoder, _EncoderSurface, write_official_json  # noqa: E402
from acis.mteb_meta import MODEL_NAME, make_model_meta, revision_for  # noqa: E402

N_DOCS = 24
N_QUERIES = 6
TOP_K = 1000  # deliberately larger than the corpus: INV-10 says min(top_k, N)


_VERBS = ("search", "traverse", "reverse", "multiply", "sort", "sieve", "hash", "compress")
_NOUNS = ("matrix", "graph", "string", "interval", "heap", "prime", "suffix", "bitset")
_EXTRA = ("greedy", "recursive", "iterative", "randomised", "parallel", "streaming", "offline", "online")


def _toy_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, int]]]:
    """A corpus in which **the relevant document is never tied with another**.

    That property is what makes a mode-parity test mean something. Documents share topic words, so the ranking is
    not trivial, but each one also carries a token no other document has; the query for `d_i` uses both, so `d_i`
    scores strictly highest. Documents further down may tie freely — our tie-break is the content hash and mteb's
    is its own argsort, and a tie *below* the gold cannot move the gold's rank, so it cannot move NDCG or MRR.
    """
    corpus = []
    for i in range(N_DOCS):
        topic = f"{_VERBS[i % len(_VERBS)]} {_NOUNS[i % len(_NOUNS)]} {_EXTRA[i % len(_EXTRA)]}"
        corpus.append(
            {
                "id": f"d{i}",
                "title": "",
                "text": f"# {topic} marker{i:03d}\ndef solve_{i}(x):\n    return x + {i}\n",
            }
        )
    queries = [
        {"id": f"q{i}", "text": f"{_VERBS[i % len(_VERBS)]} {_NOUNS[i % len(_NOUNS)]} marker{i:03d}"}
        for i in range(N_QUERIES)
    ]
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(N_QUERIES)}
    return corpus, queries, qrels


def test_the_toy_fixture_never_ties_the_relevant_document():
    """Guard for the fixture itself: if this fails, the parity tests below would fail for the wrong reason."""
    import numpy as np

    from acis.embed.hashing import HashingEncoder

    corpus, queries, qrels = _toy_rows()
    encoder = HashingEncoder()
    doc_matrix = encoder.encode([c["text"] for c in corpus])
    query_matrix = encoder.encode([q["text"] for q in queries], is_query=True)
    sims = query_matrix @ doc_matrix.T
    for i, q in enumerate(queries):
        gold_index = [c["id"] for c in corpus].index(next(iter(qrels[q["id"]])))
        gold_score = sims[i][gold_index]
        others = np.delete(sims[i], gold_index)
        assert gold_score > others.max() + 1e-6, q["id"]


def make_toy_task(name: str = "AcisToyRetrieval") -> AbsTaskRetrieval:
    from datasets import Dataset

    corpus, queries, qrels = _toy_rows()

    class ToyTask(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=name,
            description="Synthetic offline retrieval task used to pin the adapter contract.",
            reference="https://example.invalid/acis-toy",
            dataset={"path": "acis/toy", "revision": "toy-revision-0"},
            type="Retrieval",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn", "python-Code"],
            main_score="ndcg_at_10",
            date=("2026-01-01", "2026-01-02"),
            domains=["Programming"],
            task_subtypes=["Code retrieval"],
            license="mit",
            annotations_creators="derived",
            dialect=[],
            sample_creation="created",
            bibtex_citation="",
        )

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            if self.data_loaded:
                return
            self.dataset = {
                "default": {
                    "test": {
                        "corpus": Dataset.from_list(corpus),
                        "queries": Dataset.from_list(queries),
                        "relevant_docs": qrels,
                        "top_ranked": None,
                    }
                }
            }
            self.data_loaded = True

    return ToyTask()


@pytest.fixture
def mode_a(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "A")
    monkeypatch.setenv("ACIS_CONFIG", "configs/dev.yaml")
    return PrePostPipelineEncoder()


@pytest.fixture
def mode_b(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "B")
    monkeypatch.setenv("ACIS_CONFIG", "configs/dev.yaml")
    return PrePostPipelineEncoder()


# -- dispatch ------------------------------------------------------------------------------------------------------
def test_mode_a_is_dispatched_directly(mode_a):
    assert isinstance(mode_a, SearchProtocol)
    assert not isinstance(mode_a, CrossEncoderProtocol)


def test_mode_b_takes_the_encoder_wrapper_path(mode_b):
    assert isinstance(mode_b, EncoderProtocol)
    assert not isinstance(mode_b, SearchProtocol)
    assert not hasattr(mode_b, "index") and not hasattr(mode_b, "search")
    assert type(mode_b) is _EncoderSurface


def test_the_adapter_never_defines_predict():
    """A `predict()` would route us to the cross-encoder branch and our `search()` would never run (V-02)."""
    assert not hasattr(PrePostPipelineEncoder, "predict")
    assert not hasattr(_EncoderSurface, "predict")


def test_index_and_search_signatures_are_keyword_only_and_future_proof():
    import inspect

    for name in ("index", "search"):
        params = inspect.signature(getattr(PrePostPipelineEncoder, name)).parameters
        positional = [p for p in params.values() if p.kind is p.POSITIONAL_OR_KEYWORD and p.name != "self"]
        assert len(positional) == 1, f"{name} must take exactly one positional argument"
        assert params["num_proc"].default is None
        assert any(p.kind is p.VAR_KEYWORD for p in params.values()), f"{name} must absorb later mteb arguments"


# -- end-to-end through mteb ---------------------------------------------------------------------------------------
def _evaluate(model: Any, task: AbsTaskRetrieval, prediction_folder=None) -> Any:
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 8},
        cache=None,
        overwrite_strategy="always",
        prediction_folder=prediction_folder,
        show_progress_bar=False,
    )
    return list(result.task_results)[0] if hasattr(result, "task_results") else result


def test_mode_a_runs_our_pipeline_and_never_calls_encode(mode_a, tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_RUN_DIR", str(tmp_path))
    task_result = _evaluate(mode_a, make_toy_task())
    assert mode_a.adapter_invocations == 1, "our search() must run exactly once"
    assert mode_a.encode_calls == 0, "Mode A must not go through mteb's encoder wrapper"
    scores = task_result.scores["test"][0]
    assert 0.0 <= scores["ndcg_at_10"] <= 1.0


def test_mode_b_runs_through_the_encoder_wrapper(mode_b, tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_RUN_DIR", str(tmp_path))
    task_result = _evaluate(mode_b, make_toy_task())
    assert mode_b.encode_calls > 0, "Mode B must be graded on honest embeddings"
    assert mode_b.adapter_invocations == 0
    assert 0.0 <= task_result.scores["test"][0]["ndcg_at_10"] <= 1.0


def test_both_modes_agree_for_the_same_encoder(mode_a, mode_b, tmp_path, monkeypatch):
    """Parity P1: our SearchProtocol path and mteb's own wrapper must rank the same documents the same way."""
    monkeypatch.setenv("ACIS_RUN_DIR", str(tmp_path))
    a = _evaluate(mode_a, make_toy_task("AcisToyA")).scores["test"][0]
    b = _evaluate(mode_b, make_toy_task("AcisToyB")).scores["test"][0]
    assert a["ndcg_at_10"] == pytest.approx(b["ndcg_at_10"], abs=1e-9)
    assert a["mrr_at_10"] == pytest.approx(b["mrr_at_10"], abs=1e-9)


def test_search_returns_min_top_k_and_n_entries(mode_a):
    """INV-10, and the reason spec 09 §4 corrected 'exactly 1,000' to `min(top_k, N)`."""
    corpus, queries, _ = _toy_rows()
    task = make_toy_task()
    mode_a.index(_as_dataset(corpus), task_metadata=task.metadata, hf_split="test", hf_subset="default")
    out = mode_a.search(_as_dataset(queries), task_metadata=task.metadata, hf_split="test", top_k=TOP_K)
    for qid, scores in out.items():
        values = list(scores.values())
        assert len(scores) == min(TOP_K, N_DOCS), qid
        assert all(a > b for a, b in zip(values, values[1:], strict=False))
        assert all(v == v and abs(v) != float("inf") for v in values)
        assert set(scores) <= {row["id"] for row in corpus}


def test_search_respects_top_ranked_restriction(mode_a):
    corpus, queries, _ = _toy_rows()
    task = make_toy_task()
    mode_a.index(_as_dataset(corpus), task_metadata=task.metadata, hf_split="test", hf_subset="default")
    allowed = {"q0": ["d0", "d1", "d2"]}
    out = mode_a.search(
        _as_dataset([queries[0]]), task_metadata=task.metadata, hf_split="test", top_k=3, top_ranked=allowed
    )
    assert set(out["q0"]) <= set(allowed["q0"])


def test_search_before_index_raises(mode_a):
    from acis.core.errors import IndexRequired

    _, queries, _ = _toy_rows()
    with pytest.raises(IndexRequired):
        mode_a.search(_as_dataset(queries), top_k=10)


def _as_dataset(rows: list[dict[str, str]]):
    from datasets import Dataset

    return Dataset.from_list(rows)


# -- run manifest ----------------------------------------------------------------------------------------------------
def test_search_writes_a_run_manifest_proving_our_code_ran(mode_a, tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_RUN_DIR", str(tmp_path))
    _evaluate(mode_a, make_toy_task())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_invocations"] == 1
    assert manifest["mode"] == "A"
    assert manifest["agent_calls"] == 0  # INV-13
    assert manifest["fallbacks"] == 0
    assert manifest["config_hash"] and manifest["model_revision"]
    assert manifest["snapshot_id"]


# -- ModelMeta identity and the stale-cache hazard ----------------------------------------------------------------------
def test_model_meta_identity(mode_a):
    meta = mode_a.mteb_model_meta
    assert meta.name == MODEL_NAME
    assert meta.revision == revision_for(mode_a.cfg)
    git12, _, config12 = meta.revision.partition("+")
    assert len(git12) == 12 and len(config12) == 12


def test_revision_moves_when_a_ranking_relevant_key_moves():
    base = freeze_config({"prep": {"query": {"max_tokens": 1024}}})
    changed = base.with_overrides(**{"prep.query.max_tokens": 512})
    operational = base.with_overrides(**{"run.threads": 99})
    assert revision_for(changed) != revision_for(base)
    assert revision_for(operational) == revision_for(base)
    assert make_model_meta(changed).revision != make_model_meta(base).revision


def test_stale_cache_hazard_is_reproduced_then_defeated(monkeypatch, tmp_path):
    """V-04: mteb's default cache + `only-missing` can hand back a previous run's numbers.

    First reproduce the hazard with a fake cache that answers for a known model identity, then show the two
    defences the official script uses: a fresh identity, and `cache=None` + `overwrite_strategy="always"`.
    """
    from mteb.models.model_meta import ModelMeta

    seen: list[str] = []

    class StaleCache:
        """Stands in for mteb's persistent ResultCache: keyed on (name, revision), ignorant of the code."""

        def __init__(self) -> None:
            self.store: dict[tuple[str, str], float] = {}

        def get(self, meta: ModelMeta) -> float | None:
            seen.append(meta.revision)
            return self.store.get((meta.name, meta.revision))

        def put(self, meta: ModelMeta, score: float) -> None:
            self.store[(meta.name, meta.revision)] = score

    cache = StaleCache()
    cfg = freeze_config({"prep": {"query": {"max_tokens": 1024}}})
    meta = make_model_meta(cfg)
    cache.put(meta, 0.99)  # a previous run's number

    # hazard: an unchanged identity returns the stale score even though the code changed
    assert cache.get(make_model_meta(cfg)) == 0.99

    # defence 1: the revision carries the config hash, so a ranking-relevant change is a new identity
    changed = cfg.with_overrides(**{"prep.query.max_tokens": 512})
    assert cache.get(make_model_meta(changed)) is None

    # defence 2: the official script passes cache=None and overwrite_strategy="always"
    import inspect

    from acis.eval import official

    source = inspect.getsource(official.run_mode)
    assert "cache=None" in source and 'overwrite_strategy="always"' in source


# -- the JSON writer ----------------------------------------------------------------------------------------------------
def test_official_writer_survives_a_date_that_plain_json_dump_cannot(tmp_path):
    """V-07: `TaskResult.to_dict()` carries a `date`; the PDF sample's bare `json.dump` raises on it."""
    from datetime import datetime

    import numpy as np

    class FakeResult:
        def to_dict(self) -> dict[str, Any]:
            return {
                "task_name": "AppsRetrieval",
                "date": datetime(2026, 9, 21, 12, 0, 0),
                "scores": {"test": [{"ndcg_at_10": np.float32(0.75)}]},
                "path": tmp_path,
            }

    with pytest.raises(TypeError):
        json.dumps(FakeResult().to_dict())

    path = write_official_json(FakeResult(), tmp_path / "result.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-09-21T12:00:00"
    assert payload["scores"]["test"][0]["ndcg_at_10"] == pytest.approx(0.75)
    assert isinstance(payload["path"], str)


def test_official_writer_refuses_unknown_objects(tmp_path):
    class Weird:
        def to_dict(self) -> dict[str, Any]:
            return {"x": object()}

    with pytest.raises(TypeError, match="not JSON-serialisable"):
        write_official_json(Weird(), tmp_path / "bad.json")
