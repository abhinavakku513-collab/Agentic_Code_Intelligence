"""Harness-validation parity tests P1–P7 (docs/spec/06 §5, blueprint §6.6).

These are the tests that make every later number believable:

| id | claim |
|---|---|
| P1 | our `SearchProtocol` path and mteb's own `SearchEncoderWrapper` rank identically for the same encoder |
| P2 | our BM25 equals `mteb/baseline-bm25s` rank-for-rank with matched tokenisation |
| P3 | metrics re-scored from `run.trec` equal the metrics computed directly, to 1e-9 |
| P4 | a query ranks identically alone and inside a batch |
| P5 | corpus order and document relabelling change nothing |
| P6 | a warm run (cached query vectors) ranks identically to a cold one |
| P7 | adversarial ties and sub-float32 gaps still give NDCG@10 = MRR@10 = 1.0 for a gold-at-rank-1 fixture |

B0 (metric parity against mteb/pytrec_eval on random, oracle and BM25 runs) is the first block below.
"""

from __future__ import annotations

import random

import pytest

from acis.core.config import freeze_config
from acis.core.types import Snippet
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG
from acis.eval import runfile
from acis.eval.metrics import K_VALUES, score_run, score_run_reference
from acis.lexical.bm25 import Bm25Index
from acis.lexical.tokenize import corpus_text
from acis.rank.compose import rank_derived_scores

TOLERANCE = 1e-9


def _fixture_corpus(n: int = 60) -> list[tuple[str, str]]:
    """A deterministic pseudo-corpus: varied enough to move a ranking, small enough to be instant."""
    topics = [
        "graph bfs queue",
        "dynamic programming knapsack",
        "string palindrome reverse",
        "modular arithmetic power",
        "binary search sorted array",
        "heap priority dijkstra",
        "matrix multiplication",
        "prime sieve",
    ]
    rng = random.Random(11)
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        filler = " ".join(rng.sample(["alpha", "beta", "gamma", "delta", "sigma", "omega"], 4))
        docs.append((f"d{i}", f"# {topic}\ndef solve_{i}(n):\n    # {filler}\n    return {i}\n"))
    return docs


def _qrels_and_queries(docs: list[tuple[str, str]], n_queries: int = 20):
    qrels = {f"q{i}": {docs[i][0]: 1} for i in range(n_queries)}
    queries = {f"q{i}": docs[i][1].splitlines()[0].lstrip("# ") for i in range(n_queries)}
    return qrels, queries


# -- B0: metric parity against mteb / pytrec_eval ------------------------------------------------------------------
def _random_run(qrels, doc_ids, seed=0, top_k=50):
    rng = random.Random(seed)
    return {q: rank_derived_scores(rng.sample(doc_ids, min(top_k, len(doc_ids))), top_k) for q in qrels}


def _oracle_run(qrels, doc_ids, top_k=50):
    run = {}
    for q, rels in qrels.items():
        gold = [d for d, r in rels.items() if r > 0]
        rest = [d for d in doc_ids if d not in set(gold)]
        run[q] = rank_derived_scores((gold + rest)[:top_k], top_k)
    return run


def _bm25_run(docs, queries, top_k=50):
    index = Bm25Index.build([i for i, _ in docs], [corpus_text("", t) for _, t in docs])
    ids = list(queries)
    rows = index.retrieve([queries[q] for q in ids], top_k)
    return {q: dict(row) for q, row in zip(ids, rows, strict=True)}


@pytest.mark.parametrize("kind", ["random", "oracle", "bm25"])
def test_b0_metrics_equal_mteb_and_pytrec_eval_to_1e9(kind):
    docs = _fixture_corpus()
    doc_ids = [i for i, _ in docs]
    qrels, queries = _qrels_and_queries(docs)
    run = {
        "random": lambda: _random_run(qrels, doc_ids),
        "oracle": lambda: _oracle_run(qrels, doc_ids),
        "bm25": lambda: _bm25_run(docs, queries),
    }[kind]()

    ours = score_run(qrels, run, K_VALUES)
    theirs = score_run_reference(qrels, run, K_VALUES)
    assert theirs, "the reference implementation returned nothing"
    for key, value in theirs.items():
        assert ours[key] == pytest.approx(value, abs=TOLERANCE), f"{kind}: {key}"


def test_b0_oracle_scores_exactly_one():
    docs = _fixture_corpus()
    qrels, _ = _qrels_and_queries(docs)
    scores = score_run(qrels, _oracle_run(qrels, [i for i, _ in docs]))
    assert scores["ndcg_at_10"] == pytest.approx(1.0, abs=TOLERANCE)
    assert scores["mrr_at_10"] == pytest.approx(1.0, abs=TOLERANCE)


# -- P3: re-scoring a run file -------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["random", "oracle", "bm25"])
def test_p3_rescoring_run_trec_reproduces_the_metrics(kind, tmp_path):
    docs = _fixture_corpus()
    qrels, queries = _qrels_and_queries(docs)
    run = {
        "random": lambda: _random_run(qrels, [i for i, _ in docs]),
        "oracle": lambda: _oracle_run(qrels, [i for i, _ in docs]),
        "bm25": lambda: _bm25_run(docs, queries),
    }[kind]()
    direct = score_run(qrels, run, K_VALUES)
    path = runfile.write_trec(run, tmp_path / "run.trec")
    rescored = runfile.rescore_trec(path, qrels, K_VALUES)
    for key, value in direct.items():
        assert rescored[key] == pytest.approx(value, abs=TOLERANCE), key


# -- P4/P5/P6 on the real engine ------------------------------------------------------------------------------------
def _engine(docs):
    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snap = engine.build_snapshot([Snippet(handle=i, text=t) for i, t in docs], source="parity")
    return engine, snap


def test_p4_query_alone_equals_query_in_a_batch():
    docs = _fixture_corpus()
    _, queries = _qrels_and_queries(docs)
    engine, snap = _engine(docs)
    ids = list(queries)
    batched = engine.search_batch(snap, ids, [queries[q] for q in ids], top_k=10)
    for q in ids:
        assert engine.search_batch(snap, [q], [queries[q]], top_k=10)[q] == batched[q]


def test_p5_corpus_order_does_not_change_the_ranking():
    docs = _fixture_corpus()
    _, queries = _qrels_and_queries(docs)
    engine_a, snap_a = _engine(docs)
    engine_b, snap_b = _engine(list(reversed(docs)))
    for q, text in queries.items():
        a = [d for d, _ in engine_a.search_batch(snap_a, [q], [text], top_k=10)[q]]
        b = [d for d, _ in engine_b.search_batch(snap_b, [q], [text], top_k=10)[q]]
        assert a == b, q


def test_p5_relabelling_documents_does_not_change_the_ranking():
    docs = _fixture_corpus()
    _, queries = _qrels_and_queries(docs)
    relabelled = [(f"renamed-{i}", t) for i, t in docs]
    engine_a, snap_a = _engine(docs)
    engine_b, snap_b = _engine(relabelled)
    for q, text in queries.items():
        a = [d for d, _ in engine_a.search_batch(snap_a, [q], [text], top_k=10)[q]]
        b = [d.replace("renamed-", "") for d, _ in engine_b.search_batch(snap_b, [q], [text], top_k=10)[q]]
        assert a == b, q


def test_p6_warm_cache_ranks_identically_to_cold():
    docs = _fixture_corpus()
    _, queries = _qrels_and_queries(docs)
    engine, snap = _engine(docs)
    ids = list(queries)
    distinct_texts = len({queries[q] for q in ids})
    cold = engine.search_batch(snap, ids, [queries[q] for q in ids], top_k=10)
    assert engine.counters.get("cache.qemb.miss") == distinct_texts  # the cache is keyed on content, not on query id
    hits_after_cold = engine.counters.get("cache.qemb.hit")
    warm = engine.search_batch(snap, ids, [queries[q] for q in ids], top_k=10)
    assert engine.counters.get("cache.qemb.hit") == hits_after_cold + len(ids)
    assert engine.counters.get("cache.qemb.miss") == distinct_texts  # a warm pass encodes nothing new
    assert cold == warm


# -- P7: adversarial ties -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gap,base", [(0.0, 1.0), (1e-9, 1.0), (1e-8, 1.0), (1e-6, 10.0), (1e-5, 100.0)])
def test_p7_rank_derived_scores_keep_ndcg_and_mrr_in_agreement(gap, base):
    """A raw-score ranking can disagree with itself in the collapse zone; a rank-derived one cannot (V-08)."""
    gold = "d3"
    ids = [gold] + [f"x{i:04d}" for i in range(29)]
    qrels = {"q": {gold: 1}}

    raw = {"q": {d: base - i * gap for i, d in enumerate(ids)}}
    raw_scores = score_run(qrels, raw, (10,))
    assert score_run_reference(qrels, raw, (10,))["ndcg_at_10"] == pytest.approx(
        raw_scores["ndcg_at_10"], abs=TOLERANCE
    )

    composed = {"q": rank_derived_scores(ids, 1000)}
    scores = score_run(qrels, composed, (10,))
    assert scores["ndcg_at_10"] == pytest.approx(1.0, abs=TOLERANCE)
    assert scores["mrr_at_10"] == pytest.approx(1.0, abs=TOLERANCE)
    assert scores["ndcg_at_10"] == pytest.approx(scores["mrr_at_10"], abs=TOLERANCE)


def test_p7_exact_duplicate_documents_get_distinct_adjacent_ranks():
    docs = _fixture_corpus(20) + [("dup_a", "identical body\n"), ("dup_b", "identical body\n")]
    engine, snap = _engine(docs)
    ranked = engine.search_batch(snap, ["q"], ["identical body"], top_k=5)["q"]
    positions = {d: i for i, (d, _) in enumerate(ranked)}
    assert abs(positions["dup_a"] - positions["dup_b"]) == 1
    scores = rank_derived_scores(ranked, 1000)
    assert len(set(scores.values())) == len(scores)


# -- P2: BM25 parity against the mteb baseline (needs the dataset) ---------------------------------------------------
@pytest.mark.dataset
@pytest.mark.slow
def test_p2_bm25_parity_with_the_mteb_baseline():
    """B1: ≥ 95 % of queries must have an identical top-10; matched tokenisation should give 100 %."""
    from acis.appsdata import apps
    from acis.eval import ladder

    ids = list(apps.dev_query_ids())[:150]
    ours = ladder.run_bm25_acis(ids, top_k=100)
    theirs = ladder.run_bm25_mteb(ids, top_k=100)
    agreement = ladder.top_k_agreement(ours, theirs, 10)
    assert agreement >= 0.95, f"top-10 identical for only {agreement:.1%} of queries"
    assert score_run(apps.load_qrels(), ours)["ndcg_at_10"] == pytest.approx(
        score_run(apps.load_qrels(), theirs)["ndcg_at_10"], abs=1e-12
    )
