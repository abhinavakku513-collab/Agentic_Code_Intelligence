"""Phase 3 export: the decontaminated training bundle for the GPU hand-off (docs/spec/02 §6, docs/GPU_HANDOFF.md).

What must hold, whatever the GPU side does with it: no held-out id or text ever enters the bundle; a query dropped by
decontamination is gone everywhere, its document included; a negative is never the positive, never a near-copy of
it, never the solution of a sibling problem; negatives come only from documents that are positives of dev queries
(an unlabelled document is never shown to training), and each carries its fold so a fold model can refuse the
held-out fold's documents.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from acis.core.errors import SealedDataAccess
from acis.eval import adapt_export as ax


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def toy():
    """Four problems, four folds; d2 is a near-copy of d1, and q3's statement is q0's with one word changed."""
    queries = {
        "q0": "given an array of n integers print the maximum sum of a contiguous subarray modulo 10^9+7",
        "q1": "count the number of palindromic substrings of the given string s and print it",
        "q2": "find the shortest path between two nodes of a weighted graph using dijkstra",
        "q3": "given an array of n integers print the maximum sum of a contiguous subarray modulo 998244353",
    }
    positive = {"q0": "d0", "q1": "d1", "q2": "d2", "q3": "d3"}
    docs = {
        "d0": "def kadane(a):\n    best = cur = 0\n    for x in a:\n        cur = max(x, cur + x)\n    return best\n",
        "d1": "def pal(s):\n    return sum(s[i:j] == s[i:j][::-1] for i in range(len(s)) for j in range(i, len(s)))\n",
        "d2": "import heapq\ndef dijkstra(g, s):\n    pq = [(0, s)]\n    return {}\n",
        "d3": "def kadane2(a):\n    best = cur = 0\n    for x in a:\n        cur = max(x, cur + x)\n    return best\n",
        "u9": "def unlabelled():\n    return 42\n",
    }
    folds = {"q0": 0, "q1": 1, "q2": 2, "q3": 3}
    return queries, positive, docs, folds


class StubEncoder:
    """Vectors that make every document a plausible negative for every query, so the filters are what decide."""

    name = "stub"
    fingerprint = "stub-fp"
    prep_hash = "p"

    def encode(self, texts, *, is_query=False, route="generic", batch_size=64):
        rows = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            rows.append(unit([1.0] + [b / 255.0 for b in h[:7]]))
        return np.stack(rows)


def build(tmp_path, **overrides):
    queries, positive, docs, folds = toy()
    kwargs = dict(
        queries=queries,
        positive=positive,
        docs=docs,
        folds=folds,
        holdout_texts=["an unrelated held-out statement about trees and forests entirely"],
        encoder=StubEncoder(),
        mine_top=10,
        dev_ids=set(queries),
        meta={"test": True},
    )
    kwargs.update(overrides)
    return ax.build_bundle(tmp_path / "bundle", **kwargs)


def read_pairs(tmp_path):
    lines = (tmp_path / "bundle" / "pairs.jsonl").read_text().splitlines()
    return {row["qid"]: row for row in map(json.loads, lines)}


# -- mining ------------------------------------------------------------------------------------------------------
def test_the_positive_is_never_its_own_negative(tmp_path):
    build(tmp_path)
    for qid, row in read_pairs(tmp_path).items():
        assert row["positive"] not in [n for n, _ in row["negatives"]], qid


def test_an_unlabelled_document_is_never_a_negative(tmp_path):
    """Only positives of dev queries form the pool: training never sees a document outside the labelled set."""
    build(tmp_path)
    assert all("u9" not in [n for n, _ in row["negatives"]] for row in read_pairs(tmp_path).values())


def test_a_sibling_problems_solution_is_filtered_as_a_false_negative(tmp_path):
    """q3 is q0 with a different modulus: its solution is effectively a correct answer to q0, not a negative."""
    build(tmp_path)
    pairs = read_pairs(tmp_path)
    assert "d3" not in [n for n, _ in pairs["q0"]["negatives"]]
    assert pairs["q0"]["filtered"]["statement_overlap"] >= 1


def test_a_near_copy_of_the_positive_is_filtered(tmp_path):
    """d3 is d0 with a renamed function: token Jaccard far above 0.8 against q0's positive."""
    reasons = ax.false_negative_reason(
        pos_vec=unit([1, 0]),
        cand_vec=unit([0, 1]),
        pos_text=toy()[2]["d0"],
        cand_text=toy()[2]["d3"],
        query_text="anything",
        cand_query_text="something else entirely different here",
    )
    assert reasons == "token_jaccard"
    assert (
        ax.false_negative_reason(
            pos_vec=unit([1, 0]),
            cand_vec=unit([1, 0.01]),
            pos_text="a",
            cand_text="b",
            query_text="x",
            cand_query_text="y",
        )
        == "embedding"
    )


def test_every_negative_carries_its_fold(tmp_path):
    build(tmp_path)
    _, positive, _, folds = toy()
    fold_of_doc = {d: folds[q] for q, d in positive.items()}
    for row in read_pairs(tmp_path).values():
        for doc, fold in row["negatives"]:
            assert fold == fold_of_doc[doc]


# -- integrity ---------------------------------------------------------------------------------------------------
def test_a_decontaminated_query_is_gone_everywhere_its_document_included(tmp_path):
    queries, *_ = toy()
    build(tmp_path, holdout_texts=[queries["q1"]])
    pairs = read_pairs(tmp_path)
    assert "q1" not in pairs
    assert all("d1" not in [n for n, _ in row["negatives"]] for row in pairs.values())
    drop = json.loads((tmp_path / "bundle" / "drop_list.json").read_text())
    assert drop["decontaminated"] == ["q1"]


def test_a_held_out_id_stops_the_export(tmp_path):
    queries, positive, docs, folds = toy()
    with pytest.raises(SealedDataAccess):
        build(tmp_path, dev_ids={"q0", "q1", "q2"})  # q3 is not a dev query


def test_the_bundle_holds_no_held_out_text(tmp_path):
    secret = "an unrelated held-out statement about trees and forests entirely"
    build(tmp_path, holdout_texts=[secret])
    for path in (tmp_path / "bundle").iterdir():
        assert secret not in path.read_text(errors="ignore"), path.name


def test_the_manifest_hashes_every_file_and_the_training_config_is_declared(tmp_path):
    build(tmp_path)
    root = tmp_path / "bundle"
    listed = dict(line.split("  ", 1)[::-1] for line in (root / "MANIFEST.sha256").read_text().splitlines())
    for name, digest in listed.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    config = json.loads((root / "config.json").read_text())
    for key in ("tau", "lora_r", "lr", "epochs", "hard_negatives", "alphas", "folds", "seed"):
        assert key in config["train"]
    assert set(listed) >= {"pairs.jsonl", "corpus.jsonl", "folds.json", "drop_list.json", "config.json"}


def test_the_corpus_file_is_the_full_corpus_for_evaluation_only(tmp_path):
    """Out-of-fold evaluation ranks over the whole corpus, as the official task does; training never reads it."""
    build(tmp_path)
    corpus = [json.loads(line) for line in (tmp_path / "bundle" / "corpus.jsonl").read_text().splitlines()]
    assert {row["doc_id"] for row in corpus} == set(toy()[2])


# -- train/serve parity ------------------------------------------------------------------------------------------
def test_the_export_prepares_a_query_exactly_as_the_engine_encodes_it():
    """A query trained on one text and served as another is a skew no metric would point at."""
    from acis.core.config import freeze_config
    from acis.core.types import Snippet
    from acis.engine import AcisEngine
    from acis.engine.core import DEFAULT_CONFIG, query_encoder_text

    class Recording(StubEncoder):
        dim = 8
        submission_capable = True
        route_sensitive = False

        def __init__(self):
            self.seen = []

        def encode(self, texts, *, is_query=False, route="generic", batch_size=64):
            if is_query:
                self.seen.extend(texts)
            return super().encode(texts, is_query=is_query, route=route)

    config = freeze_config({**DEFAULT_CONFIG, "run": {**DEFAULT_CONFIG["run"], "channel": "dense"}})
    raw = "  Given   N (1 <= N <= 10^5) integers,\\n\\n-----Input-----\\nprint the answer  modulo 10^9+7.  " * 40
    encoder = Recording()
    engine = AcisEngine.from_config(config, encoder=encoder)
    snap = engine.build_snapshot([Snippet(handle="d", text="def f():\n    return 1\n")], source="parity")
    engine.search_batch(snap, ["q"], [raw], top_k=1)
    assert encoder.seen == [query_encoder_text(config, engine.normalise_query(raw)[0])]


def test_out_of_fold_evaluation_covers_every_dev_query_even_a_dropped_one(tmp_path):
    """A dropped query is never trained on, so it can be scored out of fold — and the OOF set stays complete."""
    queries, *_ = toy()
    build(tmp_path, holdout_texts=[queries["q1"]])
    rows = [json.loads(x) for x in (tmp_path / "bundle" / "eval_queries.jsonl").read_text().splitlines()]
    assert {r["qid"] for r in rows} == set(queries)
    assert "q1" not in read_pairs(tmp_path)
