"""The dev retrieval task — an `AbsTaskRetrieval` over the **dev-split labels only** (docs/spec/03 §5).

Dev must exercise the same code the official run exercises: the same adapter, the same dispatch, the same metric
functions. So it is a real mteb task, not a private loop. Two differences from the official task, both deliberate:

* it loads the dev split, and only the dev split — `load_data()` never reaches the network and never sees the
  held-out labels (INV-8);
* it keeps the **full 8,765-document corpus**, so unlabelled documents act as distractors exactly as they do in the
  real task. Dev absolute numbers are still not comparable to official ones; decisions use paired deltas only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from acis.appsdata import apps
from acis.appsdata.sources import APPS, DEV_SPLIT
from acis.core.errors import InvalidInput
from acis.eval.guard import assert_dev_split

DEV_TASK_NAME = "AcisAppsDev"


def _dataset(rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset  # noqa: PLC0415

    return Dataset.from_list(rows)


def build_split_data(query_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """`{corpus, queries, relevant_docs, top_ranked}` for the dev split, optionally restricted to some queries."""
    qrels = apps.load_qrels(DEV_SPLIT)
    queries = apps.load_queries(DEV_SPLIT)
    selected = list(query_ids) if query_ids is not None else list(queries)
    unknown = [q for q in selected if q not in queries]
    if unknown:
        raise InvalidInput("unknown dev query ids", n=len(unknown), example=unknown[0])

    corpus_rows = [{"id": d.doc_id, "title": d.title, "text": d.text} for d in apps.load_documents()]
    query_rows = [{"id": qid, "text": queries[qid]} for qid in selected]
    relevant = {qid: dict(qrels[qid]) for qid in selected}
    return {
        "corpus": _dataset(corpus_rows),
        "queries": _dataset(query_rows),
        "relevant_docs": relevant,
        "top_ranked": None,
    }


def make_dev_task(query_ids: Sequence[str] | None = None) -> Any:
    """Build the dev task class lazily, so importing this module costs nothing when mteb is not needed."""
    from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: PLC0415
    from mteb.abstasks.task_metadata import TaskMetadata  # noqa: PLC0415

    class AcisAppsDev(AbsTaskRetrieval):
        """AppsRetrieval restricted to the dev split — the harness's decision set (all 5,000 queries by default)."""

        metadata = TaskMetadata(
            name=DEV_TASK_NAME,
            description=(
                "Dev-split mirror of AppsRetrieval: dev queries against the full corpus, used for every gate "
                "decision. Never uses held-out labels."
            ),
            reference="https://arxiv.org/abs/2105.09938",
            dataset={"path": APPS.repo, "revision": APPS.revision},
            type="Retrieval",
            category="t2t",
            modalities=["text"],
            eval_splits=[DEV_SPLIT],
            eval_langs=["eng-Latn", "python-Code"],
            main_score="ndcg_at_10",
            date=("2021-05-20", "2021-05-20"),
            domains=["Programming", "Written"],
            task_subtypes=["Code retrieval"],
            license="mit",
            annotations_creators="derived",
            dialect=[],
            sample_creation="found",
            bibtex_citation="",
        )

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            """Load from the local, allow-listed assets. No network, no held-out labels (INV-8)."""
            if self.data_loaded:
                return
            assert_dev_split(DEV_SPLIT, context="dev task")
            # `RetrievalSplitData` is a TypedDict; the loader builds the same keys from local assets.
            self.dataset = {"default": {DEV_SPLIT: cast("Any", build_split_data(query_ids))}}
            self.data_loaded = True

    return AcisAppsDev()


def dev_qrels(query_ids: Sequence[str] | None = None) -> Mapping[str, Mapping[str, int]]:
    qrels = apps.load_qrels(DEV_SPLIT)
    if query_ids is None:
        return qrels
    return {qid: qrels[qid] for qid in query_ids if qid in qrels}


__all__ = ["DEV_TASK_NAME", "build_split_data", "dev_qrels", "make_dev_task"]
