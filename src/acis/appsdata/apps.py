"""APPS / AppsRetrieval loaders — **dev split only** (docs/spec/03 §5, INV-8).

Verified layout at the pinned revision: `corpus` and `queries` each hold 8,765 rows with columns
`_id, partition, text, language, meta_information, title`; `data/train-*.parquet` holds 5,000
`query-id, corpus-id, score` triples. The corpus therefore contains the documents of both partitions, exactly as the
official task does — unlabelled documents are distractors, which is what makes the dev task mirror the real one.

**The `partition` column is metadata, never a feature.** Nothing downstream of these loaders may see it: a filter or
prior that knows "this document belongs to the training partition" is forbidden (DESIGN_RULES, INV-4). It is exposed
only to `dataset_audit()`, which writes a report a human reads.
"""

from __future__ import annotations

import ast
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from acis.appsdata.fetch import local_asset
from acis.appsdata.sources import DEV_SPLIT, SEALED_SPLIT
from acis.core.errors import InvalidInput, SealedDataAccess
from acis.core.hashing import fold_of, normalized_text_key
from acis.core.types import Snippet

CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
QRELS_FILE = f"data/{DEV_SPLIT}-00000-of-00001.parquet"


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus row. `partition` is audit-only metadata (see the module docstring)."""

    doc_id: str
    title: str
    text: str
    partition: str = ""
    meta_information: str = ""

    @property
    def mteb_text(self) -> str:
        """mteb's document convention: `"{title} {text}"` when a title exists, else the text."""
        return f"{self.title} {self.text}".strip() if self.title else self.text

    def as_snippet(self) -> Snippet:
        return Snippet(handle=self.doc_id, text=self.mteb_text)


def _read_table(repo_file: str) -> Any:
    import pyarrow.parquet as pq  # noqa: PLC0415 — parquet is only needed on the data path

    return pq.read_table(local_asset(repo_file))


def assert_dev_split(split: str) -> str:
    """Only the dev split may be loaded here. The held-out split belongs to `acis.eval.final` alone (INV-8)."""
    if split != DEV_SPLIT:
        raise SealedDataAccess(
            f"split {split!r} is not loadable in the dev environment; only {DEV_SPLIT!r} is "
            f"(the {SEALED_SPLIT!r} labels live outside the working tree, D19)"
        )
    return split


@lru_cache(maxsize=1)
def load_documents() -> tuple[Document, ...]:
    """All 8,765 corpus documents, in corpus order (the only ordering any code may rely on, INV-4)."""
    table = _read_table(CORPUS_FILE)
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    n = table.num_rows
    titles = cols.get("title") or [""] * n
    partitions = cols.get("partition") or [""] * n
    metas = cols.get("meta_information") or [""] * n
    return tuple(
        Document(
            doc_id=str(cols["_id"][i]),
            title=str(titles[i] or ""),
            text=str(cols["text"][i]),
            partition=str(partitions[i] or ""),
            meta_information=str(metas[i] or ""),
        )
        for i in range(n)
    )


def load_corpus() -> list[Snippet]:
    """The corpus as engine-level snippets. Handles are opaque; corpus order is preserved."""
    return [doc.as_snippet() for doc in load_documents()]


@lru_cache(maxsize=1)
def _all_queries() -> dict[str, str]:
    table = _read_table(QUERIES_FILE)
    ids = [str(v) for v in table.column("_id").to_pylist()]
    texts = [str(v) for v in table.column("text").to_pylist()]
    return dict(zip(ids, texts, strict=True))


@lru_cache(maxsize=1)
def load_qrels(split: str = DEV_SPLIT) -> Mapping[str, Mapping[str, int]]:
    """`{query_id: {doc_id: relevance}}` for the dev split only."""
    assert_dev_split(split)
    table = _read_table(QRELS_FILE)
    qids = [str(v) for v in table.column("query-id").to_pylist()]
    dids = [str(v) for v in table.column("corpus-id").to_pylist()]
    scores = [int(v) for v in table.column("score").to_pylist()]
    out: dict[str, dict[str, int]] = {}
    for qid, did, score in zip(qids, dids, scores, strict=True):
        out.setdefault(qid, {})[did] = score
    return out


def load_queries(split: str = DEV_SPLIT) -> dict[str, str]:
    """`{query_id: text}` restricted to the queries that have dev-split labels (5,000)."""
    assert_dev_split(split)
    every = _all_queries()
    missing = [qid for qid in load_qrels(split) if qid not in every]
    if missing:
        raise InvalidInput("dev queries missing from the queries file", n_missing=len(missing), example=missing[0])
    return {qid: every[qid] for qid in load_qrels(split)}


def all_query_ids() -> tuple[str, ...]:
    """Every query id in the queries file, in file order. **Ids only** — no label is read here (INV-8)."""
    return tuple(_all_queries().keys())


def query_text(query_id: str) -> str:
    """The text of any query id. Query texts are public inputs; labels are not, and are never loaded here."""
    every = _all_queries()
    if query_id not in every:
        raise InvalidInput(f"unknown query id: {query_id!r}")
    return every[query_id]


def dev_query_ids(split: str = DEV_SPLIT) -> tuple[str, ...]:
    return tuple(sorted(load_qrels(split).keys(), key=_natural_key))


def corpus_ids() -> tuple[str, ...]:
    return tuple(doc.doc_id for doc in load_documents())


def _natural_key(value: str) -> tuple[int, str]:
    digits = "".join(ch for ch in value if ch.isdigit())
    return (int(digits) if digits else 0, value)


def fold_assignments(split: str = DEV_SPLIT, folds: int = 5) -> dict[str, int]:
    """`sha256(query_text) mod folds` — deterministic, id-free fold membership (docs/spec/03 §5, INV-4)."""
    return {qid: fold_of(text, folds) for qid, text in load_queries(split).items()}


def is_available() -> bool:
    try:
        local_asset(CORPUS_FILE)
    except Exception:  # noqa: BLE001 — availability probe used by tests to skip cleanly
        return False
    return True


# -- G0.2 dataset audit --------------------------------------------------------------------------------------------
def _token_estimate(text: str) -> int:
    """Whitespace-token count: a cheap, model-independent length proxy for the audit."""
    return len(text.split())


def _percentiles(values: Sequence[int], points: Sequence[int] = (50, 90, 95, 99, 100)) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    out: dict[str, int] = {}
    for p in points:
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = ordered[idx]
    return out


def _alternate_solution_availability() -> dict[str, Any]:
    """Does a corpus row carry alternative solutions? (D7 / spec 02 §6 wants to train on a *different* solution.)

    `meta_information` is a Python-literal mapping. It is parsed with `ast.literal_eval` — a parser, never `eval`
    (INV-5) — and only its key inventory is reported.
    """
    keys: dict[str, int] = {}
    parsed = 0
    solution_like = 0
    for doc in load_documents():
        if not doc.meta_information:
            continue
        try:
            meta = ast.literal_eval(doc.meta_information)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if not isinstance(meta, dict):
            continue
        parsed += 1
        for key, value in meta.items():
            keys[str(key)] = keys.get(str(key), 0) + 1
            if "solution" in str(key).lower() and isinstance(value, (list, tuple)) and len(value) > 1:
                solution_like += 1
    return {
        "rows_with_meta": parsed,
        "meta_keys": dict(sorted(keys.items())),
        "rows_with_multiple_solutions": solution_like,
        "verdict": (
            "no alternative solutions in this corpus: a LoRA positive must be the corpus copy itself, "
            "so decontamination and the D1 exposure diagnostic carry the whole weight (spec 02 §6)"
            if solution_like == 0
            else "alternative solutions are available"
        ),
    }


def dataset_audit() -> dict[str, Any]:
    """G0.2: counts, id patterns, duplicates, length percentiles, marker inventory, overlap facts."""
    docs = load_documents()
    qrels = load_qrels()
    queries = load_queries()

    doc_tokens = [_token_estimate(d.text) for d in docs]
    q_tokens = [_token_estimate(t) for t in queries.values()]
    doc_chars = [len(d.text) for d in docs]
    q_chars = [len(t) for t in queries.values()]

    by_content: dict[str, list[str]] = {}
    for d in docs:
        by_content.setdefault(normalized_text_key(d.text), []).append(d.doc_id)
    duplicate_groups = {k: v for k, v in by_content.items() if len(v) > 1}

    exact: dict[str, list[str]] = {}
    for d in docs:
        exact.setdefault(d.text, []).append(d.doc_id)
    exact_groups = {k: v for k, v in exact.items() if len(v) > 1}

    markers = (
        "-----Input-----",
        "-----Output-----",
        "-----Example",
        "-----Note-----",
        "Input",
        "Output",
        "Example",
        "Constraints",
    )
    marker_counts = {m: sum(1 for t in queries.values() if m in t) for m in markers}

    partitions: dict[str, int] = {}
    for d in docs:
        partitions[d.partition or "unknown"] = partitions.get(d.partition or "unknown", 0) + 1

    labelled_docs = {did for rels in qrels.values() for did in rels}
    rels_per_query = [len(v) for v in qrels.values()]
    folds: dict[int, int] = {}
    for f in fold_assignments().values():
        folds[f] = folds.get(f, 0) + 1

    parse_ok = 0
    parse_fail_examples: list[str] = []
    with warnings.catch_warnings():
        # Corpus code raises SyntaxWarning (invalid escapes, Python-2 idioms). It is data, not our code.
        warnings.simplefilter("ignore")
        for d in docs:
            try:
                ast.parse(d.text)  # parse-only, never executed (INV-5)
                parse_ok += 1
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                if len(parse_fail_examples) < 5:
                    parse_fail_examples.append(d.doc_id)

    return {
        "dataset": {
            "corpus_docs": len(docs),
            "all_queries": len(_all_queries()),
            "dev_queries": len(queries),
            "dev_qrels": sum(rels_per_query),
        },
        "ids": {
            "doc_id_pattern": "d<int>" if all(d.doc_id.startswith("d") for d in docs) else "mixed",
            "query_id_pattern": "q<int>" if all(q.startswith("q") for q in queries) else "mixed",
            "doc_ids_sorted_equals_corpus_order": list(corpus_ids()) == sorted(corpus_ids(), key=_natural_key),
        },
        "labels": {
            "relevant_docs_per_query": {
                "min": min(rels_per_query),
                "max": max(rels_per_query),
                "mean": round(sum(rels_per_query) / len(rels_per_query), 4),
            },
            "distinct_labelled_docs": len(labelled_docs),
            "labelled_docs_are_subset_of_corpus": labelled_docs.issubset(set(corpus_ids())),
        },
        "partitions_metadata_only": partitions,
        "duplicates": {
            "exact_text_groups": len(exact_groups),
            "exact_text_docs": sum(len(v) for v in exact_groups.values()),
            "normalised_text_groups": len(duplicate_groups),
            "normalised_text_docs": sum(len(v) for v in duplicate_groups.values()),
            "example_group": next(iter(exact_groups.values()), []),
        },
        "lengths": {
            "doc_chars": {
                "mean": round(sum(doc_chars) / len(doc_chars), 1),
                **_percentiles(doc_chars),
                "min": min(doc_chars),
            },
            "doc_ws_tokens": {"mean": round(sum(doc_tokens) / len(doc_tokens), 1), **_percentiles(doc_tokens)},
            "query_chars": {
                "mean": round(sum(q_chars) / len(q_chars), 1),
                **_percentiles(q_chars),
                "min": min(q_chars),
            },
            "query_ws_tokens": {"mean": round(sum(q_tokens) / len(q_tokens), 1), **_percentiles(q_tokens)},
        },
        "markers": marker_counts,
        "folds": {str(k): v for k, v in sorted(folds.items())},
        "python_parse": {"ok": parse_ok, "failed": len(docs) - parse_ok, "fail_examples": parse_fail_examples},
        "alternate_solutions": _alternate_solution_availability(),
        "notes": [
            "partition counts are for this report only; no feature, filter or prior may read them (DESIGN_RULES)",
            "the corpus holds both partitions, so unlabelled documents act as distractors (docs/spec/03 §5)",
        ],
    }


def audit_json() -> str:
    return json.dumps(dataset_audit(), indent=2, sort_keys=True)


__all__ = [
    "CORPUS_FILE",
    "QRELS_FILE",
    "QUERIES_FILE",
    "Document",
    "all_query_ids",
    "assert_dev_split",
    "audit_json",
    "corpus_ids",
    "dataset_audit",
    "dev_query_ids",
    "fold_assignments",
    "is_available",
    "load_corpus",
    "load_documents",
    "load_qrels",
    "load_queries",
]
