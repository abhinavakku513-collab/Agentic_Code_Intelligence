"""The baseline ladder (docs/spec/03 §7, Phase 1 rungs B0 and B1).

A rung answers exactly one question and is recorded in the ledger with the data it ran on. Phase 1 ships the rungs
that validate the *harness* — nothing here is a submission:

| rung | system | question |
|---|---|---|
| `random` | seeded pseudo-random ranking | does the metric code behave on a run with no signal? |
| `oracle` | gold at rank 1 (dev labels) | does a perfect run score exactly 1.0 end to end? |
| `bm25_mteb` | `mteb/baseline-bm25s` | the independent lexical floor (B0) |
| `bm25_acis` | our per-snapshot BM25 | must match `bm25_mteb` rank-for-rank (B1, parity P2) |
| `stub_dense` | the harness stand-in encoder through our engine | does the Mode A pipeline run end to end? |

`oracle` reads dev labels on purpose — it exists to prove the metric path, is dev-only, and can never be a
submission rung.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from acis.appsdata import apps
from acis.core.errors import InvalidInput
from acis.core.types import EvalReport, EvalSpec
from acis.eval import ledger
from acis.eval.dev_task import dev_qrels
from acis.eval.metrics import K_VALUES, score_run
from acis.eval.splits import split_lock_hash

RUNG_NAMES = ("random", "oracle", "bm25_mteb", "bm25_acis", "stub_dense")
DEFAULT_TOP_K = 1000


@dataclass(slots=True)
class LadderResult:
    rung: str
    run: dict[str, dict[str, float]]
    metrics: dict[str, float]
    seconds: float
    n_queries: int
    n_docs: int
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _selected_queries(query_ids: Sequence[str] | None, limit: int) -> list[str]:
    ids = list(query_ids) if query_ids is not None else list(apps.dev_query_ids())
    return ids[:limit] if limit and limit > 0 else ids


# -- rungs -----------------------------------------------------------------------------------------------------
def run_random(query_ids: Sequence[str], *, top_k: int, seed: int = 0) -> dict[str, dict[str, float]]:
    """A seeded random ranking. Each query is sampled from its own generator, so batching changes nothing (INV-3)."""
    doc_ids = list(apps.corpus_ids())
    run: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        rng = random.Random(f"{seed}|{qid}")
        sample = rng.sample(doc_ids, min(top_k, len(doc_ids)))
        run[qid] = {doc: (top_k + 1 - rank) / top_k for rank, doc in enumerate(sample, start=1)}
    return run


def run_oracle(query_ids: Sequence[str], *, top_k: int) -> dict[str, dict[str, float]]:
    """Gold at rank 1, then corpus order. Dev-only; it reads labels to prove the metric path scores 1.0."""
    qrels = dev_qrels(query_ids)
    doc_ids = list(apps.corpus_ids())
    run: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        gold = [d for d, rel in qrels.get(qid, {}).items() if rel > 0]
        rest = [d for d in doc_ids if d not in set(gold)]
        ordered = (gold + rest)[:top_k]
        run[qid] = {doc: (top_k + 1 - rank) / top_k for rank, doc in enumerate(ordered, start=1)}
    return run


def run_bm25_acis(query_ids: Sequence[str], *, top_k: int) -> dict[str, dict[str, float]]:
    """Our per-snapshot BM25, matched to the mteb baseline's tokenisation. Raw scores, not rank-derived."""
    from acis.lexical.bm25 import Bm25Index  # noqa: PLC0415
    from acis.lexical.tokenize import corpus_text  # noqa: PLC0415

    docs = apps.load_documents()
    index = Bm25Index.build([d.doc_id for d in docs], [corpus_text(d.title, d.text) for d in docs])
    queries = apps.load_queries()
    texts = [queries[qid] for qid in query_ids]
    results = index.retrieve(texts, top_k)
    return {qid: dict(hits) for qid, hits in zip(query_ids, results, strict=True)}


def run_bm25_mteb(query_ids: Sequence[str], *, top_k: int) -> dict[str, dict[str, float]]:
    """`mteb/baseline-bm25s` driven directly, in the PyStemmer-free configuration both sides can execute."""
    from mteb.models.model_implementations.bm25 import BM25Search  # noqa: PLC0415

    from acis.eval.dev_task import build_split_data  # noqa: PLC0415
    from acis.lexical.tokenize import stemmer_available  # noqa: PLC0415

    split = build_split_data(query_ids)
    model = BM25Search(stopwords="en", stemmer_language="english" if stemmer_available() else "")
    task = _dev_metadata()
    model.index(split["corpus"], task_metadata=task, hf_split="train", hf_subset="default", encode_kwargs={})
    return model.search(
        split["queries"], task_metadata=task, hf_split="train", hf_subset="default", top_k=top_k, encode_kwargs={}
    )


def run_stub_dense(query_ids: Sequence[str], *, top_k: int) -> dict[str, dict[str, float]]:
    """The whole Mode A pipeline with the harness stand-in encoder: prep -> dense -> compose."""
    from acis.core.config import freeze_config  # noqa: PLC0415
    from acis.embed.hashing import HashingEncoder  # noqa: PLC0415
    from acis.engine import AcisEngine  # noqa: PLC0415
    from acis.engine.core import DEFAULT_CONFIG  # noqa: PLC0415
    from acis.rank.compose import rank_derived_scores  # noqa: PLC0415

    engine = AcisEngine.from_config(freeze_config(DEFAULT_CONFIG), encoder=HashingEncoder())
    snapshot = engine.build_snapshot(apps.load_corpus(), source="ladder:stub_dense")
    queries = apps.load_queries()
    ranked = engine.search_batch(snapshot, list(query_ids), [queries[q] for q in query_ids], top_k=top_k)
    return {qid: rank_derived_scores(hits, top_k) for qid, hits in ranked.items()}


def _dev_metadata() -> Any:
    from acis.eval.dev_task import make_dev_task  # noqa: PLC0415

    return make_dev_task().metadata


_RUNGS = {
    "random": run_random,
    "oracle": run_oracle,
    "bm25_mteb": run_bm25_mteb,
    "bm25_acis": run_bm25_acis,
    "stub_dense": run_stub_dense,
}


def run_rung(
    rung: str, *, query_ids: Sequence[str] | None = None, limit: int = 0, top_k: int = DEFAULT_TOP_K
) -> LadderResult:
    if rung not in _RUNGS:
        raise InvalidInput(f"unknown ladder rung {rung!r}; known: {RUNG_NAMES}")
    ids = _selected_queries(query_ids, limit)
    started = time.perf_counter()
    run = _RUNGS[rung](ids, top_k=top_k)
    seconds = time.perf_counter() - started
    metrics = score_run(dev_qrels(ids), run, K_VALUES)
    return LadderResult(
        rung=rung,
        run=run,
        metrics=metrics,
        seconds=seconds,
        n_queries=len(ids),
        n_docs=len(apps.corpus_ids()),
        notes="dev split only; absolute numbers are not comparable to official ones",
    )


# -- comparison and recording ------------------------------------------------------------------------------------
def top_k_agreement(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], k: int = 10) -> float:
    """Fraction of shared queries whose top-`k` document lists are identical (order included) — parity P2."""
    from acis.eval.metrics import rank_order  # noqa: PLC0415

    shared = sorted(set(a) & set(b))
    if not shared:
        return 0.0
    identical = sum(1 for q in shared if rank_order(a[q])[:k] == rank_order(b[q])[:k])
    return identical / len(shared)


def top_k_overlap(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], k: int = 10) -> float:
    """Mean set overlap of the top-`k` lists (order ignored)."""
    from acis.eval.metrics import rank_order  # noqa: PLC0415

    shared = sorted(set(a) & set(b))
    if not shared:
        return 0.0
    total = 0.0
    for q in shared:
        left, right = set(rank_order(a[q])[:k]), set(rank_order(b[q])[:k])
        total += len(left & right) / max(1, min(k, len(left), len(right)))
    return total / len(shared)


def record(result: LadderResult, *, kind: str = "dev", extra: Mapping[str, Any] | None = None) -> str:
    """Append a ladder result to the ledger and return its `run_id` (INV-14: numbers exist only here)."""
    row = (
        ledger.LedgerRowBuilder(kind=kind)
        .with_metrics(result.metrics)
        .with_fields(
            rung=result.rung,
            dataset="dev",
            n_queries=result.n_queries,
            n_docs=result.n_docs,
            split_lock_hash=_split_lock_hash_or_blank(),
            seconds=round(result.seconds, 3),
            notes=result.notes,
            **dict(extra or {}),
        )
        .build()
    )
    return ledger.append(row).run_id


def _split_lock_hash_or_blank() -> str:
    try:
        return split_lock_hash()
    except Exception:  # noqa: BLE001 — the lock is written once, by `acis eval splits --write`
        return ""


def run_spec(engine: Any, spec: EvalSpec) -> EvalReport:
    """`AcisEngine.evaluate()` — CLI-only (docs/spec/06 §1)."""
    if spec.kind not in ("dev", "ladder"):
        raise InvalidInput(f"evaluation kind {spec.kind!r} is not available yet (docs/spec/07)")
    rung = str(spec.options.get("rung", "bm25_acis"))
    limit = int(spec.options.get("limit", 0))
    result = run_rung(rung, limit=limit)
    run_id = record(result)
    return EvalReport(kind=spec.kind, run_id=run_id, metrics=result.metrics, notes=result.notes)


__all__ = [
    "DEFAULT_TOP_K",
    "RUNG_NAMES",
    "LadderResult",
    "record",
    "run_bm25_acis",
    "run_bm25_mteb",
    "run_oracle",
    "run_random",
    "run_rung",
    "run_spec",
    "run_stub_dense",
    "top_k_agreement",
    "top_k_overlap",
]
