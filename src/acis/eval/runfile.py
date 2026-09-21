"""Run files: `run.trec`, `run.csv` and the re-scoring path (docs/spec/03 §4).

`verify-submission` re-scores `run.trec` and must reproduce the JSON to 1e-9 (parity P3). That only works if writing
and reading a run is exactly round-trip stable, so both formats are written from the same ordering function and read
back with full float precision (`repr`-grade formatting, 17 significant digits).

`run.csv` carries an explicit `rank` column (O-05's default schema: `query_id,corpus_id,rank,score`).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from acis.core.errors import InvalidInput
from acis.eval.metrics import Run, rank_order

TREC_NAME = "run.trec"
CSV_NAME = "run.csv"
CSV_HEADER = ("query_id", "corpus_id", "rank", "score")


def _fmt(value: float) -> str:
    """Round-trip float formatting: `float(_fmt(x)) == x` for every finite float."""
    return repr(float(value))


def ranked_rows(run: Run, *, top_k: int | None = None) -> list[tuple[str, str, int, float]]:
    """`(query_id, doc_id, rank, score)` rows in grading order, query ids sorted for a stable file."""
    rows: list[tuple[str, str, int, float]] = []
    for qid in sorted(run):
        order = rank_order(run[qid])
        if top_k is not None:
            order = order[:top_k]
        for rank, doc_id in enumerate(order, start=1):
            rows.append((qid, doc_id, rank, float(run[qid][doc_id])))
    return rows


def write_trec(run: Run, path: str | Path, *, run_tag: str = "acis", top_k: int | None = None) -> Path:
    """TREC six-column format: `qid Q0 docid rank score tag`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for qid, doc_id, rank, score in ranked_rows(run, top_k=top_k):
            fh.write(f"{qid} Q0 {doc_id} {rank} {_fmt(score)} {run_tag}\n")
    return p


def read_trec(path: str | Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise InvalidInput(f"malformed TREC line {lineno} in {path}")
            qid, _, doc_id, _, score = parts[0], parts[1], parts[2], parts[3], parts[4]
            run.setdefault(qid, {})[doc_id] = float(score)
    return run


def write_csv(run: Run, path: str | Path, *, top_k: int | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for qid, doc_id, rank, score in ranked_rows(run, top_k=top_k):
            writer.writerow([qid, doc_id, rank, _fmt(score)])
    return p


def read_csv(path: str | Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_HEADER:
            raise InvalidInput(f"unexpected CSV header in {path}", header=reader.fieldnames)
        for row in reader:
            run.setdefault(row["query_id"], {})[row["corpus_id"]] = float(row["score"])
    return run


def write_run(run: Run, directory: str | Path, *, run_tag: str = "acis", top_k: int | None = None) -> dict[str, str]:
    d = Path(directory)
    return {
        "run.trec": str(write_trec(run, d / TREC_NAME, run_tag=run_tag, top_k=top_k)),
        "run.csv": str(write_csv(run, d / CSV_NAME, top_k=top_k)),
    }


def runs_equal(a: Run, b: Run, *, tol: float = 0.0) -> bool:
    """Exact (or tolerant) equality of two runs, used by the round-trip tests."""
    if set(a) != set(b):
        return False
    for qid in a:
        if set(a[qid]) != set(b[qid]):
            return False
        for doc_id, score in a[qid].items():
            if abs(float(score) - float(b[qid][doc_id])) > tol:
                return False
    return True


def rescore_trec(
    path: str | Path, qrels: Mapping[str, Mapping[str, int]], k_values: tuple[int, ...] | None = None
) -> dict[str, float]:
    """Independent re-scoring of a run file — the P3 parity check behind `verify-submission`."""
    from acis.eval.metrics import K_VALUES, score_run  # noqa: PLC0415

    return score_run(qrels, read_trec(path), k_values or K_VALUES)


__all__ = [
    "CSV_HEADER",
    "CSV_NAME",
    "TREC_NAME",
    "ranked_rows",
    "read_csv",
    "read_trec",
    "rescore_trec",
    "runs_equal",
    "write_csv",
    "write_run",
    "write_trec",
]
