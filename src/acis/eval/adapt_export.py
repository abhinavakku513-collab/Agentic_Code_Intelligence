"""Phase 3 export: the decontaminated training bundle for the GPU hand-off (docs/spec/02 §6, docs/GPU_HANDOFF.md).

`acis train export` writes, into one directory:

| file | content |
|---|---|
| `pairs.jsonl` | per kept dev query: fold, prepared text, positive doc id, negatives `[doc_id, fold]` |
| `eval_queries.jsonl` | every dev query with its fold, prepared — out-of-fold evaluation covers all of them |
| `corpus.jsonl` | every corpus document, prepared — **for out-of-fold evaluation only**; training never reads it |
| `folds.json` | the locked folds (`configs/splits.lock.json`), restricted to the kept queries |
| `drop_list.json` | what decontamination and exact deduplication removed, and why |
| `config.json` | the frozen training configuration and the provenance of the bundle |
| `MANIFEST.sha256` | a SHA-256 for every file above |

Four rules, each tested:

* **Nothing held out.** Every id is checked against the dev pool before a byte is written, and no held-out text
  appears anywhere — held-out query *texts* are read only to decide what to drop (decontamination).
* **Dropped means gone.** A query removed by decontamination or deduplication is absent from the pairs, and its
  positive document is absent from every negative pool.
* **Negatives come only from labelled documents.** The pool is the positives of kept dev queries, so training never
  shows the model an unlabelled document — which is what keeps a learned "is this a train solution" signal
  (CLAUDE.md §4) from being trainable at all. Each negative carries its fold, so a fold model refuses the held-out
  fold's documents and its out-of-fold score stays honest.
* **False negatives are filtered** (spec 02 §6): a sibling problem's solution (statement 5-gram Jaccard ≥ 0.7), a
  near-copy of the positive by embedding (cosine > 0.92) or by tokens (Jaccard > 0.8) is not a negative.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from acis.core.errors import SealedDataAccess
from acis.eval.decontam import decontaminate, exact_duplicate_groups, jaccard, shingles

BUNDLE_FORMAT = "acis-train-bundle/1"
MINE_TOP = 50
FN_EMBEDDING_COS = 0.92
FN_TOKEN_JACCARD = 0.8
FN_STATEMENT_OVERLAP = 0.7

#: The one training configuration, declared before any GPU time is spent (spec 02 §6 ranges; docs/GPU_HANDOFF.md
#: cheaper variants). Changing it is a new, logged trial — not a tweak.
TRAIN_CONFIG: dict[str, Any] = {
    "objective": "infonce",
    "tau": 0.05,
    "in_batch_negatives": True,
    "hard_negatives": 7,
    "batch_queries": 64,
    "grad_cache_chunk": 16,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["Wqkv", "Wo", "Wi"],
    "lr": 1e-4,
    "schedule": "cosine",
    "warmup_ratio": 0.05,
    "epochs": 2,
    "remine_after_epoch": 1,
    "max_query_tokens": 1024,
    "max_doc_tokens": 1024,
    "alphas": [0.25, 0.5, 0.75, 1.0],
    "folds": 5,
    "seed": 0,
    "precision": "bf16 where supported, else fp16 with loss scaling (T4)",
    "replay": None,
    "replay_note": "spec 02 §6 lists 25 % general code-retrieval replay; not included in this trial. Generality "
    "is protected by the alpha interpolation and checked by REG and G-OOD, which G3 requires.",
}

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _token_jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def false_negative_reason(
    *,
    pos_vec: np.ndarray,
    cand_vec: np.ndarray,
    pos_text: str,
    cand_text: str,
    query_text: str,
    cand_query_text: str,
) -> str | None:
    """Why a mined candidate is really a correct answer, or `None` when it is a genuine negative."""
    if jaccard(shingles(query_text), shingles(cand_query_text)) >= FN_STATEMENT_OVERLAP:
        return "statement_overlap"
    denominator = float(np.linalg.norm(pos_vec) * np.linalg.norm(cand_vec)) or 1.0
    if float(np.dot(pos_vec, cand_vec)) / denominator > FN_EMBEDDING_COS:
        return "embedding"
    if _token_jaccard(pos_text, cand_text) > FN_TOKEN_JACCARD:
        return "token_jaccard"
    return None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_bundle(
    out_dir: str | Path,
    *,
    queries: Mapping[str, str],
    positive: Mapping[str, str],
    docs: Mapping[str, str],
    folds: Mapping[str, int],
    holdout_texts: Iterable[str],
    encoder: Any,
    dev_ids: Iterable[str],
    prepare_query: Callable[[str], str] | None = None,
    prepare_doc: Callable[[str], str] | None = None,
    mine_top: int = MINE_TOP,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the bundle. `queries`/`positive`/`folds` cover dev queries only; `docs` is the whole corpus."""
    dev = {str(i) for i in dev_ids}
    outside = sorted(set(queries) - dev)
    if outside:
        raise SealedDataAccess(
            "ids outside the dev pool reached the training export", n_offenders=len(outside), example=outside[0]
        )
    prep_q = prepare_query or (lambda t: t)
    prep_d = prepare_doc or (lambda t: t)

    # 1. What training may not use: near-copies of held-out inputs, and exact duplicates inside the dev pool.
    report = decontaminate(dict(queries), list(holdout_texts))
    decontaminated = sorted(report.dropped)
    duplicates: list[str] = []
    for members in exact_duplicate_groups({q: t for q, t in queries.items() if q not in report.dropped}).values():
        duplicates.extend(members[1:])  # keep the first of each group
    dropped = set(decontaminated) | set(duplicates)
    kept = sorted((q for q in queries if q not in dropped), key=lambda q: (folds[q], q))
    dropped_docs = {positive[q] for q in dropped} - {positive[q] for q in kept}

    # 2. The negative pool: positives of kept queries, each labelled with its query's fold.
    pool_docs = sorted({positive[q] for q in kept} - dropped_docs)
    fold_of_doc = {positive[q]: folds[q] for q in kept}
    query_of_doc = {positive[q]: q for q in kept}

    prepared_q = {q: prep_q(queries[q]) for q in kept}
    prepared_d = {d: prep_d(t) for d, t in docs.items()}
    q_vecs = np.asarray(encoder.encode([prepared_q[q] for q in kept], is_query=True, route="generic"), np.float32)
    d_vecs = np.asarray(encoder.encode([prepared_d[d] for d in pool_docs]), np.float32)
    d_index = {d: i for i, d in enumerate(pool_docs)}

    # 3. Mining: the frozen model's top candidates, minus the positive and every false negative.
    scores = q_vecs @ d_vecs.T
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(kept):
        pos = positive[q]
        order = np.argsort(-scores[i], kind="stable")
        negatives: list[list[Any]] = []
        filtered = {"statement_overlap": 0, "embedding": 0, "token_jaccard": 0}
        for j in order:
            cand = pool_docs[int(j)]
            if cand == pos or docs[cand] == docs[pos]:
                continue
            reason = false_negative_reason(
                pos_vec=d_vecs[d_index[pos]],
                cand_vec=d_vecs[int(j)],
                pos_text=docs[pos],
                cand_text=docs[cand],
                query_text=queries[q],
                cand_query_text=queries[query_of_doc[cand]],
            )
            if reason:
                filtered[reason] += 1
                continue
            negatives.append([cand, fold_of_doc[cand]])
            if len(negatives) >= mine_top:
                break
        rows.append(
            {
                "qid": q,
                "fold": folds[q],
                "query": prepared_q[q],
                "positive": pos,
                "positive_fold": folds[q],
                "negatives": negatives,
                "filtered": filtered,
            }
        )

    # 4. Write, then hash what was written.
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Every dev query, dropped or not, for out-of-fold *evaluation*: a dropped query is never trained on in any fold,
    # so scoring it out of fold is honest, and the OOF decision set stays all 5,000 (docs/spec/09 §2).
    eval_rows = [
        {"qid": q, "fold": folds[q], "query": prep_q(queries[q])} for q in sorted(queries, key=lambda q: (folds[q], q))
    ]
    files: dict[str, bytes] = {
        "eval_queries.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in eval_rows).encode(),
        "pairs.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode(),
        "corpus.jsonl": "".join(
            json.dumps({"doc_id": d, "text": prepared_d[d]}, sort_keys=True) + "\n" for d in sorted(docs)
        ).encode(),
        "folds.json": json.dumps(
            {str(f): sorted(q for q in kept if folds[q] == f) for f in sorted(set(folds.values()))}, indent=1
        ).encode(),
        "drop_list.json": json.dumps(
            {
                "decontaminated": decontaminated,
                "exact_duplicates": sorted(duplicates),
                "threshold": report.threshold,
                "method": "MinHash char 5-grams, 128 permutations, confirmed exactly",
            },
            indent=1,
        ).encode(),
        "config.json": json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "train": TRAIN_CONFIG,
                "mining": {
                    "top": mine_top,
                    "pool": "positives of kept dev queries only",
                    "false_negative_filter": {
                        "statement_5gram_jaccard_ge": FN_STATEMENT_OVERLAP,
                        "embedding_cos_gt": FN_EMBEDDING_COS,
                        "token_jaccard_gt": FN_TOKEN_JACCARD,
                    },
                },
                "base_model": getattr(encoder, "name", "unknown"),
                "base_fingerprint": getattr(encoder, "fingerprint", "unknown"),
                "prep_hash": getattr(encoder, "prep_hash", ""),
                "git_sha": _git_sha(),
                "n_pairs": len(rows),
                "n_corpus": len(docs),
                "meta": dict(meta or {}),
            },
            indent=1,
            sort_keys=True,
        ).encode(),
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    (root / "MANIFEST.sha256").write_text("".join(f"{_sha(data)}  {name}\n" for name, data in sorted(files.items())))
    return {
        "n_pairs": len(rows),
        "decontaminated": len(decontaminated),
        "exact_duplicates": len(duplicates),
        "pool": len(pool_docs),
        "out": str(root),
    }


__all__ = [
    "BUNDLE_FORMAT",
    "FN_EMBEDDING_COS",
    "FN_STATEMENT_OVERLAP",
    "FN_TOKEN_JACCARD",
    "MINE_TOP",
    "TRAIN_CONFIG",
    "build_bundle",
    "false_negative_reason",
]
