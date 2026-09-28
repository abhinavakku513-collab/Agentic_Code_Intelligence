"""Phase 3 import: bring GPU training results home and check them (docs/GPU_HANDOFF.md §2 steps 5–6, spec 07).

`acis train import <out_dir>` takes the directory `scripts/train/train_lora.py` wrote and:

1. **verifies** every output against `TRAIN_MANIFEST.json` — a file that does not match is refused, not repaired;
2. **scores out of fold** with our own metric code: each fold model ranked its held-out queries over the whole
   corpus, so every dev query is scored by a model that never trained on it. The baseline is the unadapted base
   on the *same* GPU and precision (`oof/base`), so G3 compares like with like;
3. **merges** the final adapter on the CPU at every alpha (θ = θ_base + α·Δ) into safetensors under `ACIS_HOME`;
4. **checks parity**: each merged model re-embeds the GPU's 1 % sample on the CPU in fp32; cosine ≥ 0.999 or the
   import stops.

It decides nothing. The G3 decision needs these out-of-fold numbers *and* REG and G-OOD (docs/spec/03 §7), and it
is recorded through the gate procedure, never here. `peft` is imported only inside the merge: never at inference.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from acis.core.errors import InvalidInput
from acis.core.paths import acis_home

PARITY_MIN_COSINE = 0.999
TOP_K = 1000


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_outputs(out_dir: str | Path) -> dict[str, Any]:
    """Every file listed in TRAIN_MANIFEST.json must exist and hash to what the GPU side recorded."""
    root = Path(out_dir)
    manifest = json.loads((root / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    bad = [
        name
        for name, digest in manifest["outputs"].items()
        if not (root / name).is_file() or _sha(root / name) != digest
    ]
    if bad:
        raise InvalidInput("training outputs do not match TRAIN_MANIFEST.json", n_bad=len(bad), example=bad[0])
    return dict(manifest)


def _run(vec_dir: Path, top_k: int = TOP_K) -> dict[str, dict[str, float]]:
    """Exact dense ranking of one directory's queries over its corpus (D5), top `top_k` per query."""
    qids = json.loads((vec_dir / "qids.json").read_text())
    doc_ids = json.loads((vec_dir / "doc_ids.json").read_text())
    q = np.load(vec_dir / "queries.npy", allow_pickle=False).astype(np.float32)
    d = np.load(vec_dir / "corpus.npy", allow_pickle=False).astype(np.float32)
    scores = q @ d.T
    k = min(top_k, d.shape[0])
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(qids):
        row = scores[i]
        top = np.argpartition(-row, k - 1)[:k] if k < row.shape[0] else np.arange(row.shape[0])
        run[str(qid)] = {str(doc_ids[j]): float(row[j]) for j in top}
    return run


def oof_report(
    out_dir: str | Path, qrels: Mapping[str, Mapping[str, int]], alphas: Sequence[float], folds: int
) -> dict[str, Any]:
    """Per-query NDCG@10 and mean metrics for the base and for every alpha, out of fold; alpha vs base bootstrapped."""
    from acis.eval.bootstrap import paired_bootstrap
    from acis.eval.metrics import per_query, score_run

    root = Path(out_dir) / "oof"
    systems: dict[str, dict[str, dict[str, float]]] = {"base": _run(root / "base")}
    for alpha in alphas:
        run: dict[str, dict[str, float]] = {}
        for fold in range(folds):
            target = root / f"fold{fold}" / f"alpha{alpha}"
            if target.is_dir():
                run.update(_run(target))
        systems[f"alpha{alpha}"] = run

    report: dict[str, Any] = {}
    base_pq = per_query(qrels, systems["base"], "ndcg", 10)
    for name, run in systems.items():
        pq = per_query(qrels, run, "ndcg", 10)
        entry: dict[str, Any] = {"n_queries": len(pq), "metrics": score_run(qrels, run), "per_query": pq}
        if name != "base":
            boot = paired_bootstrap(pq, base_pq)
            entry["vs_base"] = {"delta_pts": boot.delta, "ci": [boot.ci_low, boot.ci_high]}
        report[name] = entry
    return report


def merge_alpha(base_dir: str | Path, adapter_dir: str | Path, alpha: float, target: str | Path) -> dict[str, str]:
    """θ = θ_base + α·Δ on the CPU, saved as safetensors with the base's tokenizer. Returns file hashes."""
    from peft import PeftModel
    from transformers import AutoModel

    base, out = Path(base_dir), Path(target)
    model = PeftModel.from_pretrained(AutoModel.from_pretrained(str(base)), str(adapter_dir))
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for key in scaling:
                scaling[key] = scaling[key] * alpha
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    for path in base.iterdir():
        if path.is_file() and (path.name.startswith("tokenizer") or path.name in ("special_tokens_map.json",)):
            shutil.copy(path, out / path.name)
        if path.is_dir() and path.name == "1_Pooling":
            shutil.copytree(path, out / path.name, dirs_exist_ok=True)
    return {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()}


def cpu_vectors(model_dir: str | Path, texts: Sequence[str], max_len: int = 1024) -> np.ndarray:
    """CLS + L2 normalisation in fp32 on the CPU — how the card uses the model at inference."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tokenizer(
                list(texts[i : i + 16]), padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            )
            hidden = model(**batch).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(hidden, dim=-1).numpy())
    return np.concatenate(out)


def parity(model_dir: str | Path, parity_dir: str | Path, texts: Mapping[str, str], max_len: int = 1024) -> float:
    """Worst cosine between the GPU's vectors of the parity sample and the merged model's CPU fp32 vectors."""
    root = Path(parity_dir)
    doc_ids = json.loads((root / "doc_ids.json").read_text())
    gpu = np.load(root / "vectors.npy", allow_pickle=False).astype(np.float32)
    cpu = cpu_vectors(model_dir, [texts[d] for d in doc_ids], max_len)
    denominator = np.linalg.norm(gpu, axis=1) * np.linalg.norm(cpu, axis=1)
    denominator[denominator == 0] = 1.0
    return float((np.sum(gpu * cpu, axis=1) / denominator).min())


def import_training(
    out_dir: str | Path,
    *,
    base_dir: str | Path,
    qrels: Mapping[str, Mapping[str, int]],
    texts: Mapping[str, str],
    models_root: str | Path | None = None,
    model_key: str = "gte-modernbert-base-apps",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify, score out of fold, merge at every alpha, check parity. Returns (and writes) the report."""
    root = Path(out_dir)
    manifest = verify_outputs(root)
    cfg = manifest["config"]
    alphas = [float(a) for a in cfg["alphas"]]
    oof = oof_report(root, qrels, alphas, int(cfg["folds"]))

    models = Path(models_root) if models_root else acis_home() / "models"
    merged: dict[str, Any] = {}
    for alpha in alphas:
        target = models / f"{model_key}-a{alpha}"
        hashes = merge_alpha(base_dir, root / "final" / "adapter", alpha, target)
        worst = parity(target, root / "final" / "parity" / f"alpha{alpha}", texts, int(cfg["max_doc_tokens"]))
        if worst < PARITY_MIN_COSINE:
            raise InvalidInput(
                "parity gate failed: the merged model on the CPU does not reproduce the GPU's vectors",
                alpha=alpha,
                worst_cosine=round(worst, 6),
            )
        merged[f"alpha{alpha}"] = {"dir": str(target), "files": hashes, "parity_worst_cosine": worst}

    report = {
        "train_manifest_sha256": _sha(root / "TRAIN_MANIFEST.json"),
        "gpu": manifest.get("gpu"),
        "seconds": manifest.get("seconds"),
        "config": cfg,
        "oof": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"} for k, v in oof.items()},
        "merged": merged,
        "parity_min_cosine": PARITY_MIN_COSINE,
    }
    if report_path:
        target_report = Path(report_path)
        target_report.parent.mkdir(parents=True, exist_ok=True)
        target_report.write_text(
            json.dumps({**report, "per_query": {k: v["per_query"] for k, v in oof.items()}}, indent=1, sort_keys=True)
        )
    return report


__all__ = ["PARITY_MIN_COSINE", "import_training", "merge_alpha", "oof_report", "parity", "verify_outputs"]
