#!/usr/bin/env python
"""Bulk-embed with a second encoder on the GPU, behind the parity gate (docs/spec/02 §3, ADR-0008).

Runs with the offline GPU tool environment (`scripts/train/gpu_env_setup.sh`), importing `acis` from `src/`:

    PYTHONPATH=src ~/.acis/gpu/venv/bin/python scripts/train/gpu_embed.py --model qwen3-embedding-0.6b

The encoder is the project's own pinned runtime (card, commit, file hashes, pooling, normalisation, prep), so every
vector is keyed exactly as the CPU reference profile keys it. Before a single GPU vector enters the shared cache:

1. a 1 % sample of every text set is encoded on the **CPU in fp32** (the reference profile);
2. the model moves to the GPU in fp16 and encodes the same sample;
3. the gate: every sampled pair must have cosine ≥ 0.999 (spec 02 §3). Failing it writes nothing.

Then everything not already cached is encoded on the GPU and written under the reference key, and an `audit`
ledger row records the parity statistics and that the vectors came from a GPU (disclosed, never hidden).
Text sets: the P0 corpus, all dev queries under both routes, the G-OOD perturbed dev sample, CosQA and
CodeSearchNet-Python (REG). No TEST data is touched.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "robustness"))

PARITY_MIN_COSINE = 0.999


def text_sets(config: Any, engine: Any) -> dict[str, tuple[list[str], bool, str]]:
    """name -> (texts, is_query, route). Documents are prepared exactly as the engine prepares them."""
    import perturb  # tests/robustness: the G-OOD families

    from acis.appsdata import apps
    from acis.core.paths import reg_home
    from acis.embed.demo_index import prepared_documents
    from acis.engine.core import query_encoder_texts

    def q_texts(raw: list[str]) -> list[str]:
        return [t for q in raw for t in query_encoder_texts(config, engine.normalise_query(q)[0])]

    sets: dict[str, tuple[list[str], bool, str]] = {}
    sets["p0-corpus"] = (prepared_documents(config, [d.mteb_text for d in apps.load_documents()]), False, "generic")
    queries = apps.load_queries()
    ids = list(apps.dev_query_ids())
    dev = q_texts([queries[q] for q in ids])
    sample = sorted(random.Random(0).sample(ids, 1000))
    families = (
        "format_noise",
        "strip_headings",
        "lower",
        "collapse_whitespace",
        "drop_last_paragraph",
        "sentence_dropout",
        "typos",
        "shuffle_paragraphs",
        "truncate",
    )
    perturbed = q_texts([getattr(perturb, f)(queries[q], seed=0) for f in families for q in sample])
    for route in ("statement_like", "generic"):
        sets[f"dev-queries-{route}"] = (dev, True, route)
        sets[f"g-ood-queries-{route}"] = (perturbed, True, route)
    for task in ("cosqa", "csn-python"):
        root = reg_home() / "export" / task
        if not root.is_dir():
            continue
        corpus = [json.loads(line)["text"] for line in (root / "corpus.jsonl").read_text("utf-8").splitlines()]
        sets[f"{task}-corpus"] = (prepared_documents(config, corpus), False, "generic")
        rows = [
            json.loads(line)["text"]
            for p in sorted(root.glob("queries.*.jsonl"))
            for line in p.read_text("utf-8").splitlines()
        ]
        for route in ("statement_like", "generic"):
            sets[f"{task}-queries-{route}"] = (q_texts(rows), True, route)
    return sets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-embedding-0.6b")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--sample", type=float, default=0.01)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from acis.core.config import load_frozen_config
    from acis.embed.factory import build_encoder
    from acis.engine import AcisEngine

    if not torch.cuda.is_available():
        print("no CUDA device in this environment; run scripts/train/gpu_env_setup.sh first", file=sys.stderr)
        return 2
    config = load_frozen_config(args.config).with_overrides(**{"model.encoder": args.model})
    runtime = build_encoder(config)  # the pinned CPU fp32 reference runtime, with the shared vector cache
    engine = AcisEngine.from_config(config, encoder=runtime)
    sets = text_sets(config, engine)
    print({k: len(v[0]) for k, v in sets.items()}, flush=True)

    # 1. CPU fp32 reference vectors for a sample of each set (computed directly, never read from the cache).
    rng = random.Random(0)
    samples: dict[str, list[int]] = {
        name: sorted(rng.sample(range(len(texts)), max(20, int(len(texts) * args.sample))))
        for name, (texts, _, _) in sets.items()
    }
    reference: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for name, (texts, is_query, route) in sets.items():
        rendered = [runtime._render(texts[i], is_query=is_query, route=route) for i in samples[name]]
        reference[name] = np.asarray(runtime._forward(rendered), dtype=np.float32)
    cpu_seconds = time.perf_counter() - started
    print(f"CPU fp32 reference sample: {sum(len(v) for v in samples.values())} texts in {cpu_seconds:.0f}s", flush=True)

    # 2. The same runtime on the GPU in fp16.
    backend = runtime.backend
    backend._model.to(device="cuda", dtype=torch.float16)
    cpu_forward = backend.forward

    def gpu_forward(batch: Any) -> Any:
        return cpu_forward({k: v.to("cuda") for k, v in batch.items()})

    backend.forward = gpu_forward  # type: ignore[method-assign]
    runtime.token_budget = args.token_budget

    # 3. The parity gate.
    worst = 1.0
    per_set: dict[str, dict[str, float]] = {}
    for name, (texts, is_query, route) in sets.items():
        rendered = [runtime._render(texts[i], is_query=is_query, route=route) for i in samples[name]]
        gpu = np.asarray(runtime._forward(rendered), dtype=np.float32)
        ref = reference[name]
        cos = (gpu * ref).sum(1) / (np.linalg.norm(gpu, axis=1) * np.linalg.norm(ref, axis=1))
        per_set[name] = {"n": int(len(cos)), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean())}
        worst = min(worst, float(cos.min()))
    print(json.dumps(per_set, indent=1), flush=True)
    if worst < PARITY_MIN_COSINE:
        print(f"PARITY FAILED: min cosine {worst:.6f} < {PARITY_MIN_COSINE}; nothing written", file=sys.stderr)
        return 1

    # 4. Everything not cached yet, on the GPU, written under the reference key.
    started = time.perf_counter()
    before = dict(runtime.cache.stats) if runtime.cache else {}
    for name, (texts, is_query, route) in sets.items():
        t = time.perf_counter()
        runtime.encode(texts, is_query=is_query, route=route)
        print(f"{name}: {len(texts)} texts in {time.perf_counter() - t:.0f}s", flush=True)
    gpu_seconds = time.perf_counter() - started
    after = dict(runtime.cache.stats) if runtime.cache else {}

    if not args.no_ledger:
        from acis.eval import ledger

        row = (
            ledger.LedgerRowBuilder(kind="audit")
            .with_metrics({"parity_min_cosine": worst, "gpu_seconds": gpu_seconds})
            .with_fields(
                rung=f"gpu-embed:{args.model}",
                model=runtime.name,
                model_fingerprint=runtime.fingerprint,
                profile_keyed_as=runtime.profile,
                computed_on=f"{torch.cuda.get_device_name(0)} fp16 (vectors admitted by the parity gate)",
                parity_threshold=PARITY_MIN_COSINE,
                parity=per_set,
                sets={k: len(v[0]) for k, v in sets.items()},
                cache_before=before,
                cache_after=after,
                cpu_reference_seconds=round(cpu_seconds, 1),
            )
            .build()
        )
        print("ledger:", ledger.append(row).run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
