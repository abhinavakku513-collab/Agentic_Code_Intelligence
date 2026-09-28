#!/usr/bin/env python
"""Phase 3 on a GPU: LoRA adaptation with K-fold out-of-fold vectors (docs/spec/02 §6, docs/GPU_HANDOFF.md).

Standalone on purpose — torch, transformers, peft, numpy and the standard library, nothing from `acis` — so a Colab
or Kaggle notebook needs only this file, the bundle from `acis train export` and the pinned base model.

    python train_lora.py --bundle bundle/ --base base_model/ --out out/ --resume auto

runs every stage in order and skips what is already done, so a notebook that loses its session just runs the same
command again:

1. five fold models, each trained on the other four folds; a fold model never sees its held-out fold's queries,
   positives or documents (negatives from that fold are dropped);
2. out-of-fold vectors: each fold model, merged at every alpha, embeds its held-out queries and the whole corpus;
3. the final model, trained on every kept pair with the same configuration;
4. `TRAIN_MANIFEST.json`: hashes of every input and output, the configuration, seeds, versions, GPU and wall-clock.

Outputs are LoRA adapters (safetensors, a few MB) and fp16 vectors. The alpha merge of the shipped model runs on
the CPU at import (`acis train import`), where it is checked against the vectors produced here (cosine >= 0.999).

Training detail: InfoNCE with in-batch and hard negatives, GradCache so a 64-query batch fits a T4, dropout RNG
replayed between the two GradCache passes, an in-batch negative whose text equals the query's positive masked out,
and after `remine_after_epoch` the hard negatives re-ranked by the current model within the export's filtered top
list. Checkpoints are safetensors + JSON; the optimizer state is loaded with `torch.load(weights_only=True)`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

# -- the bundle ---------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle(root: Path) -> dict[str, Any]:
    """Verify every file against MANIFEST.sha256, then load. A bundle that does not verify is not trained on."""
    manifest = {}
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    for name, digest in manifest.items():
        if sha256_file(root / name) != digest:
            raise SystemExit(f"bundle file {name} does not match MANIFEST.sha256; refusing to train")
    pairs = [json.loads(line) for line in (root / "pairs.jsonl").read_text().splitlines()]
    corpus = [json.loads(line) for line in (root / "corpus.jsonl").read_text().splitlines()]
    eval_queries = [json.loads(line) for line in (root / "eval_queries.jsonl").read_text().splitlines()]
    config = json.loads((root / "config.json").read_text())
    return {"pairs": pairs, "corpus": corpus, "eval_queries": eval_queries, "config": config, "manifest": manifest}


def training_rows(pairs: list[dict[str, Any]], held_out: int | None) -> list[dict[str, Any]]:
    """The rows a model may train on, each with only the negatives from folds it is allowed to see."""
    rows = []
    for row in pairs:
        if held_out is not None and row["fold"] == held_out:
            continue
        negatives = [doc for doc, fold in row["negatives"] if held_out is None or fold != held_out]
        rows.append({**row, "train_negatives": negatives})
    return rows


def batches(rows: list[dict[str, Any]], size: int, seed: int, epoch: int) -> list[list[dict[str, Any]]]:
    order = list(rows)
    random.Random(seed * 1000 + epoch).shuffle(order)
    return [order[i : i + size] for i in range(0, len(order), size) if len(order[i : i + size]) >= 2]


# -- the model ----------------------------------------------------------------------------------------------------


class RandContext:
    """Capture the RNG state before a forward pass so the GradCache replay sees the same dropout masks."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def __enter__(self) -> None:
        self._saved = (self.torch.get_rng_state(), self.torch.cuda.get_rng_state_all() if self.cuda else None)
        self.torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            self.torch.cuda.set_rng_state_all(self.cuda)

    def __exit__(self, *exc: object) -> None:
        self.torch.set_rng_state(self._saved[0])
        if self._saved[1] is not None:
            self.torch.cuda.set_rng_state_all(self._saved[1])


class Encoder:
    """CLS pooling + L2 normalisation — the card of gte-modernbert-base, used the way it is used at inference."""

    def __init__(self, base: Path, cfg: dict[str, Any], device: str, *, lora: bool = True) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(base))
        model = AutoModel.from_pretrained(str(base))
        if lora:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg["lora_r"],
                    lora_alpha=cfg["lora_alpha"],
                    lora_dropout=cfg["lora_dropout"],
                    target_modules=list(cfg["lora_targets"]),
                    bias="none",
                ),
            )
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
        self.model = model.to(device)

    def embed(self, texts: list[str], max_len: int) -> Any:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.model(**batch).last_hidden_state[:, 0]
        return self.torch.nn.functional.normalize(hidden.float(), dim=-1)

    def encode(self, texts: list[str], max_len: int, chunk: int = 64) -> np.ndarray:
        self.model.eval()
        out = []
        with self.torch.no_grad(), self.autocast():
            for i in range(0, len(texts), chunk):
                out.append(self.embed(texts[i : i + chunk], max_len).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def autocast(self) -> Any:
        if self.device.startswith("cuda"):
            dtype = self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
            return self.torch.autocast("cuda", dtype=dtype)
        return nullcontext()


def infonce(q: Any, d: Any, tau: float, same_as_positive: Any) -> Any:
    """Query i's positive is document i; `same_as_positive[i, j]` masks a document identical to that positive."""
    import torch

    scores = q @ d.T / tau
    scores = scores.masked_fill(same_as_positive, float("-inf"))
    return torch.nn.functional.cross_entropy(scores, torch.arange(q.shape[0], device=q.device))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def train_step(
    enc: Encoder, batch: list[dict[str, Any]], docs: dict[str, str], cfg: dict[str, Any], opt: Any, scaler: Any
) -> float:
    """One GradCache step: embed without graph, take gradients at the embeddings, replay chunk by chunk."""
    torch = enc.torch
    k, chunk = cfg["hard_negatives"], cfg["grad_cache_chunk"]
    q_texts = [r["query"] for r in batch]
    d_ids = [r["positive"] for r in batch] + [n for r in batch for n in r["train_negatives"][:k]]
    d_texts = [docs[d] for d in d_ids]
    pos_hash = [text_hash(docs[r["positive"]]) for r in batch]
    d_hash = [text_hash(t) for t in d_texts]
    same = torch.tensor(
        [[d_hash[j] == pos_hash[i] and j != i for j in range(len(d_texts))] for i in range(len(batch))],
        device=enc.device,
    )

    enc.model.train()
    groups = [("q", q_texts, cfg["max_query_tokens"]), ("d", d_texts, cfg["max_doc_tokens"])]
    reps, states = {}, {}
    with torch.no_grad(), enc.autocast():
        for name, texts, max_len in groups:
            parts, rng = [], []
            for i in range(0, len(texts), chunk):
                rng.append(RandContext(torch))
                parts.append(enc.embed(texts[i : i + chunk], max_len))
            reps[name], states[name] = torch.cat(parts), rng
    q = reps["q"].detach().requires_grad_()
    d = reps["d"].detach().requires_grad_()
    loss = infonce(q, d, cfg["tau"], same)
    loss.backward()
    grads = {"q": q.grad, "d": d.grad}

    for name, texts, max_len in groups:
        for n, i in enumerate(range(0, len(texts), chunk)):
            with states[name][n], enc.autocast():
                part = enc.embed(texts[i : i + chunk], max_len)
            surrogate = (part * grads[name][i : i + chunk]).sum()
            (scaler.scale(surrogate) if scaler else surrogate).backward()
    if scaler:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in enc.model.parameters() if p.requires_grad], 1.0)
    if scaler:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.item())


# -- checkpoints --------------------------------------------------------------------------------------------------


def save_checkpoint(path: Path, enc: Encoder, opt: Any, sched: Any, scaler: Any, state: dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    torch = enc.torch
    tmp = path.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous().cpu() for k, v in get_peft_model_state_dict(enc.model).items()},
        str(tmp / "adapter.safetensors"),
    )
    torch.save(
        {
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "trainer_state.pt",
    )
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    if path.exists():
        import shutil

        shutil.rmtree(path)
    os.replace(tmp, path)


def load_checkpoint(path: Path, enc: Encoder, opt: Any, sched: Any, scaler: Any) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    torch = enc.torch
    set_peft_model_state_dict(enc.model, load_file(str(path / "adapter.safetensors"), device=enc.device))
    trainer = torch.load(path / "trainer_state.pt", map_location=enc.device, weights_only=True)
    opt.load_state_dict(trainer["opt"])
    sched.load_state_dict(trainer["sched"])
    if scaler and trainer["scaler"]:
        scaler.load_state_dict(trainer["scaler"])
    torch.set_rng_state(trainer["rng_cpu"].cpu())
    if trainer["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in trainer["rng_cuda"]])
    return json.loads((path / "state.json").read_text())


# -- stages -------------------------------------------------------------------------------------------------------


def remine(enc: Encoder, rows: list[dict[str, Any]], docs: dict[str, str], cfg: dict[str, Any]) -> None:
    """Re-rank each row's allowed, already-filtered candidates by the current model (inside the training fold)."""
    cand_ids = sorted({d for r in rows for d in r["train_negatives"]})
    if not cand_ids:
        return
    d_vecs = enc.encode([docs[d] for d in cand_ids], cfg["max_doc_tokens"])
    q_vecs = enc.encode([r["query"] for r in rows], cfg["max_query_tokens"])
    index = {d: i for i, d in enumerate(cand_ids)}
    for r, qv in zip(rows, q_vecs, strict=True):
        cands = r["train_negatives"]
        scores = d_vecs[[index[c] for c in cands]] @ qv
        r["train_negatives"] = [cands[i] for i in np.argsort(-scores, kind="stable")]


def train_model(
    name: str,
    rows: list[dict[str, Any]],
    docs: dict[str, str],
    cfg: dict[str, Any],
    base: Path,
    out: Path,
    device: str,
    *,
    resume: bool,
    save_every: int,
    log: Any,
) -> Path:
    """Train one model; return the directory holding its final adapter. Resumes from the latest checkpoint."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    final = out / name / "adapter"
    if (final / "DONE").is_file():
        log(f"{name}: already trained")
        return final
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    enc = Encoder(base, cfg, device)
    params = [p for p in enc.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    per_epoch = len(batches(rows, cfg["batch_queries"], cfg["seed"], 0))
    total = max(1, per_epoch * cfg["epochs"])
    sched = get_cosine_schedule_with_warmup(opt, int(total * cfg["warmup_ratio"]), total)
    use_fp16 = device.startswith("cuda") and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    ckpt = out / name / "checkpoint"
    state = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "losses": [], "seconds": 0.0}
    if resume and ckpt.is_dir():
        state = load_checkpoint(ckpt, enc, opt, sched, scaler)
        log(f"{name}: resumed at epoch {state['epoch']} step {state['step_in_epoch']}")

    started = time.time() - state["seconds"]
    for epoch in range(state["epoch"], cfg["epochs"]):
        if epoch >= cfg["remine_after_epoch"]:
            remine(enc, rows, docs, cfg)
        plan = batches(rows, cfg["batch_queries"], cfg["seed"], epoch)
        for step in range(state["step_in_epoch"] if epoch == state["epoch"] else 0, len(plan)):
            loss = train_step(enc, plan[step], docs, cfg, opt, scaler)
            sched.step()
            state.update(epoch=epoch, step_in_epoch=step + 1, global_step=state["global_step"] + 1)
            state["losses"].append(round(loss, 5))
            if state["global_step"] % save_every == 0:
                state["seconds"] = time.time() - started
                save_checkpoint(ckpt, enc, opt, sched, scaler, state)
                log(f"{name}: epoch {epoch} step {step + 1}/{len(plan)} loss {loss:.4f} (checkpointed)")
        state.update(epoch=epoch + 1, step_in_epoch=0, seconds=time.time() - started)
        save_checkpoint(ckpt, enc, opt, sched, scaler, state)

    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    final.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous().cpu() for k, v in get_peft_model_state_dict(enc.model).items()},
        str(final / "adapter_model.safetensors"),
    )
    enc.model.peft_config["default"].save_pretrained(str(final))
    (final / "train_state.json").write_text(json.dumps(state, indent=1))
    (final / "DONE").write_text("ok")
    log(f"{name}: trained in {state['seconds']:.0f}s, final loss {state['losses'][-1] if state['losses'] else 'n/a'}")
    return final


def merged_encoder(base: Path, adapter: Path, alpha: float, cfg: dict[str, Any], device: str) -> Encoder:
    """θ = θ_base + α·Δ: scale every LoRA layer's contribution by α, then merge it into the base weights."""
    from peft import PeftModel

    enc = Encoder(base, cfg, device, lora=False)
    model = PeftModel.from_pretrained(enc.model, str(adapter))
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for key in scaling:
                scaling[key] = scaling[key] * alpha
    enc.model = model.merge_and_unload().to(device)
    return enc


def export_oof(
    fold: int,
    adapter: Path,
    eval_rows: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    cfg: dict[str, Any],
    base: Path,
    out: Path,
    device: str,
    log: Any,
) -> None:
    held = [r for r in eval_rows if r["fold"] == fold]
    for alpha in cfg["alphas"]:
        target = out / "oof" / f"fold{fold}" / f"alpha{alpha}"
        if (target / "DONE").is_file():
            continue
        enc = merged_encoder(base, adapter, alpha, cfg, device)
        target.mkdir(parents=True, exist_ok=True)
        np.save(
            target / "queries.npy",
            enc.encode([r["query"] for r in held], cfg["max_query_tokens"]).astype(np.float16),
            allow_pickle=False,
        )
        np.save(
            target / "corpus.npy",
            enc.encode([c["text"] for c in corpus], cfg["max_doc_tokens"]).astype(np.float16),
            allow_pickle=False,
        )
        (target / "qids.json").write_text(json.dumps([r["qid"] for r in held]))
        (target / "doc_ids.json").write_text(json.dumps([c["doc_id"] for c in corpus]))
        (target / "DONE").write_text("ok")
        log(f"oof fold {fold} alpha {alpha}: {len(held)} queries, {len(corpus)} documents")


def export_base(
    eval_rows: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    cfg: dict[str, Any],
    base: Path,
    out: Path,
    device: str,
    log: Any,
) -> None:
    """The unadapted base on the same GPU and precision: G3's baseline, so the gate compares like with like."""
    target = out / "oof" / "base"
    if (target / "DONE").is_file():
        return
    enc = Encoder(base, cfg, device, lora=False)
    target.mkdir(parents=True, exist_ok=True)
    np.save(
        target / "queries.npy",
        enc.encode([r["query"] for r in eval_rows], cfg["max_query_tokens"]).astype(np.float16),
        allow_pickle=False,
    )
    np.save(
        target / "corpus.npy",
        enc.encode([c["text"] for c in corpus], cfg["max_doc_tokens"]).astype(np.float16),
        allow_pickle=False,
    )
    (target / "qids.json").write_text(json.dumps([r["qid"] for r in eval_rows]))
    (target / "doc_ids.json").write_text(json.dumps([c["doc_id"] for c in corpus]))
    (target / "DONE").write_text("ok")
    log(f"base: {len(eval_rows)} queries, {len(corpus)} documents")


def export_parity_sample(
    adapter: Path,
    corpus: list[dict[str, Any]],
    cfg: dict[str, Any],
    base: Path,
    out: Path,
    device: str,
    *,
    fraction: float = 0.01,
) -> None:
    """Vectors of a fixed 1 % of the corpus under the final model at every alpha: the import's parity reference."""
    rng = random.Random(cfg["seed"])
    sample = sorted(rng.sample(range(len(corpus)), max(1, int(len(corpus) * fraction))))
    for alpha in cfg["alphas"]:
        target = out / "final" / "parity" / f"alpha{alpha}"
        if (target / "DONE").is_file():
            continue
        enc = merged_encoder(base, adapter, alpha, cfg, device)
        target.mkdir(parents=True, exist_ok=True)
        np.save(
            target / "vectors.npy",
            enc.encode([corpus[i]["text"] for i in sample], cfg["max_doc_tokens"]).astype(np.float32),
            allow_pickle=False,
        )
        (target / "doc_ids.json").write_text(json.dumps([corpus[i]["doc_id"] for i in sample]))
        (target / "DONE").write_text("ok")


def write_manifest(
    cfg: dict[str, Any],
    out: Path,
    bundle: dict[str, Any],
    base: Path,
    device: str,
    args: argparse.Namespace,
    seconds: float,
) -> None:
    import peft
    import torch
    import transformers

    outputs = {
        str(p.relative_to(out)): sha256_file(p)
        for p in sorted(out.rglob("*"))
        # train.log is appended on every run and resume, after the manifest: informational, not an output.
        if p.is_file() and "checkpoint" not in p.parts and p.name not in ("TRAIN_MANIFEST.json", "train.log")
    }
    manifest = {
        "format": "acis-train-output/1",
        "bundle_manifest": bundle["manifest"],
        "config": cfg,
        "overrides": json.loads(args.override) if args.override else {},
        "base_model": bundle["config"].get("base_model"),
        "base_files": {p.name: sha256_file(p) for p in sorted(base.iterdir()) if p.is_file()},
        "git_sha": args.git_sha,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "seconds": round(seconds, 1),
        "outputs": outputs,
    }
    (out / "TRAIN_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path, help="the pinned base model directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", default="auto", choices=["auto", "no"])
    parser.add_argument("--save-every", type=int, default=25, help="checkpoint every N optimizer steps")
    parser.add_argument("--git-sha", default="unknown", help="the commit this script was taken from")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", default="all", help="'all', or a comma list, e.g. 0,1 (for splitting sessions)")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--override", default="", help="JSON merged into the training config (smoke tests only)")
    args = parser.parse_args(argv)

    # Every model and tokenizer is read from a local directory. Offline mode turns a wrong path into an error
    # instead of a silent download of whatever the hub serves today.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    bundle = load_bundle(args.bundle)
    cfg = {**bundle["config"]["train"], **(json.loads(args.override) if args.override else {})}
    docs = {c["doc_id"]: c["text"] for c in bundle["corpus"]}
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    started = time.time()
    folds = range(cfg["folds"]) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    resume = args.resume == "auto"
    export_base(bundle["eval_queries"], bundle["corpus"], cfg, args.base, args.out, device, log)
    for fold in folds:
        rows = training_rows(bundle["pairs"], fold)
        adapter = train_model(
            f"fold{fold}",
            rows,
            docs,
            cfg,
            args.base,
            args.out,
            device,
            resume=resume,
            save_every=args.save_every,
            log=log,
        )
        export_oof(fold, adapter, bundle["eval_queries"], bundle["corpus"], cfg, args.base, args.out, device, log)
    if not args.skip_final:
        rows = training_rows(bundle["pairs"], None)
        adapter = train_model(
            "final", rows, docs, cfg, args.base, args.out, device, resume=resume, save_every=args.save_every, log=log
        )
        export_parity_sample(adapter, bundle["corpus"], cfg, args.base, args.out, device)
    write_manifest(cfg, args.out, bundle, args.base, device, args, time.time() - started)
    log(f"done in {time.time() - started:.0f}s; TRAIN_MANIFEST.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
