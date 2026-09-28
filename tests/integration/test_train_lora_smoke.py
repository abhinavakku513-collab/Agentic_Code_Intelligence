"""The GPU trainer, end to end on a CPU with a tiny random ModernBERT (docs/GPU_HANDOFF.md §2).

Nothing here says the adaptation *works* — a two-layer random model on four pairs cannot. It says the notebook will
not die an hour in: every stage runs, a re-run skips what is done, a checkpoint resumes mid-training, a fold model
never trains on its held-out fold, and every output is hashed in TRAIN_MANIFEST.json.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("peft")

from tests.unit.test_adapt_export import StubEncoder, toy  # type: ignore[import-not-found]  # noqa: E402

GTE = Path.home() / ".acis" / "home" / "models" / "gte-modernbert-base"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train" / "train_lora.py"
SMOKE = {
    "epochs": 2,
    "remine_after_epoch": 1,
    "batch_queries": 2,
    "grad_cache_chunk": 1,
    "hard_negatives": 2,
    "folds": 2,
    "alphas": [0.5, 1.0],
    "max_query_tokens": 32,
    "max_doc_tokens": 32,
}


def trainer():
    spec = importlib.util.spec_from_file_location("train_lora", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_lora"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    if not (GTE / "tokenizer.json").is_file():
        pytest.skip("the gte-modernbert-base tokenizer is not fetched")
    from transformers import AutoModel, ModernBertConfig

    root = tmp_path_factory.mktemp("tiny")
    config = ModernBertConfig.from_pretrained(str(GTE))
    small = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    if getattr(config, "layer_types", None):
        small["layer_types"] = list(config.layer_types[:2])
    config.update(small)
    AutoModel.from_config(config).save_pretrained(str(root), safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(GTE / name, root / name)
    return root


@pytest.fixture
def bundle(tmp_path):
    from acis.eval import adapt_export as ax

    queries, positive, docs, folds = toy()
    folds = {q: i % 2 for i, q in enumerate(sorted(queries))}
    ax.build_bundle(
        tmp_path / "bundle",
        queries=queries,
        positive=positive,
        docs=docs,
        folds=folds,
        holdout_texts=["unrelated"],
        encoder=StubEncoder(),
        dev_ids=set(queries),
        mine_top=10,
    )
    return tmp_path / "bundle"


def run(tl, bundle, base, out, *extra):
    return tl.main(
        [
            "--bundle",
            str(bundle),
            "--base",
            str(base),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--save-every",
            "1",
            "--override",
            json.dumps(SMOKE),
            *extra,
        ]
    )


@pytest.mark.slow
def test_every_stage_runs_and_is_hashed(bundle, tiny_base, tmp_path):
    tl = trainer()
    out = tmp_path / "out"
    assert run(tl, bundle, tiny_base, out) == 0
    for fold in (0, 1):
        assert (out / f"fold{fold}" / "adapter" / "DONE").is_file()
        for alpha in SMOKE["alphas"]:
            d = out / "oof" / f"fold{fold}" / f"alpha{alpha}"
            q = np.load(d / "queries.npy", allow_pickle=False)
            c = np.load(d / "corpus.npy", allow_pickle=False)
            assert q.shape[0] == len(json.loads((d / "qids.json").read_text())) and c.shape[0] == 5
    assert (out / "final" / "adapter" / "DONE").is_file()
    assert (out / "oof" / "base" / "DONE").is_file()  # G3's same-hardware baseline
    manifest = json.loads((out / "TRAIN_MANIFEST.json").read_text())
    assert "final/adapter/adapter_model.safetensors" in manifest["outputs"]
    assert manifest["config"]["tau"] == 0.05 and manifest["bundle_manifest"]
    assert manifest["overrides"] == SMOKE and manifest["config"]["alphas"] == SMOKE["alphas"]  # what actually ran

    # LoRA's B matrices start at zero: if they are still zero, no gradient ever reached the adapter.
    from safetensors.numpy import load_file

    weights = load_file(str(out / "final" / "adapter" / "adapter_model.safetensors"))
    b = [v for k, v in weights.items() if "lora_B" in k]
    assert b and any(np.abs(v).max() > 0 for v in b)


@pytest.mark.slow
def test_a_rerun_skips_what_is_done(bundle, tiny_base, tmp_path):
    tl = trainer()
    out = tmp_path / "out"
    run(tl, bundle, tiny_base, out)
    before = (out / "final" / "adapter" / "adapter_model.safetensors").read_bytes()
    run(tl, bundle, tiny_base, out)
    assert (out / "final" / "adapter" / "adapter_model.safetensors").read_bytes() == before
    assert "already trained" in (out / "train.log").read_text()


@pytest.mark.slow
def test_training_resumes_from_its_checkpoint(bundle, tiny_base, tmp_path):
    tl = trainer()
    out = tmp_path / "out"
    run(tl, bundle, tiny_base, out, "--folds", "0", "--skip-final")
    shutil.rmtree(out / "fold0" / "adapter")  # as if the session died after the last checkpoint
    run(tl, bundle, tiny_base, out, "--folds", "0", "--skip-final")
    assert "resumed at epoch 2" in (out / "train.log").read_text()


def test_a_fold_model_never_sees_its_held_out_fold():
    tl = trainer()
    pairs = [
        {"qid": "a", "fold": 0, "negatives": [["x", 1], ["y", 0]]},
        {"qid": "b", "fold": 1, "negatives": [["z", 0], ["w", 1]]},
    ]
    rows = tl.training_rows(pairs, 0)
    assert [r["qid"] for r in rows] == ["b"] and rows[0]["train_negatives"] == ["w"]
    assert [r["train_negatives"] for r in tl.training_rows(pairs, None)] == [["x", "y"], ["z", "w"]]


def test_a_tampered_bundle_is_not_trained_on(bundle):
    tl = trainer()
    (bundle / "pairs.jsonl").write_text((bundle / "pairs.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit, match="MANIFEST"):
        tl.load_bundle(bundle)


# -- the import side (acis.embed.adapt_import) -------------------------------------------------------------------
def _trained(bundle, tiny_base, tmp_path):
    tl = trainer()
    out = tmp_path / "out"
    run(tl, bundle, tiny_base, out)
    return out


@pytest.mark.slow
def test_import_scores_out_of_fold_merges_and_passes_parity(bundle, tiny_base, tmp_path):
    from acis.embed import adapt_import

    out = _trained(bundle, tiny_base, tmp_path)
    _, positive, _, _ = toy()
    corpus = [json.loads(x) for x in (bundle / "corpus.jsonl").read_text().splitlines()]
    report = adapt_import.import_training(
        out,
        base_dir=tiny_base,
        qrels={q: {d: 1} for q, d in positive.items()},
        texts={c["doc_id"]: c["text"] for c in corpus},
        models_root=tmp_path / "models",
        report_path=tmp_path / "report.json",
    )
    assert set(report["oof"]) == {"base", "alpha0.5", "alpha1.0"}
    assert all(v["n_queries"] == 4 for v in report["oof"].values())  # every dev query, out of fold
    assert all(m["parity_worst_cosine"] >= adapt_import.PARITY_MIN_COSINE for m in report["merged"].values())
    assert (tmp_path / "models" / "gte-modernbert-base-apps-a1.0" / "model.safetensors").is_file()


@pytest.mark.slow
def test_an_altered_training_output_is_refused(bundle, tiny_base, tmp_path):
    from acis.core.errors import InvalidInput
    from acis.embed import adapt_import

    out = _trained(bundle, tiny_base, tmp_path)
    target = out / "oof" / "base" / "queries.npy"
    target.write_bytes(target.read_bytes()[:-8])
    with pytest.raises(InvalidInput, match="TRAIN_MANIFEST"):
        adapt_import.verify_outputs(out)
