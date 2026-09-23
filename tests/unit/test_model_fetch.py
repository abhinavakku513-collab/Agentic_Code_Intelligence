"""Fetching and pinning model weights — the model supply chain (D4, D15, CLAUDE.md §4).

This is the one place besides `acis fetch` that touches the network, and it is the place where a supply-chain
mistake would be easiest to make and hardest to see: a pickle-format checkpoint that executes on load, a model
swapped under a moving tag, a non-permissively licensed model quietly ending up in the submission.

The hub is injected, so every path here is tested without a network and without weights.
"""

from __future__ import annotations

import json

import pytest

from acis.core.errors import InvalidInput, NotReady
from acis.embed import modelfetch
from acis.embed.registry import load_card

CARD = "qwen3-embedding-0.6b"


class FakeHub:
    """A hub double: lists a repository and writes the files it claims to download."""

    def __init__(self, files, *, commit="a" * 40, bodies=None):
        self.files = list(files)
        self.commit = commit
        self.bodies = bodies or {}
        self.downloaded: list[str] = []

    def list_repo_files(self, repo_id, *, revision=None, **_):
        return list(self.files)

    def model_info(self, repo_id, *, revision=None, **_):
        return type("Info", (), {"sha": self.commit, "id": repo_id})()

    def hf_hub_download(self, *, repo_id, filename, revision=None, local_dir=None, **_):
        from pathlib import Path

        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.bodies.get(filename, f"content-of-{filename}"), encoding="utf-8")
        self.downloaded.append(filename)
        return str(target)


REPO_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
    "pytorch_model.bin",
    "onnx/model_quantized.onnx",
]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIS_HOME", str(tmp_path / "home"))


# -- what may be downloaded at all -------------------------------------------------------------------------------
def test_pickle_format_weights_are_never_downloaded():
    """`torch.load` on a `.bin` executes whatever is pickled in it. The fix is not to have the file."""
    wanted, skipped = modelfetch.plan_files(REPO_FILES)
    assert "model.safetensors" in wanted and "config.json" in wanted
    assert "pytorch_model.bin" in skipped
    assert not any(name.endswith((".bin", ".pt", ".pkl", ".h5", ".msgpack")) for name in wanted)


def test_a_repository_with_no_safetensors_is_refused_before_anything_downloads():
    hub = FakeHub(["config.json", "pytorch_model.bin"])
    with pytest.raises(NotReady, match="safetensors"):
        modelfetch.fetch_model(CARD, hub=hub)
    assert hub.downloaded == []


def test_documentation_and_images_are_not_part_of_the_model():
    wanted, _ = modelfetch.plan_files(REPO_FILES)
    assert "README.md" not in wanted


# -- licence and identity ------------------------------------------------------------------------------------------
def test_a_non_permissive_model_needs_an_explicit_reference_flag(tmp_path, monkeypatch):
    """D4: it may be *measured* for the gap it would close, never shipped — so asking for it has to be deliberate."""
    monkeypatch.setattr(modelfetch, "load_card", lambda key: _card_with(licence="cc-by-nc-4.0"))
    hub = FakeHub(REPO_FILES)
    with pytest.raises(InvalidInput, match="reference"):
        modelfetch.fetch_model(CARD, hub=hub)
    assert hub.downloaded == []

    fetched = modelfetch.fetch_model(CARD, hub=hub, reference=True)
    assert fetched.reference_only and fetched.files


def test_a_moving_revision_is_resolved_to_a_commit_and_recorded(monkeypatch):
    """A tag can move under us; a commit cannot. What was downloaded has to be nameable afterwards."""
    monkeypatch.setattr(modelfetch, "load_card", lambda key: _card_with(base_commit=None))
    hub = FakeHub(REPO_FILES, commit="b" * 40)
    fetched = modelfetch.fetch_model(CARD, hub=hub)
    assert fetched.commit == "b" * 40
    assert all(digest and len(digest) == 64 for digest in fetched.sha256.values())


def test_the_pinned_commit_is_used_when_the_card_has_one(monkeypatch):
    monkeypatch.setattr(modelfetch, "load_card", lambda key: _card_with(base_commit="c" * 40))
    hub = FakeHub(REPO_FILES, commit="d" * 40)  # the hub's head differs from the pin
    assert modelfetch.fetch_model(CARD, hub=hub).commit == "c" * 40


# -- pinning ---------------------------------------------------------------------------------------------------------
def test_pinning_writes_the_commit_and_every_file_hash_into_the_card(tmp_path, monkeypatch):
    card_path = tmp_path / f"{CARD}.yaml"
    card_path.write_text(json.dumps({"name": "Qwen/Q", "pooling": "last_token", "normalize": True}), encoding="utf-8")
    monkeypatch.setattr(modelfetch, "load_card", lambda key: _card_with(base_commit=None))
    hub = FakeHub(REPO_FILES, commit="e" * 40)

    fetched = modelfetch.fetch_model(CARD, hub=hub)
    modelfetch.pin_card(fetched, path=card_path)

    import yaml

    written = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    assert written["base_commit"] == "e" * 40
    assert written["file_sha256"]["model.safetensors"] == fetched.sha256["model.safetensors"]


def test_repinning_to_a_different_commit_is_refused(tmp_path):
    card_path = tmp_path / f"{CARD}.yaml"
    card_path.write_text(
        json.dumps({"name": "Qwen/Q", "pooling": "last_token", "normalize": True, "base_commit": "f" * 40}),
        encoding="utf-8",
    )
    fetched = modelfetch.fetch_model(CARD, hub=FakeHub(REPO_FILES, commit="0" * 40))
    with pytest.raises(InvalidInput, match="already pinned"):
        modelfetch.pin_card(fetched, path=card_path)


# -- verification ------------------------------------------------------------------------------------------------------
def test_a_changed_file_on_disk_is_caught_against_the_pins(monkeypatch):
    hub = FakeHub(REPO_FILES)
    fetched = modelfetch.fetch_model(CARD, hub=hub)
    monkeypatch.setattr(modelfetch, "load_card", lambda key: _card_with(file_sha256=dict(fetched.sha256)))
    assert modelfetch.verify_model(CARD) == []

    (fetched.model_dir / "model.safetensors").write_text("tampered", encoding="utf-8")
    problems = modelfetch.verify_model(CARD)
    assert problems and "model.safetensors" in problems[0]


def test_verifying_a_model_that_was_never_fetched_says_so():
    with pytest.raises(NotReady, match="no weights"):
        modelfetch.verify_model(CARD)


# -- where things land ---------------------------------------------------------------------------------------------
def test_weights_land_where_the_factory_looks_for_them():
    from acis.embed.factory import model_dir

    fetched = modelfetch.fetch_model(CARD, hub=FakeHub(REPO_FILES))
    assert fetched.model_dir == model_dir(CARD)
    assert (fetched.model_dir / "model.safetensors").is_file()


def _card_with(**overrides):
    from dataclasses import replace

    return replace(load_card(CARD), **overrides)
