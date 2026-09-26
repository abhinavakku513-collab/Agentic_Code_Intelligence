"""The prebuilt demo index (docs/spec/09 R9, D17): shipped, labelled, checksummed and recomputable.

A judge's `make demo` on a clean machine would otherwise embed the whole corpus before its first answer. The pack
fills the vector cache instead — and only the cache, which a cold official run never reads — so it can make a demo
fast but can never make a scored number faster.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from tests.unit.test_embed_runtime import StubBackend, make_runtime  # type: ignore[import-not-found]

from acis.core.config import freeze_config
from acis.core.errors import InvalidInput
from acis.core.types import Snippet
from acis.embed import demo_index
from acis.embed.cache import VectorCache
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG

DOCS = [f"def solve_{i}(xs):\n    return sorted(xs)[{i}]\n" for i in range(40)]


def runtime(tmp_path, name, **kw):
    return make_runtime(cache=VectorCache.open(tmp_path / name), threads=1, **kw)


def exported(tmp_path, docs=DOCS):
    source = runtime(tmp_path, "src")
    path = tmp_path / "demo.zip"
    demo_index.export_index(source, docs, path, meta={"corpus": "test"})
    return source, path


def test_an_imported_pack_serves_every_vector_without_a_forward_pass(tmp_path):
    source, path = exported(tmp_path)
    target = runtime(tmp_path, "dst")
    report = demo_index.import_index(target, path, texts=DOCS, verify_fraction=0.1)

    before = target.forward_calls
    vectors = target.encode(DOCS)
    assert report["imported"] == len(DOCS)
    np.testing.assert_allclose(vectors, source.encode(DOCS), atol=0)
    assert target.forward_calls - before == 0


def test_the_engine_finds_the_imported_vectors_under_its_own_document_preparation(tmp_path):
    """The keys must be the ones `build_snapshot` asks for, or the pack is a 27 MB file that saves nothing."""
    config = freeze_config(DEFAULT_CONFIG)
    prepared = demo_index.prepared_documents(config, DOCS)
    _, path = exported(tmp_path, prepared)
    target = runtime(tmp_path, "dst")
    demo_index.import_index(target, path, verify_fraction=0.0)

    before = target.forward_calls
    AcisEngine.from_config(config, encoder=target).build_snapshot(
        [Snippet(handle=f"d{i}", text=t) for i, t in enumerate(DOCS)], source="demo"
    )
    assert target.forward_calls - before == 0


def _rewrite(path, name, data: bytes) -> None:
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    members[name] = data
    with zipfile.ZipFile(path, "w") as zf:
        for n, content in members.items():
            zf.writestr(n, content)


def _npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def test_a_tampered_pack_is_refused(tmp_path):
    _, path = exported(tmp_path)
    with zipfile.ZipFile(path) as zf:
        vectors = np.load(io.BytesIO(zf.read("vectors.npy")), allow_pickle=False)
    _rewrite(path, "vectors.npy", _npy(vectors * 0.5))
    with pytest.raises(InvalidInput, match="checksum"):
        demo_index.import_index(runtime(tmp_path, "dst"), path, texts=DOCS)


class OtherWeights(StubBackend):
    @property
    def weights_digest(self) -> str:
        return "other-weights"


def test_a_pack_from_another_model_is_refused(tmp_path):
    _, path = exported(tmp_path)
    other = make_runtime(backend=OtherWeights(), cache=VectorCache.open(tmp_path / "o"), threads=1)
    with pytest.raises(InvalidInput, match="fingerprint"):
        demo_index.import_index(other, path, texts=DOCS)


def test_a_forged_pack_with_honest_checksums_fails_the_recompute(tmp_path):
    """Checksums prove the file is intact, not that its vectors came from this model: the recompute does."""
    _, path = exported(tmp_path)
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        vectors = np.load(io.BytesIO(zf.read("vectors.npy")), allow_pickle=False)
    forged = _npy(np.roll(vectors, 1, axis=0))
    import hashlib

    manifest["sha256"]["vectors.npy"] = hashlib.sha256(forged).hexdigest()
    _rewrite(path, "vectors.npy", forged)
    _rewrite(path, "manifest.json", json.dumps(manifest).encode())
    with pytest.raises(InvalidInput, match="recompute"):
        demo_index.import_index(runtime(tmp_path, "dst"), path, texts=DOCS, verify_fraction=1.0)


def test_the_pack_is_labelled_and_holds_no_pickle(tmp_path):
    _, path = exported(tmp_path)
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        np.load(io.BytesIO(zf.read("vectors.npy")), allow_pickle=False)
    assert "demo" in manifest["label"].lower() and "official" in manifest["label"].lower()
    assert manifest["n"] == len(DOCS) and manifest["meta"] == {"corpus": "test"}


def test_an_oversized_member_is_refused_before_it_is_decompressed(tmp_path, monkeypatch):
    """A pack is a downloaded file: a zip bomb must cost nothing."""
    _, path = exported(tmp_path)
    monkeypatch.setattr(demo_index, "MAX_MEMBER_BYTES", 16)
    with pytest.raises(InvalidInput, match="too large"):
        demo_index.import_index(runtime(tmp_path, "dst"), path, texts=DOCS)


def test_importing_the_same_pack_twice_is_a_no_op(tmp_path):
    """`make demo` imports the pack on every run; the second time must not recompute or rewrite anything."""
    _, path = exported(tmp_path)
    target = runtime(tmp_path, "dst")
    first = demo_index.import_index(target, path, texts=DOCS, verify_fraction=1.0)
    before = target.forward_calls
    second = demo_index.import_index(target, path, texts=DOCS, verify_fraction=1.0)
    assert first["imported"] == len(DOCS) and second["imported"] == 0 and second["already_imported"]
    assert target.forward_calls == before
