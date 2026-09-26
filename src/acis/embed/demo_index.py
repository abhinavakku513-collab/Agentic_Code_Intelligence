"""The prebuilt demo index (docs/spec/09 R9, D17): labelled, checksummed, recomputable.

`make demo` on a clean machine would embed the whole corpus before its first answer — hours on a CPU. D17 allows a
prebuilt demo index to ship instead, on three conditions this module enforces:

* **labelled** — the manifest says what it is, and that the official run does not use it;
* **checksummed** — every member is hashed in the manifest, and a pack whose model fingerprint or document
  preparation differs from the running encoder's is refused rather than silently mis-keyed;
* **recomputable** — import recomputes a sample and requires cosine ≥ 0.9999, so a pack whose checksums are
  honest but whose vectors did not come from this model is refused too.

The pack only fills the content-addressed vector cache. A cold official run never reads that cache, so the pack can
make a demo fast and can never make a scored number faster. Contents are `.npy` (loaded with `allow_pickle=False`)
and JSON inside a zip — no pickle anywhere (CLAUDE.md §4).
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from acis.core.errors import InvalidInput
from acis.prep.normalize import d1
from acis.prep.truncate import head_tail

FORMAT = "acis-demo-index/1"
LABEL = "Prebuilt DEMO index — document vectors for `make demo` only. The official run never reads it (D17)."
MIN_COSINE = 0.9999
_HEX_KEY = re.compile(r"[0-9a-f]{64}")
MEMBERS = ("vectors.npy", "keys.json")
#: A pack is a downloaded file, so its members are size-checked before anything is decompressed (zip bombs).
#: 8,765 x 1,024 fp32 is ~36 MB; the cap leaves room for a larger corpus without trusting the archive.
MAX_MEMBER_BYTES = 512 * 1024 * 1024


def prepared_documents(config: Any, texts: Sequence[str]) -> list[str]:
    """Exactly what the engine encodes for a document (`build_snapshot` → `_embed_documents`), so keys match."""
    prep = config.section("prep").get("doc", {})
    return [
        head_tail(
            d1(t),
            max_tokens=int(prep.get("max_tokens", 1024)),
            head=int(prep.get("head", 768)),
            tail=int(prep.get("tail", 256)),
        ).text
        for t in texts
    ]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_index(runtime: Any, texts: Sequence[str], out: str | Path, *, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Write the vectors of already-prepared document `texts` (served from the cache where present)."""
    vectors = np.asarray(runtime.encode(list(texts)), dtype=np.float32)
    keys = [runtime.document_cache_key(t) for t in texts]
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    payload = {"vectors.npy": buffer.getvalue(), "keys.json": json.dumps(keys).encode("utf-8")}
    # Texts ride along only as hashes: enough to recompute a sample on import from the caller's own corpus.
    manifest = {
        "format": FORMAT,
        "label": LABEL,
        "model": runtime.name,
        "fingerprint": runtime.fingerprint,
        "profile": runtime.profile,
        "prep_hash": runtime.prep_hash,
        "n": len(keys),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "sha256": {name: _sha(data) for name, data in payload.items()},
        "meta": dict(meta),
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in payload.items():
            zf.writestr(name, data)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def import_index(
    runtime: Any,
    path: str | Path,
    *,
    texts: Sequence[str] | None = None,
    verify_fraction: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Verify a pack against the running encoder and load it into its vector cache.

    `texts` are the prepared documents the pack claims to cover; with them, a sample is recomputed with no cache
    and compared. Without them the recompute uses the runtime's own forward pass on the keys it can match — which
    needs the texts — so the recompute is skipped only when `verify_fraction` is 0.
    """
    if runtime.cache is None:
        raise InvalidInput("the running encoder has no vector cache to import into")
    with zipfile.ZipFile(path) as zf:
        for name in ("manifest.json", *MEMBERS):
            if zf.getinfo(name).file_size > MAX_MEMBER_BYTES:
                raise InvalidInput(f"demo index member {name} is too large", limit=MAX_MEMBER_BYTES)
        manifest = json.loads(zf.read("manifest.json"))
        payload = {name: zf.read(name) for name in MEMBERS}
    if manifest.get("format") != FORMAT:
        raise InvalidInput("not an ACIS demo index", format=manifest.get("format"))
    for name, data in payload.items():
        if _sha(data) != manifest["sha256"].get(name):
            raise InvalidInput(f"demo index checksum mismatch in {name}: the file was altered or truncated")
    if manifest["fingerprint"] != runtime.fingerprint or manifest["prep_hash"] != runtime.prep_hash:
        raise InvalidInput(
            "demo index fingerprint does not match the running encoder (other weights, card, profile or prep)",
            pack=manifest["model"],
            running=runtime.name,
        )
    # A pack already imported into this cache is a no-op: `make demo` imports on every run.
    marker = Path(runtime.cache.root) / f".demo-index-{manifest['sha256']['vectors.npy'][:16]}"
    if marker.is_file():
        return {
            "imported": 0,
            "recomputed": 0,
            "worst_cosine": 1.0,
            "label": manifest["label"],
            "already_imported": True,
        }
    vectors = np.load(io.BytesIO(payload["vectors.npy"]), allow_pickle=False)
    keys = [str(k) for k in json.loads(payload["keys.json"])]
    if vectors.shape[0] != len(keys):
        raise InvalidInput("demo index vectors and keys disagree in length")
    bad = [k for k in keys if not _HEX_KEY.fullmatch(k)]
    if bad:
        raise InvalidInput("demo index holds a cache key that is not a content hash", n_bad=len(bad))

    checked, worst = 0, 1.0
    if verify_fraction > 0:
        if texts is None:
            raise InvalidInput("verifying a demo index needs the documents it claims to cover")
        by_key = {runtime.document_cache_key(t): t for t in texts}
        rng = random.Random(seed)
        sample = [i for i in range(len(keys)) if keys[i] in by_key and rng.random() < verify_fraction]
        sample = sample or [i for i in range(len(keys)) if keys[i] in by_key][:1]
        if not sample:
            raise InvalidInput("none of the demo index's vectors belong to these documents")
        cache, runtime.cache = runtime.cache, None
        try:
            fresh = np.asarray(runtime.encode([by_key[keys[i]] for i in sample]), dtype=np.float32)
        finally:
            runtime.cache = cache
        stored = vectors[sample]
        denominator = np.linalg.norm(stored, axis=1) * np.linalg.norm(fresh, axis=1)
        denominator[denominator == 0.0] = 1.0
        cosines = np.sum(stored * fresh, axis=1) / denominator
        checked, worst = len(sample), float(cosines.min())
        if worst < MIN_COSINE:
            raise InvalidInput(
                "demo index failed the recompute: its vectors did not come from this encoder",
                worst_cosine=round(worst, 6),
                sampled=checked,
            )
    imported = runtime.cache.put_many(zip(keys, vectors, strict=True))
    marker.write_text(json.dumps({"model": manifest["model"], "n": manifest["n"]}), encoding="utf-8")
    return {
        "imported": int(imported),
        "recomputed": checked,
        "worst_cosine": worst,
        "label": manifest["label"],
        "already_imported": False,
    }


__all__ = ["FORMAT", "LABEL", "MIN_COSINE", "export_index", "import_index", "prepared_documents"]
