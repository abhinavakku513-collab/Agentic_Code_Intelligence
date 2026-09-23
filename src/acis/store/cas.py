"""The content-addressed blob store (docs/spec/04 §1, D10, INV-1).

Two properties carry most of the weight of P1, and both are about what happens when the process dies:

* **Append-only.** A blob's name is the hash of its bytes, so a blob that exists is already correct and is never
  rewritten. That is what makes an incremental build cheap — unchanged content is recognised, not re-read — and
  it is what lets many snapshots share one copy of a body that appears in twenty versions.
* **Atomic.** Every write goes to a temporary file in the same directory, is flushed and fsynced, and is then
  `os.replace`d into place. A crash anywhere before the rename leaves a `.tmp-*` file that recovery deletes; it
  can never leave a truncated blob under a name that claims to hash to something else.

Blobs are zstd-compressed (ADR-0001 amendment). Compression is an implementation detail of the store: the hash
is always of the *uncompressed* UTF-8 text, so a body's identity never depends on a compression level.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acis.core.errors import InvalidInput, NotFound, ResourceLimit
from acis.core.hashing import sha256_text
from acis.obs.log import get_logger
from acis.store.layout import blob_path, cas_root

log = get_logger("acis.store")

DEFAULT_MAX_BLOB_BYTES = 4 << 20  # `Limits.max_unit_bytes`
COMPRESSION_LEVEL = 3


def body_hash(text: str) -> str:
    """The identity of a body: sha256 of its UTF-8 bytes. Normalisation (`d1`) happens before this, in `prep`."""
    return sha256_text(text)


def _codec() -> Any:
    import zstandard  # noqa: PLC0415 — keeps the import out of paths that never touch the store

    return zstandard


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(slots=True)
class BlobStore:
    """An append-only store of text bodies, addressed by the hash of their content."""

    root: Path
    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES
    stats: dict[str, int] = field(default_factory=lambda: {"writes": 0, "exists": 0, "reads": 0, "bytes": 0})

    @classmethod
    def open(cls, root: str | Path | None = None, *, max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES) -> BlobStore:
        directory = Path(root) if root is not None else cas_root()
        (directory / "blobs").mkdir(parents=True, exist_ok=True)
        return cls(root=directory, max_blob_bytes=max_blob_bytes)

    # -- writing ------------------------------------------------------------------------------------------------
    def put(self, text: str) -> str:
        """Store `text` and return its hash. Idempotent: an existing blob is left exactly as it is."""
        if not isinstance(text, str):
            raise InvalidInput("only text bodies are stored", got=type(text).__name__)
        raw = text.encode("utf-8")
        if len(raw) > self.max_blob_bytes:
            raise ResourceLimit("body exceeds the maximum blob size", size=len(raw), limit=self.max_blob_bytes)

        digest = body_hash(text)
        target = self._path(digest)
        if target.is_file():
            self.stats["exists"] += 1
            return digest

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _codec().ZstdCompressor(level=COMPRESSION_LEVEL).compress(raw)
        # Same directory as the target, so the rename is atomic on every filesystem we support.
        tmp = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex[:12]}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)  # last writer wins, and every writer wrote identical bytes
            _fsync_dir(target.parent)
        finally:
            tmp.unlink(missing_ok=True)
        self.stats["writes"] += 1
        self.stats["bytes"] += len(payload)
        return digest

    def put_many(self, texts: Iterable[str]) -> tuple[list[str], int]:
        """Store many bodies. Returns `(digests in input order, number of blobs newly written)`."""
        before = self.stats["writes"]
        digests = [self.put(text) for text in texts]
        return digests, self.stats["writes"] - before

    # -- reading ------------------------------------------------------------------------------------------------
    def has(self, digest: str) -> bool:
        return self._path(digest).is_file()

    def get(self, digest: str, *, verify: bool = False) -> str:
        """Read a body back. `verify=True` re-hashes it — the check the snapshot validator samples."""
        path = self._path(digest)
        if not path.is_file():
            raise NotFound(f"no blob {digest[:12]} in the content store")
        try:
            text = str(_codec().ZstdDecompressor().decompress(path.read_bytes()).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — zstandard and codecs raise their own types
            # Bytes under this name that cannot be read back *are* corruption. The caller gets our error rather
            # than a dependency's, because "the content store is damaged" is the fact, and it is actionable.
            raise InvalidInput(
                "stored blob does not match its hash; the content store is corrupt",
                blob=digest[:12],
                reason=type(exc).__name__,
            ) from exc
        self.stats["reads"] += 1
        if verify and body_hash(text) != digest:
            raise InvalidInput("stored blob does not match its hash; the content store is corrupt", blob=digest[:12])
        return text

    def get_many(self, digests: Sequence[str]) -> list[str]:
        return [self.get(d) for d in digests]

    def verify(self, digest: str) -> bool:
        """True when the bytes on disk still hash to their name. Never raises for ordinary corruption."""
        try:
            return body_hash(self.get(digest)) == digest
        except (NotFound, InvalidInput, ValueError, OSError):
            return False

    def _path(self, digest: str) -> Path:
        path = blob_path(digest)
        # `open(root=...)` may point somewhere other than the default CAS (tests, a staged build).
        return path if self.root == cas_root() else self.root / "blobs" / digest[:2] / digest[2:]


__all__ = ["COMPRESSION_LEVEL", "DEFAULT_MAX_BLOB_BYTES", "BlobStore", "body_hash"]
