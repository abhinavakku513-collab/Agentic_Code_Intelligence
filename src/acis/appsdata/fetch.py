"""`acis fetch` — the only code path in ACIS that touches the network (AGENTS.md, D15).

It lists the pinned dataset repository, downloads **only** allow-listed, non-sealed files into `ACIS_HOME`, records a
SHA-256 manifest, and then re-verifies that no sealed pattern landed in the dev environment (G0.6). The held-out
labels are fetched separately, by the owner, with `--sealed`, into `~/.acis-sealed/hf`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from acis.appsdata.sources import APPS, DatasetPin, is_dev_allowed, is_sealed_file, partition
from acis.core.errors import InvalidInput, SealedDataAccess
from acis.core.hashing import sha256_file
from acis.core.paths import acis_home, sealed_root
from acis.obs.log import get_logger

log = get_logger("acis.fetch")
MANIFEST_NAME = "assets_manifest.json"
SEALED_OPT_IN = "ACIS_ALLOW_SEALED_FETCH"


@dataclass(frozen=True, slots=True)
class FetchedAsset:
    repo_file: str
    local_path: str
    sha256: str
    n_bytes: int


def asset_root() -> Path:
    """Where dev-visible dataset assets live (inside `ACIS_HOME`, outside git)."""
    root = acis_home() / "hf"
    root.mkdir(parents=True, exist_ok=True)
    return root


def manifest_path() -> Path:
    return acis_home() / MANIFEST_NAME


def _hub(online: bool) -> Any:
    if online:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"
    import huggingface_hub  # noqa: PLC0415 — imported only when the network path is used

    return huggingface_hub


def list_repo_files(pin: DatasetPin = APPS) -> list[str]:
    """List the pinned repository at its pinned revision (network)."""
    hub = _hub(online=True)
    return sorted(hub.list_repo_files(pin.repo, repo_type="dataset", revision=pin.revision))


def fetch_dev(pin: DatasetPin = APPS, *, force: bool = False) -> list[FetchedAsset]:
    """Download every allow-listed, non-sealed file of the pinned revision into `ACIS_HOME`."""
    files = list_repo_files(pin)
    allowed, sealed, ignored = partition(files)
    if not allowed:
        raise InvalidInput("no allow-listed files found in the dataset repository", repo=pin.repo)
    log.info(
        "fetch.plan",
        repo=pin.repo,
        revision=pin.revision[:12],
        allowed=len(allowed),
        sealed=len(sealed),
        ignored=len(ignored),
    )

    hub = _hub(online=True)
    root = asset_root()
    assets: list[FetchedAsset] = []
    for name in allowed:
        local = hub.hf_hub_download(
            repo_id=pin.repo,
            filename=name,
            repo_type="dataset",
            revision=pin.revision,
            cache_dir=str(root),
            force_download=force,
        )
        digest = sha256_file(local)
        assets.append(
            FetchedAsset(repo_file=name, local_path=str(local), sha256=digest, n_bytes=Path(local).stat().st_size)
        )
        log.info("fetch.asset", file=name, sha256=digest[:12], bytes=assets[-1].n_bytes)

    write_manifest(pin, assets, sealed_skipped=sealed)
    assert_seal()
    return assets


def write_manifest(pin: DatasetPin, assets: list[FetchedAsset], *, sealed_skipped: list[str]) -> Path:
    payload = {
        "dataset": asdict(pin),
        "fetched_ts": time.time(),
        "assets": [asdict(a) for a in assets],
        "sealed_files_skipped": sealed_skipped,
        "note": "held-out labels are fetched by the owner into ~/.acis-sealed/hf (D19); never into ACIS_HOME",
    }
    path = manifest_path()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    if not path.is_file():
        raise InvalidInput(f"dataset manifest not found: run `make fetch` first ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest() -> list[str]:
    """Re-hash every fetched asset. Returns the list of problems (empty == clean)."""
    problems: list[str] = []
    for asset in load_manifest().get("assets", []):
        local = Path(asset["local_path"])
        if not local.is_file():
            problems.append(f"missing: {asset['repo_file']}")
        elif sha256_file(local) != asset["sha256"]:
            problems.append(f"checksum mismatch: {asset['repo_file']}")
    return problems


def scan_for_sealed(root: Path | None = None) -> list[str]:
    """Return every path under `root` (default `ACIS_HOME`) whose name matches a sealed pattern."""
    base = root if root is not None else acis_home()
    if not base.exists():
        return []
    hits: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        if is_sealed_file(rel) or is_sealed_file(path.name):
            hits.append(rel)
    return sorted(hits)


def assert_seal(root: Path | None = None) -> None:
    """G0.6: the dev environment must hold no held-out label file (INV-8)."""
    hits = scan_for_sealed(root)
    if hits:
        raise SealedDataAccess(
            "held-out label files were found inside the dev environment; delete them and re-run `make fetch`",
            files=hits[:10],
        )


def fetch_sealed(pin: DatasetPin = APPS) -> list[FetchedAsset]:
    """OWNER ONLY. Download the held-out labels into `~/.acis-sealed/hf` (never into `ACIS_HOME`)."""
    if os.environ.get(SEALED_OPT_IN) != "1":
        raise SealedDataAccess(
            f"refusing to fetch held-out labels without an explicit opt-in; the owner runs "
            f"`{SEALED_OPT_IN}=1 uv run acis fetch --sealed` in their own terminal (D19)"
        )
    files = [f for f in list_repo_files(pin) if is_sealed_file(f)]
    if not files:
        raise InvalidInput("no sealed files found at the pinned revision", repo=pin.repo)
    hub = _hub(online=True)
    target = sealed_root() / "hf"
    target.mkdir(parents=True, exist_ok=True)
    assets: list[FetchedAsset] = []
    for name in files:
        local = hub.hf_hub_download(
            repo_id=pin.repo, filename=name, repo_type="dataset", revision=pin.revision, cache_dir=str(target)
        )
        assets.append(
            FetchedAsset(
                repo_file=name, local_path=str(local), sha256=sha256_file(local), n_bytes=Path(local).stat().st_size
            )
        )
    (sealed_root() / "SEALED_MANIFEST.json").write_text(
        json.dumps({"dataset": asdict(pin), "assets": [asdict(a) for a in assets]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return assets


def local_asset(repo_file: str) -> Path:
    """Path of a fetched, allow-listed asset. Raises for anything sealed or not fetched."""
    if not is_dev_allowed(repo_file):
        raise SealedDataAccess(f"{repo_file!r} is not on the dev allow-list (INV-8)")
    for asset in load_manifest().get("assets", []):
        if asset["repo_file"] == repo_file:
            return Path(asset["local_path"])
    raise InvalidInput(f"asset not fetched: {repo_file}")


def local_assets_matching(suffix: str) -> list[Path]:
    return [
        Path(a["local_path"])
        for a in load_manifest().get("assets", [])
        if a["repo_file"].startswith(suffix) and is_dev_allowed(a["repo_file"])
    ]


__all__ = [
    "MANIFEST_NAME",
    "SEALED_OPT_IN",
    "FetchedAsset",
    "asset_root",
    "assert_seal",
    "fetch_dev",
    "fetch_sealed",
    "list_repo_files",
    "load_manifest",
    "local_asset",
    "local_assets_matching",
    "manifest_path",
    "scan_for_sealed",
    "verify_manifest",
    "write_manifest",
]
