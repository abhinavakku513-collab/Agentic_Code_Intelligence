"""`acis.appsdata` — dataset acquisition and dev-split loaders (the package map's `data` role, docs/spec/06 §0).

Named `appsdata` rather than `data` because the repository's protected `.claude/settings.json` denies every write whose
path matches `data/**`, which also matches `src/acis/data/**` (ADR-0004). The module boundary and contract are
unchanged: APPS loaders (dev split only), fold assignment and the dataset audit.
"""

from __future__ import annotations

from acis.appsdata.apps import (
    corpus_ids,
    dataset_audit,
    dev_query_ids,
    fold_assignments,
    is_available,
    load_corpus,
    load_documents,
    load_qrels,
    load_queries,
)
from acis.appsdata.fetch import assert_seal, fetch_dev, scan_for_sealed, verify_manifest
from acis.appsdata.sources import APPS, DEV_SPLIT, is_dev_allowed, is_sealed_file

__all__ = [
    "APPS",
    "DEV_SPLIT",
    "assert_seal",
    "corpus_ids",
    "dataset_audit",
    "dev_query_ids",
    "fetch_dev",
    "fold_assignments",
    "is_available",
    "is_dev_allowed",
    "is_sealed_file",
    "load_corpus",
    "load_documents",
    "load_qrels",
    "load_queries",
    "scan_for_sealed",
    "verify_manifest",
]
