"""`acis.engine` — the single retrieval engine behind the CLI, the API, the demo and the mteb adapter.

The public surface is frozen at the end of Phase 1 (`protocol.SearchEngine`); Track B builds against it.
"""

from __future__ import annotations

from acis.engine.core import AcisEngine, SnapshotData
from acis.engine.protocol import SearchEngine

__all__ = ["AcisEngine", "SearchEngine", "SnapshotData"]
