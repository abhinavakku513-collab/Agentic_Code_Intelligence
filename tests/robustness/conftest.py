"""Make the robustness helpers importable without each test module mutating `sys.path` at import time.

`perturb.py` sits beside these tests rather than in `src/`, because it is a property library for the suite, not
part of the engine. A single conftest entry keeps that convenient without leaving the directory on `sys.path` for
the rest of the session, where a top-level `perturb` would shadow any same-named import.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
