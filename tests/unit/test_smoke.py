from __future__ import annotations

import acis
from acis.cli import COMMANDS, main


def test_version_exposed():
    assert acis.__version__


def test_cli_stub_exits_2_until_built():
    assert main(["doctor"]) == 2
    assert {"doctor", "fetch", "eval", "train", "report"} <= set(COMMANDS)
