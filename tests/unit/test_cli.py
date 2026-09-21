"""CLI surface (src/acis/cli/README.md rows L-1…L-6).

Replaces the Phase-0 seed smoke test: `acis doctor` and `acis fetch` are built, so asserting that every command
exits 2 would now be asserting the opposite of the contract.
"""

from __future__ import annotations

import json

import pytest

import acis
from acis.cli import NOT_YET, build_parser, main


def test_version_exposed():
    assert acis.__version__


@pytest.mark.parametrize("command", sorted(NOT_YET))
def test_unbuilt_commands_exit_2_naming_their_phase(command, capsys):
    """L-2: a command that belongs to a later phase says so instead of pretending to work."""
    assert main([command]) == 2
    err = capsys.readouterr().err
    assert NOT_YET[command] in err and "docs/spec/07" in err


def test_built_commands_are_not_in_the_not_yet_table():
    assert {"doctor", "fetch", "audit", "verify", "report", "eval"}.isdisjoint(NOT_YET)


def test_parser_exposes_the_documented_command_set():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]  # noqa: SLF001 — argparse has no public accessor
    commands = set(actions[0].choices)
    assert {"doctor", "fetch", "audit", "verify", "report", "eval", "serve", "train"} <= commands


def test_eval_subcommands_are_declared():
    parser = build_parser()
    args = parser.parse_args(["eval", "verify-submission", "--run-dir", "/tmp/x"])
    assert args.command == "eval" and args.eval_command == "verify-submission"


def test_typed_errors_are_reported_without_a_traceback(tmp_path, capsys):
    """L-6: an `AcisError` prints `code: message` and exits 1."""
    code = main(["eval", "verify-submission", "--run-dir", str(tmp_path / "missing")])
    assert code == 1  # the report fails, it does not raise
    out = capsys.readouterr().out
    assert "run directory exists" in out and "FAIL" in out


def test_verify_submission_json_output(tmp_path, capsys):
    main(["eval", "verify-submission", "--run-dir", str(tmp_path / "missing"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "FAIL" and payload["checks"][0]["name"] == "run directory exists"
