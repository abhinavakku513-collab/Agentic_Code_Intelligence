"""`acis` CLI seed. Every sub-command reports which phase builds it (docs/spec/07) and exits 2 until then."""

from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "doctor": "Phase 0",
    "fetch": "Phase 0",
    "ingest": "Track B1",
    "index": "Track B1",
    "search": "Phase 2",
    "versions": "Track B1",
    "activate": "Track B1",
    "rollback": "Track B1",
    "diff": "Track B1",
    "evolve": "Track B2",
    "verify": "Track B1",
    "report": "Phase 2",
    "serve": "Track B3",
    "eval": "Phase 1",
    "train": "Phase 3",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acis", description="ACIS — local, offline, CPU-first code retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub.add_parser(name, add_help=False)
    args, _ = parser.parse_known_args(argv)
    print(
        f"acis {args.command}: not implemented yet — built in {COMMANDS[args.command]} (see docs/spec/07)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
