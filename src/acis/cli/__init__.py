"""`acis` CLI (docs/spec/06 §3). One engine, thin surfaces; commands not yet built report the phase that builds them.

Config and data paths resolve from the repo root (D16), so a judge may run `acis` from anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from acis.core.errors import AcisError

NOT_YET = {
    "ingest": "Track B1",
    "index": "Track B1",
    "search": "Phase 2",
    "versions": "Track B1",
    "activate": "Track B1",
    "rollback": "Track B1",
    "diff": "Track B1",
    "evolve": "Track B2",
    "serve": "Track B3",
    "train": "Phase 3",
}


def _add_eval(sub: argparse._SubParsersAction) -> None:
    ev = sub.add_parser("eval", help="evaluation harness (dev, ladder, gates, robustness, official, verification)")
    evs = ev.add_subparsers(dest="eval_command", required=True)

    dev = evs.add_parser("dev", help="run a dev evaluation on the TRAIN split (never the held-out labels)")
    dev.add_argument("--config", default="configs/dev.yaml")
    dev.add_argument("--system", default="bm25", help="ladder rung to run (see `acis eval ladder --list`)")
    dev.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries (smoke runs)")
    dev.add_argument("--fold", type=int, default=-1, help="restrict to one fold (-1 = all 5,000 TRAIN queries)")
    dev.add_argument("--milestone", default="", help="required when --fold names DEV-H: its touch is counted once")
    dev.add_argument("--out", default="")

    ladder = evs.add_parser("ladder", help="run ladder rungs B0/B1 and record them in the ledger")
    ladder.add_argument("--config", default="configs/dev.yaml")
    ladder.add_argument("--rungs", default="B0,B1")
    ladder.add_argument("--limit", type=int, default=0)
    ladder.add_argument("--list", action="store_true")

    parity = evs.add_parser("parity", help="B0 vs B1: our BM25 against mteb/baseline-bm25s, recorded in the ledger")
    parity.add_argument("--limit", type=int, default=0, help="first N dev queries (0 = all 5,000)")
    parity.add_argument("--top-k", type=int, default=1000)

    gate = evs.add_parser("gate", help="run a pre-declared gate procedure")
    gate.add_argument("--gate", required=True)
    gate.add_argument("--config", default="configs/dev.yaml")

    rob = evs.add_parser("robustness", help="perturbation families vs the frozen base (G-OOD)")
    rob.add_argument("--config", default="configs/dev.yaml")

    off = evs.add_parser("official", help="OWNER-RUN: the official cold run against the held-out split")
    off.add_argument("--config", default="configs/official.yaml")
    off.add_argument("--rc", default="RC0")
    off.add_argument("--mode", default="AB", choices=["A", "B", "AB"])
    off.add_argument("--cold", action="store_true", help="record this run as a cold pass in the ledger (D17)")
    off.add_argument(
        "--strict",
        action="store_true",
        help="documents intent; an official run is strict regardless and refuses a non-strict config",
    )
    off.add_argument(
        "--cache-verify",
        action="store_true",
        help="reproduce rankings from shipped caches: books no held-out touch and writes no ledger row",
    )
    off.add_argument(
        "--reproduce",
        action="store_true",
        help="judge quick start: a cold run that books no touch and enforces no release-candidate preconditions",
    )
    off.add_argument("--force", action="store_true", help="overwrite a non-empty run directory")
    off.add_argument("--out", default="")

    ver = evs.add_parser("verify-submission", help="check a run directory against the submission contract")
    ver.add_argument("--run-dir", required=True)
    ver.add_argument("--json", action="store_true", help="print the report as JSON")

    rep = evs.add_parser("repro", help="re-derive hashes for a ledger row and compare")
    rep.add_argument("run_id")

    splits = evs.add_parser("splits", help="build or verify configs/splits.lock.json")
    splits.add_argument("--write", action="store_true")
    splits.add_argument("--verify", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acis", description="ACIS — local, offline, CPU-first code retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="hardware profile, tier and environment checks -> runs/hardware.json")
    doctor.add_argument("--no-gemm", action="store_true", help="skip the GEMM throughput measurement")
    doctor.add_argument("--json", action="store_true")

    fetch = sub.add_parser("fetch", help="download the allow-listed dataset assets (the only network path)")
    fetch.add_argument(
        "--sealed", action="store_true", help="OWNER ONLY: fetch the held-out labels into ~/.acis-sealed"
    )
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--verify", action="store_true", help="re-hash the fetched assets and check the seal")

    audit = sub.add_parser("audit", help="dataset audit (G0.2) -> runs/dataset_audit.json")
    audit.add_argument("--out", default="")

    verify = sub.add_parser("verify", help="integrity checks: seal, split lock, dataset manifest, ledger chain")
    verify.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="ledger-backed evidence table")
    report.add_argument("--claims", action="store_true")
    report.add_argument("--out", default="")

    _add_eval(sub)
    for name, phase in NOT_YET.items():
        p = sub.add_parser(name, help=f"built in {phase}")
        p.add_argument("args", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return dispatch(args)
    except AcisError as exc:
        print(f"acis: {exc.code}: {exc.message}", file=sys.stderr)
        if exc.context:
            print(json.dumps(exc.context, indent=2, default=str), file=sys.stderr)
        return 1


def dispatch(args: argparse.Namespace) -> int:
    from acis.cli import commands  # noqa: PLC0415 — keeps `acis --help` free of heavy imports

    if args.command in NOT_YET:
        print(
            f"acis {args.command}: not implemented yet — built in {NOT_YET[args.command]} (see docs/spec/07)",
            file=sys.stderr,
        )
        return 2
    handler = getattr(commands, f"cmd_{args.command.replace('-', '_')}", None)
    if handler is None:
        print(f"acis: unknown command {args.command}", file=sys.stderr)
        return 2
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
