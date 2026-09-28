"""`acis` CLI (docs/spec/06 §3). One engine, thin surfaces; commands not yet built report the phase that builds them.

Config and data paths resolve from the repo root (D16), so a judge may run `acis` from anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from acis.core.errors import AcisError

NOT_YET: dict[str, str] = {}


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
    gate.add_argument("--models", default="", help="G-M: comma-separated card keys to measure (default: all cards)")
    gate.add_argument("--reference", default="", help="G-M: card keys measured for reference, never selected")
    gate.add_argument("--limit", type=int, default=0, help="first N dev queries (0 = all 5,000; a gate needs all)")
    gate.add_argument("--route", default="generic", help="G1: the route being decided (INV-15: one at a time)")
    gate.add_argument("--cells", default="", help="G1: comma-separated cells, e.g. T1/V0/1024,T3/V1/512")
    gate.add_argument("--baseline", default="T1/V0/1024", help="G1: the incumbent cell a change has to beat")
    gate.add_argument(
        "--record", action="store_true", help="write the decision to configs/gates/<id>.yaml (written once)"
    )

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
    fetch.add_argument("--models", default="", help="card keys to fetch instead of dataset assets (G0.4)")
    fetch.add_argument("--pin", action="store_true", help="write the commit and file hashes into the model card")
    fetch.add_argument(
        "--reference", action="store_true", help="allow a non-permissive model: measured for reference, never shipped"
    )

    audit = sub.add_parser("audit", help="dataset audit (G0.2) -> runs/dataset_audit.json")
    audit.add_argument("--out", default="")

    verify = sub.add_parser("verify", help="integrity checks: seal, split lock, dataset manifest, ledger chain")
    verify.add_argument("--json", action="store_true")

    ingest = sub.add_parser("ingest", help="read a source and build every version it contains (Track B1)")
    ingest.add_argument("location", help="path to a JSONL file, directory, ZIP or git repository")
    ingest.add_argument("--repo", required=True, help="repository id the versions are recorded under")
    ingest.add_argument("--kind", default="jsonl", choices=["jsonl", "dir", "zip", "git"])
    ingest.add_argument("--ext", default="", help="comma-separated extensions to ingest (default: .py)")
    ingest.add_argument("--rev", default="", help="git only: a single revision to ingest instead of the history")
    ingest.add_argument("--config", default="configs/dev.yaml")

    index = sub.add_parser("index", help="build the versions of a repository that have no snapshot yet")
    index.add_argument("--repo", required=True)
    index.add_argument("--mode", default="eager_heads", choices=["eager_heads", "eager_all", "lazy"])
    index.add_argument("--config", default="configs/dev.yaml")

    versions = sub.add_parser("versions", help="list a repository's versions and which one is active")
    versions.add_argument("--repo", required=True)
    versions.add_argument("--json", action="store_true")
    versions.add_argument("--config", default="configs/dev.yaml")

    diff = sub.add_parser("diff", help="unit-level difference between two versions")
    diff.add_argument("--repo", required=True)
    diff.add_argument("--from", dest="from_version", required=True)
    diff.add_argument("--to", dest="to_version", required=True)
    diff.add_argument("--query", default="", help="also show what this query returns on each side")
    diff.add_argument("--json", action="store_true")
    diff.add_argument("--config", default="configs/dev.yaml")

    activate = sub.add_parser("activate", help="point a repository at one of its versions")
    activate.add_argument("--repo", required=True)
    activate.add_argument("--version", required=True)
    activate.add_argument("--config", default="configs/dev.yaml")

    rollback = sub.add_parser("rollback", help="return a repository to its previous snapshot (instant)")
    rollback.add_argument("--repo", required=True)
    rollback.add_argument("--config", default="configs/dev.yaml")

    search = sub.add_parser("search", help="rank the corpus for a free-text query, with the evidence")
    search.add_argument("--repo", default="", help="search a versioned repository instead of the APPS corpus")
    search.add_argument("--version", default="latest", help="version selector (latest, v2, as_of:…, snapshot:…)")
    search.add_argument("query")
    search.add_argument("--config", default="configs/dev.yaml")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--mode", default="auto", choices=["auto", "dense", "lexical", "hybrid"])
    search.add_argument("--explain", action="store_true")
    search.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the HTTP API and the demo UI (loopback by default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", default="configs/dev.yaml")

    evolve = sub.add_parser("evolve", help="evolution-aware retrieval across every version (Bonus)")
    evolve.add_argument("query")
    evolve.add_argument("--repo", required=True)
    evolve.add_argument("--top-k", type=int, default=10)
    evolve.add_argument("--flat", action="store_true", help="also show the ungrouped list, duplicates and all")
    evolve.add_argument("--json", action="store_true")
    evolve.add_argument("--config", default="configs/dev.yaml")

    demo_idx = sub.add_parser("demo-index", help="prebuilt demo index: export the corpus vectors, or import a pack")
    demo_idx.add_argument("action", choices=["export", "import"])
    demo_idx.add_argument("path", nargs="?", default="dist/acis-demo-index.zip")
    demo_idx.add_argument("--config", default="configs/dev.yaml")
    demo_idx.add_argument("--verify-fraction", type=float, default=0.01, help="share recomputed on import")

    train = sub.add_parser("train", help="Phase 3 GPU hand-off: export the training bundle, or import its results")
    train.add_argument("action", choices=["export", "import"])
    train.add_argument(
        "path", nargs="?", default="dist/train-bundle", help="bundle dir (export) or output dir (import)"
    )
    train.add_argument("--config", default="configs/dev.yaml")
    train.add_argument("--mine-top", type=int, default=50)
    train.add_argument("--bundle", default="dist/train-bundle", help="import: the bundle the run was trained from")

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
