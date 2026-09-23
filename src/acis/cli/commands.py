"""CLI command implementations (docs/spec/06 §3). Thin: every command calls the same library the tests call."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from acis.core.errors import InvalidInput
from acis.core.paths import acis_root
from acis.obs.log import configure


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True, default=str))


# -- Phase 0 ---------------------------------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    from acis.cli.doctor import collect, write_report

    report = collect(with_gemm=not args.no_gemm)
    path = write_report(report)
    if args.json:
        _emit(report, True)
    else:
        hw, env = report["hardware"], report["environment"]
        print(f"tier         : {report['tier']}  ({hw['physical_cores']} physical cores, {hw['ram_gb']} GB RAM)")
        print(f"cpu          : {hw['cpu']}")
        print(f"isa          : {', '.join(hw['isa']) or 'unknown'}")
        if "throughput" in report:
            print(f"gemm fp32    : {report['throughput']['gemm_fp32_gflops_1024']} GFLOP/s (1024³)")
        print(f"gpu          : {report['gpu']}")
        print(f"acis_home    : {env['acis_home']}")
        print(f"seal         : {'clean' if env['seal_ok'] else 'LEAKED: ' + str(env['sealed_files_in_dev_env'])}")
        print(f"dataset      : {env['dataset_manifest'] or 'not fetched (run `make fetch`)'}")
        print(f"written      : {path}")
    return 0 if report["environment"]["seal_ok"] else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    from acis.appsdata import fetch

    if args.verify:
        problems = fetch.verify_manifest()
        fetch.assert_seal()
        print("assets: " + ("OK" if not problems else "\n  ".join(["PROBLEMS"] + problems)))
        return 0 if not problems else 1
    if args.sealed:
        assets = fetch.fetch_sealed()
        print(f"fetched {len(assets)} sealed asset(s) into the sealed area (owner-only path)")
        return 0
    assets = fetch.fetch_dev(force=args.force)
    for a in assets:
        print(f"{a.sha256[:12]}  {a.repo_file}  ({a.n_bytes} bytes)")
    print(f"manifest: {fetch.manifest_path()}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from acis.appsdata import apps

    report = apps.dataset_audit()
    out = Path(args.out) if args.out else acis_root() / "runs" / "dataset_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset", "ids", "labels", "duplicates")}, indent=2, sort_keys=True))
    print(f"written: {out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Integrity checks that need no run directory: seal, assets, split lock, ledger chain."""
    from acis.appsdata import fetch
    from acis.eval import ledger, splits

    results: dict[str, Any] = {}
    results["sealed_files_in_dev_env"] = fetch.scan_for_sealed()
    try:
        results["assets"] = fetch.verify_manifest() or "OK"
    except InvalidInput as exc:
        results["assets"] = f"SKIP: {exc.message}"
    try:
        results["split_lock"] = splits.verify_lock() or "OK"
    except InvalidInput as exc:
        results["split_lock"] = f"SKIP: {exc.message}"
    results["ledger_chain"] = ledger.verify_chain() or "OK"
    results["test_touches_used"] = ledger.test_touches_used()

    ok = not results["sealed_files_in_dev_env"] and all(
        v == "OK" or isinstance(v, (int, str)) for k, v in results.items() if k != "sealed_files_in_dev_env"
    )
    _emit(results, args.json)
    return 0 if ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    from acis.eval import ledger

    rows = ledger.read_rows()
    lines = ["| run_id | kind | rung | n_queries | ndcg@10 | mrr@10 |", "|---|---|---|---|---|---|"]
    for row in rows:
        m = row.get("metrics", {})
        lines.append(
            f"| {row.run_id} | {row.get('kind')} | {row.get('rung', '-')} | {row.get('n_queries', '-')} | "
            f"{m.get('ndcg_at_10', float('nan')):.4f} | {m.get('mrr_at_10', float('nan')):.4f} |"
        )
    table = "\n".join(lines)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"written: {out}")
    else:
        print(table)
    return 0


# -- Phase 1: evaluation ------------------------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    handler = {
        "dev": _eval_dev,
        "ladder": _eval_ladder,
        "parity": _eval_parity,
        "gate": _eval_gate,
        "robustness": _eval_robustness,
        "official": _eval_official,
        "verify-submission": _eval_verify_submission,
        "repro": _eval_repro,
        "splits": _eval_splits,
    }[args.eval_command]
    return handler(args)


def _consume_dev_h_touch(args: argparse.Namespace, guard: Any) -> None:
    """DEV-H (F4) is a once-per-milestone confirmation, not a decision set (docs/spec/03 §5).

    Without a counter, `acis eval dev --fold 4` could be run repeatedly during tuning and DEV-H would quietly
    become the decision set, which is exactly the over-fitting the split was designed to prevent. Every use is
    recorded and a second use inside the same milestone is refused.
    """
    milestone = str(getattr(args, "milestone", "") or os.environ.get("ACIS_MILESTONE", "") or "unlabelled")
    already = guard.dev_h_touches_for(milestone)
    if already and os.environ.get("ACIS_ALLOW_REPEAT_DEV_H") != "1":
        raise InvalidInput(
            "DEV-H has already been used for this milestone; it is a confirmation, not a decision set",
            milestone=milestone,
            touches=already,
            hint="decide on all 5,000 dev queries (or K-fold OOF), or declare a new milestone",
        )
    guard.record_dev_h_touch(reason=f"acis eval dev --system {args.system}", milestone=milestone)


def _eval_dev(args: argparse.Namespace) -> int:
    from acis.appsdata import apps
    from acis.eval import guard, ladder, splits

    if not apps.is_available():
        raise InvalidInput("dataset assets are missing: run `make fetch` first")

    query_ids = None
    if args.fold >= 0:
        if args.fold not in splits.fold_members():
            raise InvalidInput(f"fold {args.fold} does not exist", folds=splits.DEFAULT_FOLDS)
        query_ids = splits.fold_members()[args.fold]
        if args.fold == splits.DEV_H_FOLD:
            _consume_dev_h_touch(args, guard)

    result = ladder.run_rung(args.system, query_ids=query_ids, limit=args.limit)
    run_id = ladder.record(result, fold=args.fold if args.fold >= 0 else None, extra={"fold": args.fold})
    print(f"rung={result.rung} queries={result.n_queries} docs={result.n_docs} seconds={result.seconds:.1f}")
    print(f"ndcg@10={result.metrics['ndcg_at_10']:.4f}  mrr@10={result.metrics['mrr_at_10']:.4f}  ")
    print(f"recall@100={result.metrics['recall_at_100']:.4f}  ledger={run_id}")
    if args.out:
        from acis.eval.runfile import write_run

        print(json.dumps(write_run(result.run, args.out), indent=2))
    return 0


def _eval_ladder(args: argparse.Namespace) -> int:
    from acis.eval import ladder

    if args.list:
        print("\n".join(ladder.RUNG_NAMES))
        return 0
    for rung in [r.strip() for r in args.rungs.split(",") if r.strip()]:
        result = ladder.run_rung(rung, limit=args.limit)
        run_id = ladder.record(result)
        print(
            f"{rung:12} ndcg@10={result.metrics['ndcg_at_10']:.4f} mrr@10={result.metrics['mrr_at_10']:.4f} "
            f"n={result.n_queries} {result.seconds:.1f}s ledger={run_id}"
        )
    return 0


def _eval_parity(args: argparse.Namespace) -> int:
    from acis.eval import ladder

    report = ladder.run_parity(limit=args.limit, top_k=args.top_k)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["parity_top10_identical"] >= 0.95 else 1


def _eval_gate(args: argparse.Namespace) -> int:
    print(
        f"gate {args.gate}: the gate procedures are declared in configs/gates/ and run from Phase 2 onwards "
        "(docs/spec/03 §7). Phase 1 ships the statistics they use: acis.eval.bootstrap.",
        file=sys.stderr,
    )
    return 2


def _eval_robustness(args: argparse.Namespace) -> int:
    print(
        "acis eval robustness measures perturbation families against the frozen base (G-OOD) and needs the dense "
        "encoder from Phase 2. The label-free property suite runs today: `make robustness`.",
        file=sys.stderr,
    )
    return 2


def _eval_official(args: argparse.Namespace) -> int:
    from acis.eval.official import run_official

    result = run_official(
        rc=args.rc,
        mode=args.mode,
        config_path=args.config,
        out=args.out or None,
        cold=args.cold,
        cache_verify=args.cache_verify,  # a cache verification books no held-out touch
        reproduce=args.reproduce,  # nor does a judge's reproduction
        force=args.force,
    )
    ndcg = result.metrics.get("ndcg_at_10", float("nan"))
    print(f"{result.rc} mode={result.mode} primary={result.primary_mode} ndcg@10={ndcg:.4f}")
    print(f"evaluation_time={result.evaluation_time:.1f}s (per mode: {dict(result.per_mode_seconds)})")
    print(f"run_dir={result.run_dir}  ledger={result.run_id or '(cache-verify: no ledger row)'}")
    print(f"verify-submission: {'PASS' if result.verified else 'FAIL — read verify_report.txt'}")
    return 0 if result.verified else 1


def _eval_verify_submission(args: argparse.Namespace) -> int:
    from acis.appsdata import apps
    from acis.eval.verify import verify_submission

    qrels = None
    manifest = Path(args.run_dir) / "manifest.json"
    if manifest.is_file():
        split = json.loads(manifest.read_text(encoding="utf-8")).get("split")
        if split == "train" and apps.is_available():
            qrels = apps.load_qrels()
    report = verify_submission(args.run_dir, qrels=qrels)
    _emit(report.to_dict() if args.json else report.render(), args.json)
    return 0 if report.passed else 1


def _eval_repro(args: argparse.Namespace) -> int:
    from acis.eval import ledger

    row = ledger.find(args.run_id)
    problems = ledger.verify_chain()
    print(json.dumps({"row": dict(row.payload), "chain_problems": problems}, indent=2, sort_keys=True, default=str))
    return 0 if not problems else 1


def _eval_splits(args: argparse.Namespace) -> int:
    from acis.eval import splits

    if args.write:
        path = splits.write_lock()
        print(f"split lock: {path}")
        print(json.dumps(json.loads(path.read_text(encoding="utf-8"))["sets"], indent=2, sort_keys=True))
        return 0
    problems = splits.verify_lock()
    print("split lock: " + ("OK" if not problems else "\n  ".join(["PROBLEMS"] + problems)))
    return 0 if not problems else 1


configure()

__all__ = [
    "cmd_audit",
    "cmd_doctor",
    "cmd_eval",
    "cmd_fetch",
    "cmd_report",
    "cmd_verify",
]
