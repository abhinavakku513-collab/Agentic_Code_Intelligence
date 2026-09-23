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

    if args.models:
        return _fetch_models(args)
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


def _fetch_models(args: argparse.Namespace) -> int:
    """Fetch (or verify) carded model weights — the other half of G0.4.

    Kept apart from the dataset path on purpose: a model is a different supply chain, with a different failure
    mode (a pickle that executes on load) and a different licence question (D4).
    """
    from acis.embed import modelfetch

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    if not keys:
        raise InvalidInput("--models takes one or more card keys, e.g. --models qwen3-embedding-0.6b")

    if args.verify:
        failed = False
        for key in keys:
            problems = modelfetch.verify_model(key)
            failed = failed or bool(problems)
            print(f"{key}: " + ("as pinned" if not problems else "\n  ".join(["PROBLEMS"] + problems)))
        return 1 if failed else 0

    for key in keys:
        fetched = modelfetch.fetch_model(key, reference=args.reference, force=args.force)
        label = "  (reference only: never shipped)" if fetched.reference_only else ""
        print(f"{fetched.repo}  {fetched.commit[:12]}  {len(fetched.files)} files  {fetched.total_mb} MB{label}")
        print(f"  -> {fetched.model_dir}")
        if args.pin:
            print(f"  pinned: {modelfetch.pin_card(fetched)}")
        else:
            print("  not pinned: re-run with --pin to write the commit and file hashes into the card (G0.4)")
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


# -- Track B1: repositories and versions --------------------------------------------------------------------------
def _engine_for_cli(config_path: str):
    """One engine, built the way every other surface builds one (the factory, the config, no special cases)."""
    from acis.core.config import load_frozen_config
    from acis.embed.factory import build_encoder
    from acis.engine import AcisEngine

    config = load_frozen_config(config_path)
    return AcisEngine.from_config(config, encoder=build_encoder(config))


def cmd_ingest(args: argparse.Namespace) -> int:
    """Read a source and build every version it contains."""
    from acis.core.types import SourceSpec

    options: dict[str, Any] = {}
    if args.ext:
        options["extensions"] = [e if e.startswith(".") else f".{e}" for e in args.ext.split(",") if e]
    if args.rev:
        options["rev"] = args.rev

    engine = _engine_for_cli(args.config)
    spec = SourceSpec(kind=args.kind, location=args.location, options=options)
    handle = engine.ingest(spec, repo_id=args.repo)
    for report in engine.build_reports(args.repo):
        print(
            f"{report.snapshot.version_id:<14} {report.snapshot.snapshot_id}  {report.units_total:>5} units  "
            f"{report.units_new:>5} new  {report.units_reused:>5} reused  {report.seconds:.2f}s"
        )
    print(f"\n{handle.state}: {handle.detail}")
    return 0


def cmd_versions(args: argparse.Namespace) -> int:
    from acis.store import snapshots

    engine = _engine_for_cli(args.config)
    active = snapshots.active_snapshot_id(args.repo)
    rows = engine.versions(args.repo)
    if args.json:
        _emit({"repo": args.repo, "active": active, "versions": rows}, True)
        return 0
    print(f"{'version':<16} {'snapshot':<20} {'units':>6}  state")
    for row in rows:
        marker = " *" if row["snapshot_id"] == active else "  "
        snapshot = str(row["snapshot_id"] or "-")
        print(f"{row['label']:<16} {snapshot:<20} {row.get('n_units', '-'):>6}  {row['state']}{marker}")
    print("\n* = active")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    engine = _engine_for_cli(args.config)
    comparison = engine.compare_versions(args.repo, args.from_version, args.to_version, query=args.query or None)
    if args.json:
        _emit(
            {
                "repo": comparison.repo_id,
                "a": comparison.a,
                "b": comparison.b,
                "added": list(comparison.added),
                "removed": list(comparison.removed),
                "changed": list(comparison.changed),
                "query_effect": {
                    k: [list(x) for x in v] if isinstance(v, list) else v
                    for k, v in dict(comparison.query_effect).items()
                },
            },
            True,
        )
        return 0
    for label, items in (("added", comparison.added), ("removed", comparison.removed), ("changed", comparison.changed)):
        for key in items:
            print(f"{label:<8} {key}")
    if not (comparison.added or comparison.removed or comparison.changed):
        print("no unit-level differences")
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    engine = _engine_for_cli(args.config)
    print(f"active: {engine.activate(args.repo, args.version)}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    engine = _engine_for_cli(args.config)
    print(f"active: {engine.rollback(args.repo)}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    engine = _engine_for_cli(args.config)
    report = engine.index(args.repo, mode=args.mode)
    print(f"built {report.snapshot.version_id}: {report.units_total} units, {report.units_new} new")
    return 0


# -- Phase 2: the search surface ----------------------------------------------------------------------------------
def cmd_search(args: argparse.Namespace) -> int:
    """Rank the corpus for a free-text query and show the evidence.

    The snapshot is built in this process and thrown away with it. Persisting one is Track B1's job (D11), so a
    real encoder pays the full embedding cost per invocation — which is exactly why `acis eval gate` and the demo
    runbook build one snapshot and ask many questions of it.
    """
    import time

    from acis.appsdata import apps
    from acis.core.config import load_frozen_config
    from acis.core.types import SearchRequest
    from acis.embed.factory import build_encoder
    from acis.engine import AcisEngine

    config = load_frozen_config(args.config)
    encoder = build_encoder(config)
    engine = AcisEngine.from_config(config, encoder=encoder)

    started = time.perf_counter()
    if args.repo:
        # A versioned repository: the snapshot is already on disk, so this opens one rather than building one.
        data = engine.open_version(args.repo, args.version)
        snapshot = data.snapshot
    else:
        if not apps.is_available():
            raise InvalidInput("dataset assets are missing: run `make fetch` first")
        snapshot = engine.build_snapshot(apps.load_corpus(), source="cli:search")
    build_seconds = time.perf_counter() - started

    response = engine.search(
        SearchRequest(
            query=args.query,
            repo_id=args.repo or "-",
            version=args.version,
            top_k=args.top_k,
            mode=args.mode,
            explain=args.explain,
            diagnostics=True,
        )
    )
    if args.json:
        _emit(
            {
                "query": args.query,
                "snapshot": {
                    "id": snapshot.snapshot_id,
                    "n_units": snapshot.n_units,
                    "version": snapshot.version_id,
                },
                "encoder": {"name": encoder.name, "submission_capable": encoder.submission_capable},
                "route": response.route,
                "no_strong_match": response.no_strong_match,
                "confidence": response.confidence,
                "timings_ms": dict(response.timings_ms),
                "degradations": list(response.degradations),
                "results": [
                    {
                        "rank": h.rank,
                        "score": round(h.score, 6),
                        "unit_id": h.unit.unit_id,
                        "key": h.unit.key,
                        "body_hash": h.unit.body_hash,
                        "n_bytes": h.unit.n_bytes,
                        "source": h.source,
                    }
                    for h in response.results
                ],
            },
            True,
        )
        return 0

    stand_in = "" if encoder.submission_capable else "  (stand-in: not submission-capable)"
    print(f"encoder  : {encoder.name}{stand_in}")
    opened = "opened" if args.repo else "built"
    version = f"  version {snapshot.version_id}" if args.repo else ""
    print(f"snapshot : {snapshot.snapshot_id}{version}  {snapshot.n_units} units, {opened} in {build_seconds:.1f}s")
    timings = "  ".join(f"{k}={v:.1f}ms" for k, v in sorted(response.timings_ms.items()))
    print(f"route    : {response.route}  confidence={response.confidence}  {timings}")
    if response.degradations:
        print(f"degraded : {', '.join(response.degradations)}")
    print()
    for hit in response.results:
        first = next((line for line in hit.source.splitlines() if line.strip()), "")
        # The key is what a person recognises ("sort.py"); the body hash is what makes the evidence checkable.
        label = (hit.unit.key or hit.unit.unit_id)[:24]
        print(f"{hit.rank:>3}  {hit.score:8.4f}  {label:<24} {hit.unit.body_hash[:12]}  {first[:76]}")
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


def _dev_h_milestone(args: argparse.Namespace, guard: Any) -> str:
    """Check that DEV-H may be used, and return the milestone it is being spent on.

    DEV-H (F4) is a once-per-milestone confirmation, not a decision set (docs/spec/03 §5). Without a counter,
    `acis eval dev --fold 4` could be repeated during tuning until DEV-H quietly became the decision set — which
    is the over-fitting the split exists to prevent. The milestone must be named: defaulting it would let the
    first unlabelled run block every later one while any invented name granted a fresh touch.
    """
    milestone = str(getattr(args, "milestone", "") or os.environ.get("ACIS_MILESTONE", "")).strip()
    if not milestone:
        raise InvalidInput(
            "using DEV-H requires naming the milestone it confirms (--milestone, or ACIS_MILESTONE)",
            fold=args.fold,
            hint="DEV-H is a once-per-milestone confirmation; decisions are made on all 5,000 dev queries",
        )
    already = guard.dev_h_touches_for(milestone)
    if already and os.environ.get("ACIS_ALLOW_REPEAT_DEV_H") != "1":
        raise InvalidInput(
            "DEV-H has already been used for this milestone; it is a confirmation, not a decision set",
            milestone=milestone,
            touches=already,
            hint="decide on all 5,000 dev queries (or K-fold OOF), or declare a new milestone",
        )
    return milestone


def _record_dev_h_touch(args: argparse.Namespace, guard: Any, milestone: str, run_id: str) -> None:
    """Book the touch **after** the run succeeded, and chain it to the ledger.

    Booking before would let a typo in `--system` burn the milestone's only confirmation, and a counter that lives
    only in a JSON file outside git leaves no auditable trace — `rm` would erase it. The ledger row makes the use
    as traceable as every other number.
    """
    from acis.eval import ledger  # noqa: PLC0415

    guard.record_dev_h_touch(reason=f"acis eval dev --system {args.system}", milestone=milestone)
    ledger.append(
        ledger.LedgerRowBuilder(kind="dev")
        .with_fields(
            rung=f"dev_h_touch:{args.system}",
            dataset="dev",
            decision_set="dev_h",
            milestone=milestone,
            confirms_run=run_id,
            notes="DEV-H confirmation touch (docs/spec/03 §5); not a decision set",
        )
        .build()
    )


def _eval_dev(args: argparse.Namespace) -> int:
    from acis.appsdata import apps
    from acis.eval import guard, ladder, splits

    if not apps.is_available():
        raise InvalidInput("dataset assets are missing: run `make fetch` first")

    query_ids = None
    milestone = ""
    if args.fold >= 0:
        if args.fold not in splits.fold_members():
            raise InvalidInput(f"fold {args.fold} does not exist", folds=splits.DEFAULT_FOLDS)
        query_ids = splits.fold_members()[args.fold]
        if args.fold == splits.DEV_H_FOLD:
            milestone = _dev_h_milestone(args, guard)  # checked before the run, booked after it

    result = ladder.run_rung(args.system, query_ids=query_ids, limit=args.limit)
    run_id = ladder.record(result, fold=args.fold if args.fold >= 0 else None, extra={"fold": args.fold})
    if milestone:
        _record_dev_h_touch(args, guard, milestone, run_id)
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
    if args.gate.upper() in ("G-M", "GM"):
        return _gate_m(args)
    if args.gate.upper() == "G1":
        return _gate_1(args)
    print(
        f"gate {args.gate}: the gate procedures are declared in configs/gates/ and run from Phase 2 onwards "
        "(docs/spec/03 §7). Phase 1 ships the statistics they use: acis.eval.bootstrap.",
        file=sys.stderr,
    )
    return 2


def _gate_m(args: argparse.Namespace) -> int:
    """The encoder bake-off: measure every candidate, apply the D4 rule, print the table.

    The decision is only *written* when asked for (`--record`), and only the full decision set can produce one —
    a smoke run over 50 queries is a look, not a gate (docs/spec/09 §2).
    """
    from acis.embed.registry import available_cards
    from acis.eval.bakeoff import FULL_DEV_POOL, record_decision, render_table, run_gate_m

    keys = [k.strip() for k in args.models.split(",") if k.strip()] or available_cards()
    if not keys:
        raise InvalidInput("no model cards in configs/models/; G0.4 writes them")
    reference = [k.strip() for k in args.reference.split(",") if k.strip()]

    candidates, decision = run_gate_m(keys, limit=args.limit, reference_only=reference)
    print(render_table(candidates, decision))
    print(f"\nwinner={decision.winner}  best={decision.best}  tolerance={decision.tolerance_pts} pt")
    if decision.reference_gap_pts:
        print(f"the best measured candidate is {decision.reference_gap_pts:.2f} pt ahead of the best that can ship")

    measured_on = min((c.n_queries for c in candidates), default=0)
    if args.record:
        if measured_on < FULL_DEV_POOL:
            raise InvalidInput("a gate decision needs all 5,000 dev queries (docs/spec/09 §2)", measured_on=measured_on)
        print(f"wrote {record_decision(decision)}")
    return 0


def _gate_1(args: argparse.Namespace) -> int:
    """The query-representation sweep, for one route: task string x view x truncation length.

    The grid is a product, so it is measured for the route named by `--route` rather than for all of them at once
    — a route is a separate decision (INV-15) and a separate afternoon of compute.
    """
    from acis.appsdata import apps
    from acis.eval.ladder import FULL_DEV_POOL
    from acis.eval.sweep import Cell, default_grid, record_decision, render_table, run_g1

    cells = [Cell(*_parse_cell(c)) for c in args.cells.split(",") if c.strip()] if args.cells else default_grid()
    ids = list(apps.dev_query_ids())
    if args.limit > 0:
        ids = ids[: args.limit]

    measurements, decision = run_g1(args.route, query_ids=ids, grid=cells, baseline=args.baseline)
    print(render_table(measurements, decision))
    print(f"\nwinner={decision.winner}  adopted={decision.adopted}  n={decision.n_queries}")
    if args.record:
        if len(ids) < FULL_DEV_POOL:
            raise InvalidInput("a gate decision needs all 5,000 dev queries (docs/spec/09 §2)", measured_on=len(ids))
        print(f"wrote {record_decision(decision)}")
    return 0


def _parse_cell(text: str) -> tuple[str, str, int]:
    """`T1/V0/1024` -> the three dimensions G1 sweeps."""
    parts = text.strip().split("/")
    if len(parts) != 3:
        raise InvalidInput(f"a cell is task/view/length, e.g. T1/V0/1024 (got {text!r})")
    return parts[0], parts[1], int(parts[2])


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
