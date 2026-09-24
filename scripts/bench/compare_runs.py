#!/usr/bin/env python
"""Paired comparison between two measured systems (gate rule, docs/spec/03 §7).

Two deltas against a shared baseline are not a comparison of the two systems, however close their intervals
look: the queries have to be paired. This reads the per-query vectors two `ltr_build` reports carry and runs the
bootstrap between them directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acis.eval.bootstrap import paired_bootstrap


def load(path: str, system: str) -> dict[str, float]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in report["rows"]:
        if row["name"].startswith(system):
            vector = row.get("per_query_ndcg_at_10")
            if not vector:
                raise SystemExit(f"{path} carries no per-query vector for {system!r}; re-run it")
            return {k: float(v) for k, v in vector.items()}
    raise SystemExit(f"{path} has no system starting with {system!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="paired bootstrap between two measured systems")
    parser.add_argument("a", help="report whose system is the candidate")
    parser.add_argument("b", help="report whose system is the incumbent")
    parser.add_argument("--system", default="LTR", help="which row to compare in both reports")
    args = parser.parse_args(argv)

    candidate, incumbent = load(args.a, args.system), load(args.b, args.system)
    result = paired_bootstrap(candidate, incumbent)
    print(f"candidate : {args.a}  mean {result.mean_a:.3f} pt")
    print(f"incumbent : {args.b}  mean {result.mean_b:.3f} pt")
    print(f"delta     : {result.delta:+.3f} pt   CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}]   n={result.n}")
    print(f"gate      : {'PASS' if result.passes() else 'no'}  (needs >= +0.5 pt and a CI lower bound above zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
