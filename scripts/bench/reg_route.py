"""REG (human-written queries): the served generic fusion vs forcing the learned ranker, through the engine."""

import sys

sys.path.insert(0, "scripts/bench")
import reg_fusion as rf

from acis.eval.bootstrap import paired_bootstrap
from acis.eval.metrics import per_query, score_run
from acis.rank.compose import rank_derived_scores

engine = rf.build_engine("configs/dev.yaml")
for task, split in (("cosqa", "test"), ("csn-python", "test")):
    corpus, splits = rf.load_task(task)
    snap = engine.build_snapshot(corpus, source=f"reg:{task}")
    data = engine.snapshot_data(snap)
    rows = splits[split]
    qrels = {r["id"]: {d: int(s) for d, s in r["relevant"].items()} for r in rows}
    runs = {}
    for mode in ("statement_like", "generic"):
        runs[mode] = {
            r["id"]: rank_derived_scores(
                [d for d, _ in engine._rank_one(data, r["text"], top_k=100, strict=False, route=mode)], 100
            )
            for r in rows
        }
    pl = {m: per_query(qrels, runs[m], "ndcg", 10) for m in runs}
    b = paired_bootstrap(pl["statement_like"], pl["generic"])
    print(
        task,
        split,
        {m: round(100 * score_run(qrels, runs[m])["ndcg_at_10"], 2) for m in runs},
        f"ranker-fusion {b.delta:+.2f} [{b.ci_low:+.2f},{b.ci_high:+.2f}]",
        flush=True,
    )
