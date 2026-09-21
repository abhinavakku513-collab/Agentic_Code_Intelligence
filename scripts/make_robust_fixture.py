#!/usr/bin/env python
"""Generate `tests/fixtures/robust_corpus.jsonl` — the fixed corpus behind `tests/robustness` (docs/spec/10 §8).

The robustness properties are only meaningful on a corpus big and varied enough for a ranking to *move*: on a dozen
toy documents a template-coupled engine is not caught (spec 10 §5 says so explicitly). So the fixture is a
deterministic, length-stratified sample of real documents from the pinned dataset (CoIR-Retrieval/apps, MIT), large
enough to expose a flip and small enough to commit.

Run: `uv run python scripts/make_robust_fixture.py [--n 256]`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acis.appsdata import apps  # noqa: E402
from acis.core.hashing import sha256_text  # noqa: E402

STRATA = 8
DEFAULT_N = 256


def select(n: int) -> list[tuple[str, str]]:
    """Length-stratified, deterministic selection: same dataset revision in, same fixture out."""
    docs = [d for d in apps.load_documents() if d.text.strip()]
    ordered = sorted(docs, key=lambda d: (len(d.text), d.doc_id))
    per_stratum = max(1, n // STRATA)
    picked: list[tuple[str, str]] = []
    size = max(1, len(ordered) // STRATA)
    for s in range(STRATA):
        chunk = ordered[s * size : (s + 1) * size] if s < STRATA - 1 else ordered[s * size :]
        if not chunk:
            continue
        # Deterministic spread inside the stratum, by content hash — no RNG, no dataset order dependence.
        chunk = sorted(chunk, key=lambda d: sha256_text(d.text))
        picked.extend((d.doc_id, d.text) for d in chunk[:per_stratum])
    return sorted(picked, key=lambda kv: kv[0])[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "robust_corpus.jsonl"))
    args = parser.parse_args()

    if not apps.is_available():
        print("dataset assets are missing: run `make fetch` first", file=sys.stderr)
        return 2

    rows = select(args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for doc_id, text in rows:
            fh.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} documents to {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
