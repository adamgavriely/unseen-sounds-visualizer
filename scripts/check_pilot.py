"""Decide, without a human reading the log, whether the pilot is sound enough to
scale up to the full run.

The plan says: do not spend eight GPU-hours on a judge that cannot discriminate. That
instruction assumes somebody is reading the pilot records. In an unattended chain
nobody is, so the checks that a person would make by eye are made here instead, and a
failure stops the chain rather than letting it burn the night on a broken protocol.

Hard failures (exit 1, chain stops):
  * an empty augmentation scored anything other than 0 or 4 -- the abstention rule is
    not being applied, which is the exact defect the pilot exposed;
  * every score identical -- a judge returning a constant is not measuring anything,
    and is the signature of scores failing to parse and defaulting to 0.

Warnings (exit 0, chain continues): things that are suspicious at n=12 but could
easily be chance, and are not worth forfeiting an unattended window over.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.stage7_evaluation.protocol import is_empty_candidate  # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "protocol_results_pilot.json"
    p = Path(name)
    if not p.exists():
        p = ROOT / "benchmark" / name
    if not p.exists():
        print(f"FAIL  no results at {name}")
        return 1
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        print(f"FAIL  {p.name} is empty -- the describe pass produced nothing")
        return 1

    print(f"[check] {len(rows)} records in {p.name}")
    fails, warns = [], []

    bad = [r for r in rows if is_empty_candidate(r.get("description", ""))
           and int(r["score"]) not in (0, 4)]
    if bad:
        fails.append(f"{len(bad)} empty augmentations scored outside {{0,4}} "
                     f"(e.g. {bad[0]['clip']} = {bad[0]['score']}) -- "
                     "the abstention rule is not being applied")

    scores = [int(r["score"]) for r in rows]
    if len(set(scores)) == 1:
        fails.append(f"every record scored {scores[0]} -- the judge is returning a "
                     "constant, or scores are failing to parse and defaulting")

    # the checks a person would make by eye, kept as warnings at n=12
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(int(r["score"]))
    means = {k: sum(v) / len(v) for k, v in by_sys.items()}
    if len(means) > 1 and max(means.values()) - min(means.values()) < 0.15:
        warns.append("all systems within 0.15 of each other -- the protocol may not be "
                     "separating them; read the records before trusting the full run")
    scn = {r.get("scenario") for r in rows}
    if len(scn) < 2:
        warns.append(f"only one scenario present ({scn}) -- sampling is not stratified")
    generic = sum(1 for r in rows if len(str(r.get("why", ""))) < 12)
    if generic > len(rows) // 3:
        warns.append(f"{generic}/{len(rows)} justifications are near-empty -- the judge "
                     "may be answering without reading")

    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    if fails:
        print("\n[check] STOPPING the chain. Fix the protocol before the full run.")
        return 1
    print(f"[check] means {', '.join(f'{k}={v:.2f}' for k, v in sorted(means.items()))}")
    print("[check] PASSED -- proceeding to the full run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
