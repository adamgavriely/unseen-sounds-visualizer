"""Which step moves an onset, and in which direction? (2026-09-24)

Reads the onset_trace.json the pipeline now writes and prints, per step, how many spans it moved
earlier, how many later, and by how much. The panel of three reviewers gave three different answers
to this question and none of them fitted the cached scores; this answers it from the pipeline's own
record.

    python benchmark/gold/trace_report.py --specs data/work/protocol_proposed_dev_trace_v30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

ORDER = ["extract", "union", "veto", "refine", "cap"]


def by_step(rows):
    out = {}
    for r in rows:
        out.setdefault(r["step"], []).append(r)
    return out


def match(prev, cur):
    """pair spans of the same label across two steps, nearest start first"""
    pairs, used = [], set()
    for a in prev:
        cand = [(abs(a["start"] - b["start"]), j) for j, b in enumerate(cur)
                if b["label"] == a["label"] and j not in used]
        if not cand:
            continue
        _, j = min(cand)
        used.add(j)
        pairs.append((a, cur[j]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    a = ap.parse_args()
    root = Path(a.specs)
    gold = S.load_gold([GOLD])

    moves = {}
    print("per-clip, the steps that moved a span's start\n")
    for d in sorted(root.iterdir()):
        f = d / "onset_trace.json"
        if not f.exists():
            continue
        steps = by_step(json.loads(f.read_text(encoding="utf-8")))
        present = [s for s in ORDER if s in steps]
        print(f"== {d.name}")
        for x, y in zip(present, present[1:]):
            for pa, pb in match(steps[x], steps[y]):
                dt = pb["start"] - pa["start"]
                if abs(dt) < 0.02:
                    continue
                moves.setdefault(y, []).append(dt)
                print(f"   {x} -> {y:8s} {pa['label'][:22]:22s} {pa['start']:6.2f} -> "
                      f"{pb['start']:6.2f}   {dt:+.2f}")
        # what the viewer finally gets, beside the sound the annotator marked
        for r in steps.get("display", []):
            cand = [g for g in gold.get(d.name, []) if S.same_family(r["label"], g["label"])]
            if not cand:
                continue
            g = min(cand, key=lambda g: abs(g["start"] - r["start"]))
            print(f"   DISPLAY            {r['label'][:22]:22s} {r['start']:6.2f} - {r['end']:6.2f}"
                  f"   annotator {g['start']:6.2f} - {g['end']:6.2f}   "
                  f"start {r['start'] - g['start']:+.2f}  end {r['end'] - g['end']:+.2f}")
        print()

    print("summary: what each step does to a start")
    print(f"   {'step':10s} {'moved':>6s} {'earlier':>8s} {'later':>6s} {'median':>8s} {'worst':>8s}")
    for step in ORDER[1:]:
        v = np.array(moves.get(step, []))
        if not len(v):
            print(f"   {step:10s} {0:6d}        -      -        -        -")
            continue
        print(f"   {step:10s} {len(v):6d} {int((v < 0).sum()):8d} {int((v > 0).sum()):6d} "
              f"{np.median(v):+8.2f} {v.min():+8.2f}")


if __name__ == "__main__":
    main()
