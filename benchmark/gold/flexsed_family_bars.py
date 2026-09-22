"""A per-family bar for FlexSED, fitted on the AudioSet-Strong calibration set (amendment 11).

The problem, measured on presence only: at the 78 needed gold sounds whose family FlexSED knows, its
score AT the sound has median 0.75 and spans 0.01-0.98, so a single bar of 0.8 refuses 59% of real
sounds. The scale is family-dependent -- Chainsaw, Bell and Siren reach 0.96-0.99 while Dog never
passes 0.56 anywhere.

The bars are fitted on the 280 clips of data/input/audioset_calib, which the gold never touches
(the id overlap with benchmark/gold/audioset_slice.json was checked first and is zero), by the same
criterion that set every other detector's bar in this project: the loosest grid bar whose false spans
per minute stay inside a budget taken from BEATs at 0.35 on the same clips. The budget is POOLED --
BEATs' global false-span rate divided by the number of eligible families -- because BEATs' per-family
rate is 0-2 spans and is exactly zero for the families it is deaf to, which would clamp FlexSED to
silence on precisely the sounds this is meant to recover.

Support rule: a family gets its own bar only with at least K = 8 positive spans in the calibration
set. Everything else keeps the global 0.8. Both numbers were fixed in docs/prereg_v4.md first.

    python benchmark/gold/flexsed_family_bars.py        # CPU, reads data/work/flexsed_calib
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical

CALIB = _ROOT / "data" / "work" / "flexsed_calib"
OUT = _ROOT / "benchmark" / "gold" / "flexsed_family_bars.json"
GRID = [round(b, 2) for b in np.arange(0.05, 0.96, 0.05)]
K = 8
GLOBAL_BAR = 0.8
MIN_DUR = 0.2


def spans_above(v, fps, bar):
    """contiguous (start, end) runs of one family's score above `bar`, at least MIN_DUR long"""
    on = v >= bar
    if not on.any():
        return []
    idx = np.flatnonzero(on)
    out = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if run.size == 0:
            continue
        a, b = run[0] / fps, (run[-1] + 1) / fps
        if b - a >= MIN_DUR:
            out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    config.use_v4("59")
    from benchmark import audioset_detector_eval as E
    E.use_set("calib")
    clips = {c["id"]: c for c in E.clips()}
    vocab = json.loads((_ROOT / "benchmark" / "gold" / "depictable_vocab.json").read_text(encoding="utf-8"))["families"]
    vset = {canonical(v) for v in vocab}

    # positive spans per family, from the calibration set's own strong labels
    truth = defaultdict(lambda: defaultdict(list))
    support = defaultdict(int)
    minutes = 0.0
    for cid, c in clips.items():
        minutes += float(c.get("duration") or 10.0) / 60.0
        for e in c.get("events", []):
            k = canonical(e["label"])
            if k in vset:
                truth[cid][k].append((float(e["start"]), float(e["end"])))
                support[k] += 1
    eligible = sorted([f for f, n in support.items() if n >= a.k], key=lambda f: -support[f])
    beats_fp_min = json.loads((_ROOT / "benchmark" / "detector_calib.json").read_text(encoding="utf-8"))["beats_ref"]["fp_per_min"]
    budget = beats_fp_min / max(1, len(eligible))
    print(f"[calib] {len(clips)} clips, {minutes:.1f} min, {len(eligible)} eligible families (K={a.k})")
    print(f"[calib] BEATs global false spans/min {beats_fp_min:.2f} -> per-family budget {budget:.3f}")

    # false spans per family at every grid bar
    false_at = defaultdict(lambda: np.zeros(len(GRID)))
    for f in sorted(CALIB.glob("*.npz")):
        cid = f.stem
        if cid not in clips:
            continue
        try:
            z = np.load(f, allow_pickle=False)
        except Exception:
            print("bad npz", f.name); continue
        labs = [str(x) for x in z["labels"]]
        fw = z["fw"].astype(np.float32)
        fps = float(z["fps"])
        for j, lab in enumerate(labs):
            k = canonical(lab)
            if k not in support or support[k] < a.k:
                continue
            gt = truth[cid].get(k, [])
            v = fw[j]
            for gi, bar in enumerate(GRID):
                for s0, s1 in spans_above(v, fps, bar):
                    if not any(s0 < g1 and g0 < s1 for g0, g1 in gt):
                        false_at[k][gi] += 1

    bars = {}
    print(f"\n{'family':26s} {'support':>7s} {'bar':>5s} {'FP/min at bar':>13s}")
    for fam in eligible:
        rates = false_at[fam] / max(1e-9, minutes)
        ok = [i for i, r in enumerate(rates) if r <= budget]
        if not ok:
            bars[fam] = GLOBAL_BAR
            print(f"{fam[:26]:26s} {support[fam]:7d} {GLOBAL_BAR:5.2f} {'over budget everywhere':>13s}")
            continue
        i = min(ok)                                   # the LOOSEST bar inside the budget
        bars[fam] = GRID[i]
        print(f"{fam[:26]:26s} {support[fam]:7d} {GRID[i]:5.2f} {rates[i]:13.3f}")
    OUT.write_text(json.dumps({"when": "2026-09-23", "k": a.k, "global_bar": GLOBAL_BAR,
                               "beats_fp_per_min": beats_fp_min, "budget_per_family": budget,
                               "calib_minutes": minutes, "bars": bars}, indent=1), encoding="utf-8")
    lower = sum(1 for v in bars.values() if v < GLOBAL_BAR)
    print(f"\n{lower} of {len(bars)} eligible families get a LOOSER bar than {GLOBAL_BAR}; the rest of the vocabulary keeps it")
    print("->", OUT)


if __name__ == "__main__":
    main()
