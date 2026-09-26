"""Week plan B.1 (docs/WEEK_PLAN_2026-09-26.md): amendment 8's three FlexSED rules, re-run on the scorer's clean DEV-49.

Why: amendment 8 was passed on split.json DEV-79, which overlaps 35 of the scorer's 60 TEST clips (prereg_v4.md,
correction of 2026-09-26). Report only: nothing is adopted or reverted. Pass -> a footnote; fail -> the thesis says
"bar selected on a set overlapping TEST". Thresholds unchanged, written before this script ran:

  1. masked sounds recovered >= 44 % (amendment 8: >= 4 of 9). The masked set is re-derived on DEV-49 by the same
     definition before any FlexSED score is read: a needed sound (importance >= 2) whose best same-family BEATs score in
     [onset - 0.5, onset + 1.0] s is < 0.05 while BEATs' top label there is Speech or Music. Recovered = FlexSED's best
     same-family score in that window >= 0.35 (this reproduces all nine original classifications: every recovered
     sound was >= 0.49, both misses <= 0.19); the share at FlexSED's shipped bar 0.8 is printed beside it.
  2. the union BEATs 0.35 + FlexSED 0.8 adds >= 0.05 onset-recall over BEATs alone at matched false labels (BEATs'
     recall at the union's false-label rate, interpolated along BEATs bars 0.10-0.40).
  3. no-ambient false labels of the union <= 2 x BEATs 0.35.

    python benchmark/gold/flexsed_recheck_dev49.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import detector_bench as DB
from benchmark.gold import score_per_sound as S

WIN = (0.5, 1.0)


def frames(stem, det):
    p = DB.CACHES[det] / f"{stem}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    fw = z["fw"].astype(np.float32); labels = [str(x) for x in z["labels"]]
    if "times" in z:
        times = z["times"].astype(np.float64)
    else:
        fw = fw.T; times = np.arange(fw.shape[0]) / float(z["fps"])
    return fw, labels, times


def best(fr, label, a, b):
    fw, labels, times = fr
    m = (times >= a) & (times <= b)
    if not m.any():
        m = np.abs(times - a) == np.abs(times - a).min()
    cols = [j for j, l in enumerate(labels) if S.same_family(l, label)]
    return (float(fw[m][:, cols].max()) if cols else 0.0), fw[m]


def main():
    config.use_v4("59")
    gold = S.load_gold([DB.GOLD])
    dev = sorted(S.subsets_of(gold)["dev"])
    # rule 1: masked set first (BEATs only), then FlexSED
    masked = []
    for st in dev:
        bf = frames(st, "beats")
        if bf is None:
            continue
        for s in gold[st]:
            if not (s["needed"] and s["importance"] >= 2):
                continue
            a, b = s["start"] - WIN[0], s["start"] + WIN[1]
            sc, block = best(bf, s["label"], a, b)
            top = bf[1][int(np.argmax(block.max(axis=0)))]
            if sc < 0.05 and any(k in top for k in ("Speech", "Music")):
                masked.append((st, s["label"], round(s["start"], 2), sc, top))
    print(f"DEV-49: masked needed sounds (BEATs < 0.05 under Speech/Music): {len(masked)}")
    rec35 = rec80 = 0; nf = 0
    for st, lab, t0, sc, top in masked:
        ff = frames(st, "flexsed")
        if ff is None:
            print(f"   {st} {lab} @ {t0}: no FlexSED cache"); continue
        nf += 1
        fsc, _ = best(ff, lab, t0 - WIN[0], t0 + WIN[1])
        rec35 += fsc >= 0.35; rec80 += fsc >= 0.8
        print(f"   {st[:34]:34s} {lab:22s} @ {t0:6.2f}  BEATs {sc:.3f} (top {top})  FlexSED {fsc:.3f}")
    r1 = rec35 / nf if nf else float("nan")
    print(f"RULE 1  recovered {rec35}/{nf} = {r1:.0%} (>= 44 % to pass) -> {'PASS' if nf and r1 >= 0.44 else 'FAIL'};"
          f"  at bar 0.8: {rec80}/{nf}")

    # rules 2-3 from the detector bench on DEV-49
    def row(cfg):
        per = DB.score(gold, dev, cfg)
        d = per["ALL"]; na = per.get("no_ambient", {"fa": 0, "clips": 1})
        return d["onset"] / d["need"], d["fa"] / d["clips"], na["fa"] / max(1, na["clips"])
    union = row("beats@0.35+flexsed@0.8")
    curve = [(bar,) + row(f"beats@{bar}") for bar in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)]
    for bar, r, fa, fna in curve:
        print(f"   BEATs {bar:.2f}: onset-recall {r:.3f}  false labels/clip {fa:.2f}  (no-ambient {fna:.2f})")
    print(f"   union BEATs 0.35 + FlexSED 0.8: onset-recall {union[0]:.3f}  false labels/clip {union[1]:.2f}  (no-ambient {union[2]:.2f})")
    pts = sorted((fa, r) for _, r, fa, _ in curve)
    matched = float(np.interp(union[1], [p[0] for p in pts], [p[1] for p in pts]))
    r2 = union[0] - matched
    print(f"RULE 2  union recall minus BEATs recall at the same false-label rate = {r2:+.3f} (>= +0.05 to pass) -> {'PASS' if r2 >= 0.05 else 'FAIL'}")
    b35 = [c for c in curve if abs(c[0] - 0.35) < 1e-9][0]
    ratio = union[2] / b35[3] if b35[3] else float("inf")
    print(f"RULE 3  no-ambient false labels {union[2]:.2f} vs BEATs 0.35 {b35[3]:.2f} = {ratio:.2f}x (<= 2x to pass) -> {'PASS' if ratio <= 2 else 'FAIL'}")
    out = {"masked": len(masked), "flexsed_cached": nf, "recovered_035": rec35, "recovered_080": rec80, "rule1": r1,
           "union": union, "beats_curve": curve, "rule2_gain": r2, "rule3_ratio": ratio}
    (_ROOT / "benchmark" / "gold" / "flexsed_recheck_dev49.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
