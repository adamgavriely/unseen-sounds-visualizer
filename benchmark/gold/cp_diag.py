"""Round 33 diagnostic (CPU, record only): SHIP7's saved merged-DEV pictures against the cached FlexSED frame scores.
For every needed miss: the same-family pictures in the clip and their offset from the gold onset, the family's FlexSED
evidence around the onset and the step-filter change points (cSEBB deltas, Ebbers 2024) near it. For every wrong picture:
its kind, the nearest gold sound, and the delta peak near its start. No rule is applied; numbers are read only to choose
which post-processing family (onset re-timing vs class competition) the pre-registered screen targets.

    python benchmark/gold/cp_diag.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP7")
from benchmark.gold import btp_screen as B
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical

B.ARM = "SHIP7"
STEP = 0.48                                             # Ebbers 2024 grid middle {0.32, 0.48, 0.64}
OUT = _ROOT / "benchmark" / "gold" / "cp_diag.json"


def family_ev(fr, fam):
    fw, ft, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols:
        return None, ft
    return fw[:, cols].max(axis=1), ft


def deltas(ev, ft, step=STEP):
    dt = float(ft[1] - ft[0]); h = max(1, int(round(step / 2 / dt)))
    n = len(ev); d = np.zeros(n, np.float32)
    cs = np.concatenate([[0.0], np.cumsum(ev, dtype=np.float64)])
    for i in range(n):
        a0, a1 = max(0, i - h), i
        b0, b1 = i, min(n, i + h)
        pre = (cs[a1] - cs[a0]) / max(1, a1 - a0); post = (cs[b1] - cs[b0]) / max(1, b1 - b0)
        d[i] = post - pre
    return d


def peaks(d, ft, lo, hi):
    out = []
    for i in range(1, len(d) - 1):
        if lo <= ft[i] <= hi and d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] > 0:
            out.append((round(float(ft[i]), 2), round(float(d[i]), 3)))
    return out


def main():
    P = B.parts()
    res = []; miss_kind = {"no_same_family_pic": 0, "same_family_pic_early": 0, "same_family_pic_late": 0}
    offs = []
    for pt, st, g, pics in P:
        fx = B.FLEX_DIR / f"{st}.npz"; fr = DCC.load_fr(fx) if fx.exists() else None
        cl = classify(g, [p[:3] for p in pics])
        gold = sorted(g, key=lambda x: x["start"])
        needed = [x for x in gold if x["needed"] and x["importance"] >= S.MIN_IMPORTANCE]
        hit_idx = {i for _l, _a, _b, k, i in cl if k in ("hit", "collision") and i is not None}
        row = {"part": pt, "clip": st, "pics": [[l, round(a, 2), round(b, 2), k] for l, a, b, k, _ in cl], "misses": [], "wrong": []}
        for gi, x in enumerate(gold):
            if x not in needed or gi in hit_idx:
                continue
            fam = canonical(x["label"])
            same = [(l, round(a, 2), round(a - x["start"], 2)) for l, a, b, r in pics if S.same_family(l, x["label"])]
            m = {"label": x["label"], "family": fam, "onset": round(x["start"], 2), "end": round(x["end"], 2), "same_family_pics": same}
            if not same:
                miss_kind["no_same_family_pic"] += 1
            else:
                o = min(same, key=lambda s: abs(s[2]))[2]; offs.append(o)
                miss_kind["same_family_pic_early" if o < 0 else "same_family_pic_late"] += 1
            if fr is not None:
                ev, ft = family_ev(fr, fam)
                if ev is not None:
                    w = (ft >= x["start"] - 0.5) & (ft <= x["start"] + 1.0)
                    m["flex_max_in_window"] = round(float(ev[w].max()), 3) if w.any() else None
                    m["flex_clip_max"] = round(float(ev.max()), 3)
                    m["delta_peaks_near"] = peaks(deltas(ev, ft), ft, x["start"] - 1.5, x["start"] + 2.5)
            row["misses"].append(m)
        for l, a, b, k, i in cl:
            if k not in ("cross", "visible", "phantom"):
                continue
            near = [(y["label"], round(y["start"], 2), bool(y["needed"])) for y in gold if S.in_window(a, y["start"], S.EARLY, S.LATE) or y["start"] <= a <= y["end"]]
            w = {"label": l, "family": canonical(l), "start": round(a, 2), "kind": k, "sounds_at": near}
            if fr is not None:
                ev, ft = family_ev(fr, canonical(l))
                if ev is not None:
                    w["delta_peaks_near"] = peaks(deltas(ev, ft), ft, a - 1.5, a + 1.5)
                    ww = (ft >= a - 0.5) & (ft <= a + 1.0)
                    w["flex_max_in_window"] = round(float(ev[ww].max()), 3) if ww.any() else None
            row["wrong"].append(w)
        res.append(row)
    print("miss kinds:", miss_kind, "offsets of nearest same-family picture:", sorted(offs))
    for r in res:
        for m in r["misses"]:
            print(f"MISS {r['part']:4s} {r['clip']:32s} {m['label']:28s} on {m['onset']:6.2f} same {m['same_family_pics']} "
                  f"fxwin {m.get('flex_max_in_window')} fxmax {m.get('flex_clip_max')} dpk {m.get('delta_peaks_near')}")
        for w in r["wrong"]:
            print(f"WRONG {r['part']:4s} {r['clip']:32s} {w['label']:28s} at {w['start']:6.2f} {w['kind']:8s} sounds {w['sounds_at']} "
                  f"fxwin {w.get('flex_max_in_window')} dpk {w.get('delta_peaks_near')}")
    OUT.write_text(json.dumps({"miss_kinds": miss_kind, "offsets": offs, "clips": res}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
