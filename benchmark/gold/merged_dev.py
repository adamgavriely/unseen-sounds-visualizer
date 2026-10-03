"""Merged DEV (2026-09-30; docs/history/preregistrations/prereg_round13_detector_push.md "Merge of the tagger set", release v1.2.0): DEV (49 clips, gold_AG)
+ the second-batch DEV part dev2 (the tg_ lines of benchmark/gold/dev_stems.txt, gold_AG), scored as ONE set: per-clip score_clip rows of each
arm on both parts, concatenated, paired clip bootstrap vs B0r (DCC.boot, as round13_dev). Needs the arms' stage-5 outputs
on both parts (round13_dev on DEV; clip_prep stage4/stage5 on dev2). TEST / TEST2 gold is never read.

    python benchmark/gold/merged_dev.py --arms B0r B1 TO1+F7F8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S

OUT = _ROOT / "benchmark" / "gold" / "merged_dev.json"
SYS = "proposed"


def rows_dev(arms):
    gold, stems = DCC.dev_stems()
    out = {}
    for a in arms:
        if a == "B1":                                   # B1 ran in the devcand harness on DEV (round13_dev scores it the same way)
            P = {st: S.load_pictures(DCC.DC / f"B1_{SYS}", st, SYS) or [] for st in stems}
        else:
            with R.flags({k: R.arm_cfg(a)[k] for k in R.DISPLAY_KEYS}):
                P = {st: S.load_pictures(R.R13 / f"{a}_{SYS}", st, SYS) or [] for st in stems}
        out[a] = [(st, S.score_clip(gold[st], P[st])) for st in stems]
    return out


def rows_dev2(arms):
    from benchmark.gold import clip_prep as T
    DCC2, R2, stems = T.configure("dev2")
    keep = set(stems)
    d = json.loads(T.GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out("dev2") / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    gold = T._REAL_LOAD_GOLD([tmp])
    assert set(gold) <= keep
    o = T.out("dev2")
    out = {}
    for a in arms:
        with R2.flags({k: R2.arm_cfg(a)[k] for k in R2.DISPLAY_KEYS}):
            P = {st: S.load_pictures(o / f"{a}_{SYS}", st, SYS) or [] for st in stems}
        out[a] = [(st, S.score_clip(gold[st], P[st])) for st in stems if st in gold]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B0r", "B1", "TO1+F7F8"])
    a = ap.parse_args()
    arms = a.arms if "B0r" in a.arms else ["B0r"] + a.arms
    dev = rows_dev(arms)
    dev2 = rows_dev2(arms)
    res = {"parts": {"dev": len(dev["B0r"]), "dev2": len(dev2["B0r"])}, "rows": {}, "delta_vs_B0r": {}, "per_part": {}}
    cost = {}
    for n in arms:
        rr = [r for _s, r in dev[n]] + [r for _s, r in dev2[n]]
        m = DCC.metrics(rr)
        res["rows"][n] = m
        res["per_part"][n] = {"dev": DCC.metrics([r for _s, r in dev[n]]), "dev2": DCC.metrics([r for _s, r in dev2[n]])}
        cost[n] = [DCC.clip_cost(r) for r in rr]
    for n in arms:
        if n != "B0r":
            res["delta_vs_B0r"][n] = DCC.boot(np.subtract(cost[n], cost["B0r"]))
    for n in arms:
        x = res["rows"][n]; d = res["delta_vs_B0r"].get(n)
        pp = res["per_part"][n]
        print(f"MERGED DEV ({res['parts']['dev']}+{res['parts']['dev2']} clips) {n:12s} hits {x['hits']}/{x['hits'] + x['misses']} "
              f"wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['viewer_cost']:.3f}"
              + (f"  d vs B0r {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] p {d[3]:.3f}" if d else "")
              + f"  | DEV {pp['dev']['hits']}/{pp['dev']['wrong']}  DEV2 {pp['dev2']['hits']}/{pp['dev2']['wrong']}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
