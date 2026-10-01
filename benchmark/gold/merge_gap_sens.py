"""Merge gap sensitivity (for Adam's decision, 2026-10-01): the shipped config (config.use_shipped) joins repeats within
MERGE_GAP 1.5 s, every score used the harness value 2.0 (round13_dev.BASE, a display key read at score time). This re-reads the
SAVED pictures of SHIP8 and B0r at one MERGE_GAP per process, on merged DEV (merged_dev.rows_dev / rows_dev2) or on merged TEST
(final_test.score's loading: old TEST arm SHIP7+K4AD in data/work/r16final + tagger test2). Reported, not selected on.

    TG_ARMS=SHIP8 python benchmark/gold/merge_gap_sens.py dev  --gap 1.5     # one process per gap (tagger_prep stubs load_gold)
    TG_ARMS=SHIP8 python benchmark/gold/merge_gap_sens.py test --gap 1.5     # -> merge_gap_sens/<set>_<gap>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import score_per_sound as S

OUTD = _ROOT / "benchmark" / "gold" / "merge_gap_sens"


def summ(DCC, rows):
    m = DCC.metrics(rows)
    return {k: m[k] for k in ("hits", "misses", "wrong", "visible", "cross", "phantom", "viewer_cost")}


def dev(gap):
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import merged_dev as M
    from benchmark.gold import round13_dev as R
    arms = ["B0r", "SHIP8"]
    for a in arms:
        R.ARMS[a]["MERGE_GAP"] = gap
    d1 = M.rows_dev(arms)
    d2 = M.rows_dev2(arms)
    return {a: summ(DCC, [r for _s, r in d1[a]] + [r for _s, r in d2[a]]) for a in arms}


def test(gap):
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import final_test as FT
    ARM = "SHIP7+K4AD"                                         # the shipped SHIP8 TEST render's arm name
    F, T, R, _arms, fin = FT.old_setup(ARM)                    # resets ARMS to the DEV originals, so the gap is set after
    stems1 = list(T.STEMS)
    P1 = {}
    for a in ("B0r", ARM):
        R.ARMS[a]["MERGE_GAP"] = gap
        with R.flags({k: R.arm_cfg(a)[k] for k in R.DISPLAY_KEYS}):
            P1[a] = {st: S.load_pictures(fin / f"{a}_proposed", st, "proposed") or [] for st in stems1}
    from benchmark.gold import tagger_prep as TP
    for k, v in FT._ARMS0.items():
        R.ARMS[k] = dict(v)
    TP._ORIG.clear()
    _D2, R2, stems2 = TP.configure("test2")
    o2 = TP.out("test2")
    P2 = {}
    for a in ("B0r", ARM):
        R2.ARMS[a]["MERGE_GAP"] = gap
        with R2.flags({k: R2.arm_cfg(a)[k] for k in R2.DISPLAY_KEYS}):
            P2[a] = {st: S.load_pictures(o2 / f"{a}_proposed", st, "proposed") or [] for st in stems2}
    S.load_gold = FT._REAL_LOAD_GOLD
    gold1 = S.load_gold([F.GOLD])
    assert sorted(S.subsets_of(gold1)["test_bench"]) == sorted(stems1)
    d = json.loads(TP.TAGGER_GOLD.read_text(encoding="utf-8"))
    keep = set(stems2)
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = o2 / "test2_gold_only.json"
    DCC.dump(tmp, d)
    gold2 = S.load_gold([tmp])
    out = {}
    for a, name in (("B0r", "B0r"), (ARM, "SHIP8")):
        rr = [S.score_clip(gold1[st], P1[a][st]) for st in stems1] + \
             [S.score_clip(gold2[st], P2[a][st]) for st in stems2 if st in gold2]
        out[name] = summ(DCC, rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=("dev", "test"))
    ap.add_argument("--gap", type=float, required=True)
    a = ap.parse_args()
    res = {"set": a.set, "gap": a.gap, "rows": (dev if a.set == "dev" else test)(a.gap)}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{a.set}_{a.gap}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for n, x in res["rows"].items():
        print(f"{a.set} gap {a.gap} {n:6s} hits {x['hits']}/{x['hits'] + x['misses']} wrong {x['wrong']} "
              f"({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['viewer_cost']:.3f}", flush=True)


if __name__ == "__main__":
    main()
