"""Oracle-label diagnostic (2026-09-22, post hoc and labelled as such): what the gate would be worth
if the detector never named a sound that is not there. Every picture whose label matches no gold
sound of that family at that moment is removed from BOTH systems — the false alarms the gate cannot
touch, which dominate the pool — and the per-sound scores are recomputed. It answers "is the gate's
null ΔF1 caused by the detector, or by the metric?".

    python benchmark/gold/oracle_label.py --tag v4b4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"


def oracle(pics, g):
    """keep only pictures that name a sound really present at that moment (same family, onset window
    or overlapping the sound); applied identically to every system"""
    return [(l, a, b) for l, a, b in pics
            if any(S.same_family(l, s["label"]) and (S.in_window(a, s["start"], 0.5, 1.0) or s["start"] <= a <= s["end"]) for s in g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4b4")
    ap.add_argument("--work", default=None)
    ap.add_argument("--subsets", nargs="+", default=["bench", "dev", "test_bench", "sliceB", "cat_mixed"])
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    work = Path(a.work) if a.work else _ROOT / "data" / "work"
    out = {}
    for sub in a.subsets:
        rows = {"proposed": {}, "blind_a2i": {}}
        for stem in sorted(subs.get(sub, ())):
            g = gold[stem]
            got = {}
            for sysn in rows:
                pics = S.load_pictures(work / f"protocol_{sysn}_{a.tag}", stem, sysn)
                if pics is None:
                    break
                got[sysn] = S.score_clip(g, oracle(pics, g))
            if len(got) == 2:
                for sysn in rows:
                    rows[sysn][stem] = got[sysn]
        both = sorted(rows["proposed"])
        if not both:
            continue
        agg = {s: S.aggregate([rows[s][x] for x in both]) for s in rows}
        d = {k: S.paired_ci([rows["proposed"][x] for x in both], [rows["blind_a2i"][x] for x in both], key=k) for k in ("F1", "P", "R")}
        out[sub] = {"clips": len(both), "proposed": agg["proposed"], "blind_a2i": agg["blind_a2i"],
                    "delta": {k: {"d": v[0], "ci": [v[1], v[2]], "p_gt0": v[3]} for k, v in d.items()}}
        print(f"[{sub:11s} oracle-label] clips {len(both):3d} | proposed P {agg['proposed']['P']:.2f} R {agg['proposed']['R']:.2f} F1 {agg['proposed']['F1']:.2f} "
              f"| blind P {agg['blind_a2i']['P']:.2f} R {agg['blind_a2i']['R']:.2f} F1 {agg['blind_a2i']['F1']:.2f} "
              f"| dF1 {d['F1'][0]:+.3f} [{d['F1'][1]:+.3f},{d['F1'][2]:+.3f}] | dP {d['P'][0]:+.3f} [{d['P'][1]:+.3f},{d['P'][2]:+.3f}] | dR {d['R'][0]:+.3f}")
    (_ROOT / "benchmark" / "gold" / f"oracle_label_{a.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
