"""Gold-input (oracle-detector) gate test — the experiment three reviewers called the highest value
one (2026-09-22). The detector is replaced by the annotator's own sound list: every gold sound
becomes a "detection" with its true label and onset, BLIND draws all of them, GATE draws the ones
its visibility check did not silence (the verdicts cached by benchmark/gold/gate_gold.py, so no new
GPU work). Scored with the same per-sound rules. It isolates the project's novelty from the
detector, and it reports the gate as what it is: a classifier of "is this sound's source on screen".

    python benchmark/gold/oracle_gate.py --arm Qwen38-27B [--rule majority]
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
from benchmark.gold.gate_gold import OUT_DIR, decide

MIN_DWELL = 1.5


def pictures(sounds, silenced, duration=None):
    """one picture per sound the system draws, from its onset (the display timeline of a system
    with a perfect detector: no merging, no row overflow -- every drawn sound gets its picture)"""
    out = []
    for i, s in enumerate(sounds):
        if i in silenced:
            continue
        a = float(s["start"]); b = max(float(s["end"]), a + MIN_DWELL)
        cap = getattr(config, "MAX_SPAN", None)
        if cap:
            b = min(b, a + float(cap))
        out.append((s["label"], a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="Qwen38-27B", help="folder under benchmark/gold/gate_gold")
    ap.add_argument("--rules", nargs="+", default=["majority", "unanimous", "majority+obvious"])
    ap.add_argument("--subsets", nargs="+", default=["bench", "dev", "test_bench", "sliceB", "cat_mixed", "cat_seen"])
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    subs = S.subsets_of(gold)
    files = {f.stem: f for f in (OUT_DIR / a.arm).glob("*.json")}
    out = {}
    for rule in a.rules:
        rows_g, rows_b, conf = {}, {}, {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for stem, snds in gold.items():
            f = files.get(stem)
            if f is None:
                continue
            cached = json.loads(f.read_text(encoding="utf-8"))["sounds"]
            # align the cached votes with the scorer's gold list by (resolved label, start)
            bykey = {(c["label"], round(float(c["start"]), 2)): c for c in cached}
            silenced = set()
            usable = True
            for i, s in enumerate(snds):
                c = bykey.get((s["label"], round(s["start"], 2)))
                if c is None:
                    usable = False; break
                seen = (c.get("seen_owl") if a.arm == "owlv2" else decide(c["stretches"], rule))
                if seen:
                    silenced.add(i)
                if s["importance"] >= 2:                       # the gate as a visibility classifier
                    if s["needed"]:
                        conf["fp" if seen else "tn"] += 1      # a needed sound silenced = false positive
                    else:
                        conf["tp" if seen else "fn"] += 1      # a visible/obvious sound silenced = true positive
            if not usable:
                continue
            rows_g[stem] = S.score_clip(snds, pictures(snds, silenced))
            rows_b[stem] = S.score_clip(snds, pictures(snds, set()))
        print(f"--- rule {rule}: {len(rows_g)} clips; gate as a visibility classifier: "
              f"sensitivity {conf['tp'] / max(1, conf['tp'] + conf['fn']):.2f} (visible/obvious sounds silenced), "
              f"specificity {conf['tn'] / max(1, conf['tn'] + conf['fp']):.2f} (needed sounds kept), "
              f"balanced {0.5 * (conf['tp'] / max(1, conf['tp'] + conf['fn']) + conf['tn'] / max(1, conf['tn'] + conf['fp'])):.2f}")
        for sub in a.subsets:
            stems = sorted(st for st in subs.get(sub, ()) if st in rows_g)
            if not stems:
                continue
            ag = S.aggregate([rows_g[s] for s in stems]); ab = S.aggregate([rows_b[s] for s in stems])
            d = {k: S.paired_ci([rows_g[s] for s in stems], [rows_b[s] for s in stems], key=k) for k in ("F1", "P", "R")}
            out[f"{rule}|{sub}"] = {"clips": len(stems), "gate": ag, "blind": ab,
                                    "delta": {k: {"d": v[0], "ci": [v[1], v[2]], "p_gt0": v[3]} for k, v in d.items()}}
            print(f"    [{sub:11s}] clips {len(stems):3d} | gate P {ag['P']:.2f} R {ag['R']:.2f} F1 {ag['F1']:.2f} (vis-FA {ag['visible']}, clean {ag['clean_acc'] if ag['clean_acc'] is None else round(ag['clean_acc'], 2)}) "
                  f"| blind P {ab['P']:.2f} R {ab['R']:.2f} F1 {ab['F1']:.2f} (vis-FA {ab['visible']}, clean {ab['clean_acc'] if ab['clean_acc'] is None else round(ab['clean_acc'], 2)}) "
                  f"| dF1 {d['F1'][0]:+.3f} [{d['F1'][1]:+.3f},{d['F1'][2]:+.3f}] P(d>0)={d['F1'][3]:.3f} | dP {d['P'][0]:+.3f} [{d['P'][1]:+.3f},{d['P'][2]:+.3f}]")
    (_ROOT / "benchmark" / "gold" / f"oracle_gate_{a.arm}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
