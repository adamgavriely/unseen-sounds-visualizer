"""v4ab2 bar (docs/prereg_v4.md): the challenger's operating point chosen by span-level F1 on the
AudioSet-Strong calibration set (precision = spans overlapping a same-family gold event of any
kind; recall = consequential gold events hit), instead of inheriting BEATs' false-alarm rate.
Same rule for BEATs, reported for reference. CPU, cached windows -> benchmark/psed_f1_bar.json"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from benchmark import audioset_detector_eval as E
from src.labels import is_salient_nonspeech, is_music

E.use_set("calib")
clips = [c for c in E.clips()]
def stats(which, bar):
    tp = fp = 0; hit = tot = 0
    for c in clips:
        p = E.WIN / which / f"{c['id']}.npz"
        if not p.exists(): continue
        fw, t, lab = E._load(p)
        ev = [e for e in E._spans(fw, t, lab, bar) if is_salient_nonspeech(e.label) and not is_music(e.label)]
        gold = [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"])]
        for e in ev:
            if any(E._same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"]): tp += 1
            else: fp += 1
        for g in gold:
            if not g["consequential"]: continue
            tot += 1
            hit += any(E._same(e.label, g["label"]) and E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev)
    P = tp / max(1, tp + fp); R = hit / max(1, tot)
    return {"bar": bar, "precision": P, "recall": R, "f1": 2 * P * R / max(1e-9, P + R), "spans": tp + fp}
out = {}
for which in ("psed", "beats"):
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]
    rows = [stats(which, b) for b in grid]
    best = max(rows, key=lambda r: r["f1"])
    out[which] = {"best": best, "curve": rows}
    print(f"{which}: best F1 {best['f1']:.3f} at bar {best['bar']:.2f} (P {best['precision']:.2f} R {best['recall']:.2f}, {best['spans']} spans);"
          f" at 0.15: F1 {[r for r in rows if r['bar']==0.15][0]['f1']:.3f}; at 0.35: F1 {[r for r in rows if r['bar']==0.35][0]['f1']:.3f}")
json.dump(out, open("benchmark/psed_f1_bar.json", "w"), indent=1)
