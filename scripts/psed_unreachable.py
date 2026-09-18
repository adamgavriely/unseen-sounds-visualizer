"""List buried consequential sounds on the calibration set that PSED never proposes (no box at
the loose bar 0.05), with the clip id, the sound, its time, what covers it, PSED's best score
for that family in the span, and what PSED heard instead."""
import sys, json, numpy as np
sys.path.insert(0, ".")
from benchmark import audioset_detector_eval as E
from benchmark.fusion_v5 import boxes, LOOSE, gold_events
from src.labels import canonical, is_descendant
E.use_set("calib")
n = 0
for c in E.clips():
    p = E.WIN / "psed" / f"{c['id']}.npz"
    if not p.exists(): continue
    fw, t, lab = E._load(p)
    bx = boxes(fw, t, lab, LOOSE)
    for g in gold_events(c):
        if not (g["consequential"] and g["masked"]): continue
        hit = any(E._same(b[0], g["label"]) and E._overlap_ok(b[1], b[2], g["start"], g["end"]) for b in bx)
        if hit: continue
        m = (t >= g["start"]) & (t <= g["end"])
        fam = [i for i, l in enumerate(lab) if l == g["label"] or canonical(l) == canonical(g["label"]) or is_descendant(l, g["label"]) or is_descendant(g["label"], l)]
        best = float(fw[m][:, fam].max()) if m.any() and fam else 0.0
        top = sorted(((float(fw[m, i].max()), lab[i]) for i in range(len(lab))), reverse=True)[:4] if m.any() else []
        covers = [e["label"] for e in c["events"] if e["label"] in ("Speech", "Music", "Male speech, man speaking", "Female speech, woman speaking") and min(e["end"], g["end"]) - max(e["start"], g["start"]) > 0]
        print(f"{c['id']}  {g['label']} {g['start']:.1f}-{g['end']:.1f}s  under {covers}  PSED best for this family {best:.2f}  PSED top: {[(round(a,2), b) for a, b in top]}  https://youtu.be/{c['id'].rsplit('_',1)[0]}?t={int(c['id'].rsplit('_',1)[1])//1000 + int(g['start'])}")
        n += 1
print("unreachable:", n)
