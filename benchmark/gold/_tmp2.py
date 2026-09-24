"""How far is our onset from the annotator's, for the sounds we did draw?"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

AUG = Path(sys.argv[1])
gold = S.load_gold([GOLD])
rows = []
for stem, snds in gold.items():
    f = AUG/stem/"augmentations.json"
    if not f.exists(): continue
    for sp in json.loads(f.read_text()):
        if not sp.get("augment"): continue
        pic = min([x[0] for x in (sp.get("spans") or [[sp["start"], sp["end"]]])])
        cand = [s for s in snds if S.same_family(sp["event_label"], s["label"])]
        if not cand: continue
        g = min(cand, key=lambda s: abs(s["start"] - pic))
        if abs(g["start"] - pic) > 6.0: continue
        rows.append({"clip": stem, "label": sp["event_label"], "ours": round(pic,2),
                     "gold": round(g["start"],2), "d": round(pic - g["start"],2),
                     "ourend": round(max([x[1] for x in (sp.get("spans") or [[sp["start"],sp["end"]]])]),2),
                     "goldend": round(g["end"],2)})
d = np.array([r["d"] for r in rows])
print(f"{len(rows)} drawn sounds matched to a gold sound of the same family\n")
print(f"  start:  median {np.median(d):+.2f}s   mean {d.mean():+.2f}s   "
      f"early(<-0.5) {int((d<-0.5).sum())}   on time {int((abs(d)<=0.5).sum())}   late(>0.5) {int((d>0.5).sum())}")
for q in (10,25,50,75,90): print(f"    p{q:<3d} {np.percentile(d,q):+.2f}s")
e = np.array([r["ourend"]-r["goldend"] for r in rows])
print(f"\n  end:    median {np.median(e):+.2f}s  (positive = our picture outlasts the sound)")
print("\n  worst early:")
for r in sorted(rows, key=lambda r: r["d"])[:10]:
    print(f"    {r['clip'][:30]:30s} {r['label'][:14]:14s} ours {r['ours']:6.2f}  gold {r['gold']:6.2f}  {r['d']:+.2f}")
json.dump(rows, open("benchmark/gold/onset_bias_dev.json","w"), indent=1)
