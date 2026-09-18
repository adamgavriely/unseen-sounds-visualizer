"""Which labelled real dev detections PretrainedSED lost at its chosen bar, and what it heard."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from benchmark.gate_dev_sweep import load
from src.labels import canonical
bar = json.load(open("benchmark/psed_setting.json"))["bar_chosen_on_dcase"]
lab = json.load(open("benchmark/dev_phantoms.json")); sounds = {r["clip"]: r["sounds"] for r in load("dev")}
W = Path("benchmark/psed_windows/dev")
for it in lab:
    if it["phantom"]:
        continue
    snd = next((s for s in sounds[it["clip"]] if s["label"] == it["label"]), None)
    z = np.load(W / (Path(it["clip"]).stem + ".npz")); fw = z["fw"].astype(float); t = z["times"]; L = [str(x) for x in z["labels"]]
    m = (t >= snd["start"] - 0.5) & (t <= snd["end"] + 0.5)
    cands = [(float(fw[m, c].max()), L[c]) for c in range(len(L)) if canonical(L[c]) == it["label"] or L[c] == it["label"]]
    best = max(cands) if cands else (None, "NO CLASS")
    top = sorted([(float(fw[m, c].max()), L[c]) for c in range(len(L))], reverse=True)[:3]
    flag = "LOST" if (best[0] is None or best[0] < bar) else "kept"
    print(f"{flag} {it['label']:18s} {snd['start']:5.1f}-{snd['end']:5.1f} beats{snd['confidence']:.2f} psed-best {best[0] if best[0] is None else round(best[0], 2)} ({best[1]})  top {[(round(a, 2), b) for a, b in top]}")
