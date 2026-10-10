"""Misses of v1.4 whose onset falls inside a same-family picture that started earlier (a repeat shown as one picture)."""
import json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.inspector_data import classify
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
n = rep = 0
for st in stems("dev") + stems("test"):
    p = pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st)
    res, _ = classify(gold[st], p)
    for g, o in zip(gold[st], res):
        if o["outcome"] == "miss":
            n += 1
            cov = [q for q in p if S.same_family(q[0], g["label"]) and q[1] < g["start"] - 0.5 and q[2] > g["start"]]
            if cov:
                rep += 1; print(f"{st[:28]:28s} {g['label'][:20]:20s} onset {g['start']:.1f} inside picture {cov[0][0]} {cov[0][1]:.1f}-{cov[0][2]:.1f}")
print("misses", n, "inside an earlier same-family picture", rep)
