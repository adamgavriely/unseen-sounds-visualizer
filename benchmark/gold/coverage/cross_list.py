"""List the current system's wrong pictures (all 158 clips) with what the gold had at that moment."""
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
for st in stems("dev") + stems("test"):
    r = st5[st]
    p = pictures(decide(r["P"], r["B"], r["gate"]), float(dur[st] or 10), st)
    _, pp = classify(gold[st], p)
    for q in pp:
        if q["class"].startswith("wrong"):
            near = [f"{g['label']}({g['start']:.1f}-{g['end']:.1f}{',vis' if g['visible'] else ''})" for g in gold[st] if g["end"] > q["start"] - 1 and g["start"] < q["start"] + 2]
            print(f"{st[:30]:30s} {q['label'][:22]:22s} {q['start']:5.1f}-{q['end']:5.1f} {q['class'][7:]:22s} gold: {'; '.join(near)[:110]}")
