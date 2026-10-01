"""Smoke test of the live GROUP path (src/stage6_visual_augmentation/group.ensure) on the two DEV clips that merge:
answers must equal the Round 47 cached ones.   python benchmark/gold/grp_live_smoke.py   (GPU, from ~/MscProj_r13)"""
import json
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.stage6_visual_augmentation import group as GR
from src.types import AugmentationSpec
config.use_shipped()
cache = _ROOT / "data" / "work" / "grp_smoke_answers.json"
cache.unlink(missing_ok=True)
config.GROUP_CACHE = str(cache)
ref = json.loads((_ROOT / "benchmark" / "gold" / "grp" / "group_answers_bench.json").read_text())
ok = True
for st in ("b3_barbershop", "mv_detective_crime_scene"):
    d = _ROOT / "data" / "work" / "r13" / "SHIP8_proposed" / st
    specs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                              augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                              talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                              breaks=[tuple(x) for x in s.get("breaks", [])])
             for s in json.loads((d / "augmentations.json").read_text())]
    dur = float(json.loads((d / "media.json").read_text())["duration"])
    GR.ensure(st, _ROOT / "data" / "work" / "devcand" / "wav16" / f"{st}.wav", specs, dur)
    got = json.loads(cache.read_text())[st]
    print(st, "live", got, "cached", ref.get(st), flush=True)
    ok &= all(got.get(k) == v for k, v in ref.get(st, {}).items() if k in got) and bool(got)
print("SMOKE", "PASS" if ok else "FAIL")
