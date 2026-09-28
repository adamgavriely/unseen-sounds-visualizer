"""Bug fix check (found in TEST clip mc_bridge_scene, 28 Sept): stage 5 silenced a sound whose OWN gate check said "not
visible" because a more GENERAL label of the same family was visible at the same time ("cars" seen -> Vehicle visible ->
Helicopter silenced). Fix: a visible label may silence only the same label or a more general one (a visible "Chink, clink"
silences "Glass"), never a more specific one. Simulated on the scored renders by re-enabling the specs the old rule silenced
parent -> child; scored with the official scorer on DEV and TEST (onset rule).

    python benchmark/gold/kinship_fix_check.py      # cluster
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from src.labels import ancestors

WORK = _ROOT / "data" / "work"
SETS = {"DEV": ("dev", "dev_monocap_v31"), "TEST": ("test_bench", "test_final_v33")}
PAT = re.compile(r"^a kind of (.+), whose source is visible - stay silent$")


def variant(root):
    tmp = Path(tempfile.mkdtemp(prefix="kin_"))
    changed = []
    for d in root.iterdir():
        if not (d / "augmentations.json").exists():
            continue
        (tmp / d.name).mkdir()
        if (d / "media.json").exists():
            shutil.copy(d / "media.json", tmp / d.name / "media.json")
        specs = json.loads((d / "augmentations.json").read_text(encoding="utf-8"))
        anypng = next((str(p) for p in (d / "augmentations").glob("*.png")), str(_ROOT / "README.md"))
        for s in specs:
            m = PAT.match(s.get("reason") or "")
            if m and m.group(1) in ancestors(s["event_label"]):        # a more general label silenced a specific one
                s["augment"] = True
                s["image_path"] = s.get("image_path") or anypng
                changed.append((d.name, s["event_label"], m.group(1)))
            elif s.get("image_path") and not Path(s["image_path"]).exists():
                c = d / "augmentations" / Path(s["image_path"]).name
                s["image_path"] = str(c) if c.exists() else anypng
        (tmp / d.name / "augmentations.json").write_text(json.dumps(specs), encoding="utf-8")
    return tmp, changed


def main():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    subs = S.subsets_of(gold)
    for split, (sub, tag) in SETS.items():
        stems = sorted(subs[sub])
        root = WORK / f"protocol_proposed_{tag}"
        tmp, changed = variant(root)
        for name, r in (("scored", root), ("fixed", tmp)):
            a = S.aggregate([S.score_clip(gold[st], S.load_pictures(r, st, "proposed") or []) for st in stems])
            print(f"{split} {name:6s} hits {a['hits']}/{a['hits'] + a['misses']} wrong {a['visible'] + a['cross'] + a['phantom']} "
                  f"F1 {a['F1']:.3f} cost {a['viewer_cost']:.2f}")
        print(f"  re-enabled: {changed}")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
