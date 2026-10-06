"""Dump the frozen arm's on-screen pictures (label, start, end) per clip for scorer v2 (score_coverage.py).
Runs on the cluster checkout of the scored runs (old module names: round13_dev, tagger_prep), CPU only. Reads the
same four parts and display flags as inspector_trail_export.run(); two end rules: MAX_AFTER_END None (scoring
harness) and 1.0 (shipped display). Decides nothing. The committed pics_frozen.json is filtered to dev_stems.txt + test_stems.txt (71 + 87 clips).

    python benchmark/gold/coverage/dump_pictures.py [arm] [out.json]
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import inspector_trail_export as X
try:
    from benchmark.gold import dev_harness as R
except ImportError:
    from benchmark.gold import round13_dev as R

arm = sys.argv[1] if len(sys.argv) > 1 else "SHIP8+MD3+WW5+SL"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "scratch_cov/pics_frozen.json")
disp = {k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}
res = {"arm": arm, "display": {k: str(v) for k, v in disp.items()}, "clips": {}}
for split, part, base, stems in X.parts(arm):
    root = base / f"{arm}_proposed"
    for st in stems:
        rec = {"split": split, "part": part, "root": str(root)}
        for tag, mae in (("pics_none", None), ("pics_1", 1.0)):
            with R.flags({**disp, "MAX_AFTER_END": mae}):
                p = S.load_pictures(root, st, "proposed")
            rec[tag] = None if p is None else [list(x) for x in p]
        m = root / st / "media.json"
        rec["dur"] = json.loads(m.read_text(encoding="utf-8")).get("duration") if m.exists() else None
        aug = root / st / "augmentations.json"
        rec["aug"] = json.loads(aug.read_text(encoding="utf-8")) if aug.exists() else None
        res["clips"][st] = rec
    print(split, part, len(stems), root, flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res), encoding="utf-8")
print("->", out)
