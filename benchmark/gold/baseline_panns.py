"""External baseline (b) (docs/prereg_v4.md, 2026-09-28): "what an engineer would do" -- ONE off-the-shelf sound-event
detector (PANNs CNN14, AudioSet, frame-level) and a picture for every drawable detection; no gate, none of our detector
stack. Onset = first frame above the threshold (span extractor, min duration 0.5 s); the same drawable-label filter as
our system. The threshold is chosen on DEV by the viewer cost at beta = 2 (grid fixed below), then TEST is scored once.
Per-sound scorer and paired clip bootstrap as for every other system; ours = the scored renders (docs/inspector/data.json).

    python benchmark/gold/baseline_panns.py                 # PANNs (cache benchmark/gold/panns_fw) -> baseline_panns.json
    python benchmark/gold/baseline_panns.py --model psed    # PretrainedSED (data/work/psed_cache) -> baseline_psed.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from src.labels import is_salient_nonspeech, is_music
from src.stage4_audio_event_detection import _extract_events

MODELS = {"panns": (_ROOT / "benchmark" / "gold" / "panns_fw", _ROOT / "benchmark" / "gold" / "baseline_panns.json"),
          # PretrainedSED (Schmid et al., CP-JKU, ICASSP 2025), BEATs backbone fine-tuned frame by frame on AudioSet-Strong
          # (447 classes): the strongest open frame-level sound-event detector on AudioSet-Strong
          "psed": (_ROOT / "data" / "work" / "psed_cache", _ROOT / "benchmark" / "gold" / "baseline_psed.json")}
CACHE, OUT = MODELS["panns"]
GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def pictures(stem, thr):
    z = np.load(CACHE / f"{stem}.npz", allow_pickle=False)
    fw, ts, labs = z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]
    ev = _extract_events(fw, ts, labs, thr, None, config.AED_MIN_DUR, low=thr * 0.5)
    return [(e.label, float(e.start), float(e.end)) for e in ev if is_salient_nonspeech(e.label) and not is_music(e.label)]


def rows_for(gold, stems, thr):
    return [S.score_clip(gold[st], pictures(st, thr)) for st in stems]


def main():
    import argparse
    global CACHE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(MODELS), default="panns")
    CACHE, OUT = MODELS[ap.parse_args().model]
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    subs = S.subsets_of(gold)
    dev, test = sorted(subs["dev"]), sorted(subs["test_bench"])
    grid = {}
    for t in GRID:
        a = S.aggregate(rows_for(gold, dev, t))
        grid[t] = {k: a[k] for k in ("hits", "misses", "visible", "cross", "phantom", "F1", "viewer_cost")}
        print(f"DEV thr {t:.2f}: hits {a['hits']}/{a['hits'] + a['misses']} wrong {a['visible'] + a['cross'] + a['phantom']} "
              f"F1 {a['F1']:.3f} cost {a['viewer_cost']:.2f}")
    pick = min(GRID, key=lambda t: grid[t]["viewer_cost"])
    print(f"threshold picked on DEV: {pick}")
    insp = json.loads((_ROOT / "docs" / "inspector" / "data.json").read_text(encoding="utf-8"))
    ours_pics = {c["clip"]: [(p["label"], p["start"], p["end"]) for p in c["systems"]["ours"]["pictures"]]
                 for c in insp["clips"] if c["split"] == "TEST"}
    base = rows_for(gold, test, pick)
    ours = [S.score_clip(gold[st], ours_pics.get(st, [])) for st in test]
    silence = [S.score_clip(gold[st], []) for st in test]
    res = {"dev_grid": grid, "threshold": pick, "test": {}}
    for name, rr in (("panns_every_detection", base), ("ours", ours), ("silence", silence)):
        a = S.aggregate(rr)
        res["test"][name] = {k: a[k] for k in ("hits", "misses", "visible", "cross", "phantom", "P", "R", "F1", "viewer_cost")}
    for key in ("F1", "viewer_cost", "P", "R"):
        d, lo, hi, _ = S.paired_ci(ours, base, key)
        res["test"][f"ours_minus_panns_{key}"] = [d, lo, hi]
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name in ("panns_every_detection", "ours", "silence"):
        a = res["test"][name]
        print(f"TEST {name:22s} hits {a['hits']}/{a['hits'] + a['misses']} wrong {a['visible'] + a['cross'] + a['phantom']} "
              f"P {a['P']:.2f} R {a['R']:.2f} F1 {a['F1']:.3f} cost {a['viewer_cost']:.2f}")
    for key in ("F1", "viewer_cost", "P", "R"):
        d, lo, hi = res["test"][f"ours_minus_panns_{key}"]
        print(f"TEST ours - PANNs {key:11s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
