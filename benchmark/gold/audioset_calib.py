"""The detector CALIBRATION set: a random sample of AudioSet-Strong evaluation clips, disjoint
from gold slice B, used only to choose each detector's bar (matched false-alarm rule) so that
slice B stays an untouched check (docs/prereg_v4.md, "Calibration on AudioSet-Strong").
Random, not filtered: it should look like ordinary YouTube audio, not like the test slice.

    python benchmark/gold/audioset_calib.py --n 320      # select + download (needs internet)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold.audioset_slice import load_labels, is_cover, CONSEQUENTIAL, fetch as _fetch, VIDEOS as SLICE_VIDEOS
from src.labels import canonical
import benchmark.gold.audioset_slice as S

VIDEOS = _ROOT / "data" / "input" / "audioset_calib"
OUT = _ROOT / "benchmark" / "gold" / "audioset_calib.json"
SEED = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    a = ap.parse_args()
    clips = load_labels()
    slice_b = json.loads((_ROOT / "benchmark" / "gold" / "audioset_slice.json").read_text(encoding="utf-8"))
    used = {c["id"] for c in slice_b["clips"]} | set(slice_b.get("missing", []))
    pool = sorted(s for s in clips if s not in used)
    random.seed(SEED); random.shuffle(pool)
    chosen = pool[:a.n]
    print(f"[calib] {len(pool)} candidate clips, {len(chosen)} chosen (seed {SEED}), disjoint from slice B")
    VIDEOS.mkdir(parents=True, exist_ok=True)
    S.VIDEOS = VIDEOS                      # fetch() writes into this folder
    out, missing = [], []
    for i, seg in enumerate(chosen, 1):
        if not _fetch(seg):
            missing.append(seg); continue
        events = clips[seg]
        covers = [(s, e) for name, s, e in events if is_cover(name)]
        recs = []
        for name, s, e in events:
            covered = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in covers)
            recs.append({"label": name, "start": round(s, 2), "end": round(e, 2),
                         "masked": bool(not is_cover(name) and e - s > 0 and covered >= 0.5 * (e - s)),
                         "consequential": bool(name in CONSEQUENTIAL or canonical(name) in CONSEQUENTIAL)})
        out.append({"id": seg, "src": f"../../data/input/audioset_calib/{seg}.mp4", "duration": 10.0, "events": sorted(recs, key=lambda r: r["start"])})
        if i % 20 == 0:
            print(f"[calib] {i}/{len(chosen)}: {len(out)} fetched, {len(missing)} gone", flush=True)
            OUT.write_text(json.dumps({"clips": out, "missing": missing, "seed": SEED}, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps({"clips": out, "missing": missing, "seed": SEED}, indent=1), encoding="utf-8")
    print(f"[calib] done: {len(out)} clips, {len(missing)} gone -> {OUT}")


if __name__ == "__main__":
    main()
