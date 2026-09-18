"""Slice B of the gold set: AudioSet-Strong evaluation clips with a consequential sound
buried under speech or music (docs/prereg_v4.md; Fable review 2026-09-18).

Why: the main slice's "sounds + times" pre-fill comes from BEATs, the model under test, so
annotators will tend to confirm BEATs and miss exactly the buried sounds. AudioSet-Strong
(Hershey et al. 2021) has HUMAN start/end times to 0.1 s for 456 classes on 10-s YouTube
clips, and its evaluation split is held out from BEATs, PretrainedSED and FLAM. Here only
the visibility / "draw?" judgement and the sentence are left to the annotator.

Selection (fixed before any download): an event of a consequential class (CONSEQUENTIAL,
mapped through the pipeline's families) at least 0.5 s long, of which at least half is
covered by a Speech or Music span in the same clip; at most PER_FAMILY clips per family;
deterministic in SEED. Videos: yt-dlp, the labelled 10 s only, <= 480p, into
data/input/audioset_strong/<segment_id>.mp4 (clips no longer on YouTube are skipped and
listed). Output: benchmark/gold/audioset_slice.json with every event of every fetched
clip (display name, start, end, masked, consequential).

    python benchmark/gold/audioset_slice.py --n 150            # select + download (needs internet)
    python benchmark/gold/audioset_slice.py --n 150 --dry      # selection only
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.labels import SPEECH_LABELS, is_music, canonical

LABELS = _ROOT / "data" / "audioset_strong_labels"
VIDEOS = _ROOT / "data" / "input" / "audioset_strong"
OUT = _ROOT / "benchmark" / "gold" / "audioset_slice.json"
SEED = 7
PER_FAMILY = 12
MIN_DUR = 0.5
MASK_FRAC = 0.5

# consequential for a deaf viewer: safety, someone/something demanding attention, a
# thing that happened. Ontology display names; matched by the pipeline's family
# (labels.canonical) so sub-labels count ("Ambulance (siren)" -> Siren).
CONSEQUENTIAL = {
    "Siren", "Civil defense siren", "Vehicle horn, car horn, honking", "Air horn, truck horn", "Train horn",
    "Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Car alarm", "Alarm clock", "Buzzer", "Doorbell",
    "Telephone", "Telephone bell ringing", "Ringtone", "Dog", "Bark", "Baby cry, infant cry",
    "Glass", "Shatter", "Gunshot, gunfire", "Explosion", "Fireworks", "Thunder", "Helicopter",
    "Knock", "Slam", "Crying, sobbing", "Cat", "Meow", "Bicycle bell", "Train", "Train whistle",
}


def load_labels():
    names = {}
    with (LABELS / "mid_to_display_name.tsv").open(encoding="utf-8") as f:
        for mid, name in csv.reader(f, delimiter="\t"):
            names[mid] = name
    clips = defaultdict(list)
    with (LABELS / "audioset_eval_strong.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            clips[r["segment_id"]].append((names.get(r["label"], r["label"]), float(r["start_time_seconds"]), float(r["end_time_seconds"])))
    return clips


def is_cover(name: str) -> bool:
    return name in SPEECH_LABELS or is_music(name)


def select(clips, n: int):
    random.seed(SEED)
    pool = defaultdict(list)
    for seg, events in clips.items():
        covers = [(s, e) for name, s, e in events if is_cover(name)]
        if not covers:
            continue
        for name, s, e in events:
            fam = canonical(name)
            if (name in CONSEQUENTIAL or fam in CONSEQUENTIAL) and e - s >= MIN_DUR:
                covered = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in covers)
                if covered >= MASK_FRAC * (e - s):
                    pool[fam].append(seg)
                    break
    chosen = []
    for fam, segs in sorted(pool.items()):
        segs = sorted(set(segs)); random.shuffle(segs)
        chosen += [(fam, s) for s in segs[:PER_FAMILY]]
    random.shuffle(chosen)
    return chosen[:n], {f: len(set(v)) for f, v in pool.items()}


def fetch(seg: str) -> bool:
    ytid, start_ms = seg.rsplit("_", 1)
    start = int(start_ms) / 1000.0
    dst = VIDEOS / f"{seg}.mp4"
    if dst.exists():
        return True
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bv*[height<=480]+ba/b[height<=480]/b",
           "--download-sections", f"*{start:.1f}-{start + 10:.1f}", "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", str(dst), f"https://www.youtube.com/watch?v={ytid}"]
    try:
        subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    except Exception:
        return False
    return dst.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    clips = load_labels()
    chosen, pool = select(clips, a.n)
    print(f"[slice] {len(clips)} eval clips; masked consequential per family: {pool}")
    print(f"[slice] chosen {len(chosen)}")
    if a.dry:
        return
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out, missing = [], []
    for i, (fam, seg) in enumerate(chosen, 1):
        if not fetch(seg):
            missing.append(seg); continue
        events = clips[seg]
        covers = [(s, e) for name, s, e in events if is_cover(name)]
        recs = []
        for name, s, e in events:
            covered = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in covers)
            recs.append({"label": name, "start": round(s, 2), "end": round(e, 2),
                         "masked": bool(not is_cover(name) and e - s > 0 and covered >= MASK_FRAC * (e - s)),
                         "consequential": bool(name in CONSEQUENTIAL or canonical(name) in CONSEQUENTIAL)})
        out.append({"id": seg, "family": fam, "src": f"../../data/input/audioset_strong/{seg}.mp4", "duration": 10.0,
                    "events": sorted(recs, key=lambda r: r["start"])})
        if i % 10 == 0:
            print(f"[slice] {i}/{len(chosen)}: {len(out)} fetched, {len(missing)} gone", flush=True)
            OUT.write_text(json.dumps({"clips": out, "missing": missing, "selection": {"seed": SEED, "per_family": PER_FAMILY, "min_dur": MIN_DUR, "mask_frac": MASK_FRAC}}, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps({"clips": out, "missing": missing, "selection": {"seed": SEED, "per_family": PER_FAMILY, "min_dur": MIN_DUR, "mask_frac": MASK_FRAC}}, indent=1), encoding="utf-8")
    print(f"[slice] done: {len(out)} clips, {len(missing)} no longer available -> {OUT}")


if __name__ == "__main__":
    main()
