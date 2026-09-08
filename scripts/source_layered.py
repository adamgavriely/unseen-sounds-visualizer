"""Source clips with a LAYERED soundscape, from AudioSet's temporally-strong labels.

A second selection rule, complementary to scripts/source_unav.py. Measured yields
were flat at 16-18% across every source, so the differentiator is not WHERE clips
come from but WHICH MOMENT is cut. Two rules are used here:

  "mixed"  -- cut where a SECOND ambient source starts while a first is already
              running. Two distinct sources are audible at once, so the clip can
              only be labelled seen or unseen per-source: exactly the selective-
              gating case the benchmark is short of.
  "unseen" -- cut where an ambient source is audible while SPEECH is also present.
              Dialogue means the camera is on people, so a siren or passing train
              in the same moment is very likely off-frame. (This is the pattern
              behind the film scenes that did work: a mob outside, an air-raid
              siren, a phone ringing elsewhere.)

Both rules use only AudioSet's human annotations, never our own models, so nothing
here biases the gate's evaluation.

Usage: python scripts/source_layered.py [mixed|unseen] [n]
"""
from __future__ import annotations

import random
import subprocess
import sys
import warnings
from collections import defaultdict, Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import screen, _accept, STAGE

DATA = ROOT / "data" / "work" / "audioset"
ARCHIVE = STAGE / "yt_archive.txt"
CLIP_LEN = 18

SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking",
          "Conversation", "Narration, monologue", "Child speech, kid speaking"}
# Incidental infrastructure: nobody films these on purpose, so they tend to be
# off-frame (the lesson from the UnAV per-class yields).
INCIDENTAL = {
    "Siren", "Ambulance (siren)", "Police car (siren)", "Fire engine, fire truck (siren)",
    "Civil defense siren", "Vehicle horn, car horn, honking", "Air horn, truck horn",
    "Train horn", "Train whistle", "Church bell", "Car alarm", "Fire alarm",
    "Smoke detector, smoke alarm", "Alarm", "Motorcycle", "Bus", "Truck",
    "Traffic noise, roadway noise", "Skidding", "Tire squeal", "Thunder",
    "Thunderstorm", "Rain", "Bark", "Dog", "Helicopter", "Jackhammer", "Sawing",
    "Power tool", "Drill", "Lawn mower", "Chainsaw", "Glass", "Shatter",
    "Explosion", "Gunshot, gunfire", "Screaming", "Crowd", "Applause",
}


def load():
    mid2name = {}
    for ln in open(DATA / "mid_to_name.tsv", encoding="utf-8"):
        q = ln.rstrip("\n").split("\t")
        if len(q) == 2:
            mid2name[q[0]] = q[1]
    segs = defaultdict(list)
    for f in ("eval_strong.tsv", "train_strong.tsv"):
        fp = DATA / f
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as fh:
            next(fh)
            for ln in fh:
                q = ln.rstrip("\n").split("\t")
                if len(q) == 4:
                    segs[q[0]].append((mid2name.get(q[3], q[3]), float(q[1]), float(q[2])))
    return segs


def candidates(segs, mode):
    out = []
    for seg, evs in segs.items():
        inc = [(n, s, e) for n, s, e in evs if n in INCIDENTAL]
        if not inc:
            continue
        if mode == "unseen":
            # an incidental sound while someone is speaking -> camera is on the people
            sp = [(s, e) for n, s, e in evs if n in SPEECH]
            if not sp:
                continue
            for n, s, e in inc:
                for ss, se in sp:
                    lo, hi = max(s, ss), min(e, se)
                    if hi - lo >= 1.0:
                        out.append((seg, n, "speech", (lo + hi) / 2))
                        break
                else:
                    continue
                break
        else:
            # two DIFFERENT incidental sources overlapping -> selective-gating case
            for i, (n1, s1, e1) in enumerate(inc):
                for n2, s2, e2 in inc[i + 1:]:
                    if n1 == n2:
                        continue
                    lo, hi = max(s1, s2), min(e1, e2)
                    if hi - lo >= 1.0:
                        out.append((seg, f"{n1}+{n2}", n2, (lo + hi) / 2))
                        break
                else:
                    continue
                break
    return out


def fetch(seg_id, ev_t, name, meta):
    ytid, _, ms = seg_id.rpartition("_")
    try:
        seg_start = int(ms) / 1000.0
    except ValueError:
        return False
    abs_t = seg_start + ev_t
    start = max(0.0, abs_t - CLIP_LEN * 0.45)
    out = STAGE / f"{name}.mp4"
    out.unlink(missing_ok=True)
    subprocess.run(
        ["yt-dlp", f"https://www.youtube.com/watch?v={ytid}",
         "--download-sections", f"*{start:.1f}-{start + CLIP_LEN:.1f}",
         "-f", "mp4[height<=720]/best[height<=720]/best",
         "--force-keyframes-at-cuts", "--download-archive", str(ARCHIVE),
         "--print-to-file", "%(title)s", str(STAGE / f"{name}.meta"),
         "--no-warnings", "-o", str(out)],
        capture_output=True, text=True, timeout=300)
    if not out.exists():
        return False
    meta_f = STAGE / f"{name}.meta"
    if meta_f.exists():
        meta["title"] = meta_f.read_text(encoding="utf-8").strip()
    return _accept(out, name, meta)


def main(mode="unseen", want=40):
    STAGE.mkdir(parents=True, exist_ok=True)
    segs = load()
    cands = candidates(segs, mode)
    print(f"[layered/{mode}] {len(cands)} candidate moments", flush=True)
    random.Random(3).shuffle(cands)
    per, got, tried, seen_vid = Counter(), 0, 0, set()
    prefix = "ly" if mode == "unseen" else "lx"
    for seg, lab, other, t in cands:
        if got >= want:
            break
        ytid = seg.rpartition("_")[0]
        if ytid in seen_vid or per[lab] >= 3:
            continue
        seen_vid.add(ytid)
        tried += 1
        slug = lab.lower().replace(" ", "_").replace(",", "").replace("+", "_and_")[:26]
        name = f"{prefix}_{slug}_{ytid[:7]}"
        print(f"  [{got}/{want}] {lab} (with {other}) {seg}", flush=True)
        if fetch(seg, t, name, {"source": f"audioset-strong/{mode}", "youtube_id": ytid,
                                "segment_id": seg, "event_label": lab, "with": other,
                                "license": "research use; not redistributed"}):
            got += 1
            per[lab] += 1
    print(f"\nlayered/{mode}: accepted {got} of {tried} tried")
    print("classes:", dict(per))


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "unseen"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(m, n)
