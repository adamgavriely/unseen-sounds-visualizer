"""Source benchmark clips from UnAV-100 (a dataset named in the project proposal, sec 5.3).

Why this source: UnAV-100 annotates UNTRIMMED video with every sound event and its
exact time span, so 4,366 of its 10,790 videos carry two or more DIFFERENT concurrent
event types. That is precisely the structure the gate is meant to reason about --
one sound belongs to what the camera is pointed at, another comes from somewhere
else -- so we can aim at the moment they overlap instead of guessing from search
keywords (which reliably returned scenes that SHOW the sound source).

Selection, entirely from dataset labels (never from our own models):
  * pick the PRIMARY event = the class occupying the most time (what the video is
    "about", hence almost certainly on camera);
  * pick a SECONDARY environmental event of a different class that overlaps it --
    the candidate "heard but not seen" sound;
  * cut a 16-20 s window (proposal sec 5: 10-20 s) placed so the overlap sits
    naturally inside with scene context before and after. The clip must let a
    viewer judge whether an augmentation HELPS, so it is not trimmed tight to
    the event.

Adam still assigns the final seen/unseen tag; the dataset never claims visibility.

Usage: python scripts/source_unav.py [n_wanted]
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import screen, _accept, STAGE

ANN = ROOT / "data" / "work" / "unav" / "annotations.json"
ARCHIVE = STAGE / "yt_archive.txt"

# Environmental / acoustic-event classes worth depicting when off-screen.
# Speech, singing and instruments are excluded: out of scope (captions cover speech,
# decision A6 covers music).
AMBIENT = {
    "airplane flyby", "ambulance siren", "auto racing", "bird chirping",
    "bull bellowing", "car passing by", "cat meowing", "chainsawing trees",
    "church bell ringing", "dog barking", "dog howling", "driving buses",
    "driving motorcycle", "engine knocking", "fire truck siren", "fireworks banging",
    "frog croaking", "hair dryer drying", "hammering nails", "helicopter",
    "horse clip-clop", "lawn mowing", "lions roaring", "machine gun shooting",
    "people cheering", "people clapping", "people crowd", "people running",
    "people shouting", "police car siren", "raining", "sea waves", "sheep bleating",
    "skidding", "telephone bell ringing", "thunder", "train horning",
    "train wheels squealing", "vacuum cleaner cleaning floors", "vehicle honking",
    "water burbling", "wind noise", "sailing", "skateboarding",
}
MIN_LEN, MAX_LEN = 16, 20          # proposal sec 5: 10-20 s clips

# Measured on Adam's tags (2026-08-23): SPECTACLE sounds fail as off-screen
# candidates because the video exists to film them -- helicopter 0/4 useful,
# airplane flyby 0/3, sea waves 0/2, sheep 0/2, all tagged seen_ambient. INCIDENTAL
# urban infrastructure succeeds -- nobody points a camera at the motorbike that
# happens to pass (2/2), the siren down the street (2/3), the church bell (1/1),
# the train horn (1/1). Sample only the incidental ones.
HIGH_VALUE = {
    "ambulance siren", "police car siren", "fire truck siren",
    "train horning", "train wheels squealing", "church bell ringing",
    "vehicle honking", "car passing by", "driving motorcycle", "driving buses",
    "skidding", "dog barking", "dog howling", "thunder", "raining",
    "lawn mowing", "chainsawing trees", "telephone bell ringing", "engine knocking",
    "hammering nails", "vacuum cleaner cleaning floors", "hair dryer drying",
}
# Filmed on purpose -> the source is almost always the subject in frame. Excluded.
SPECTACLE = {
    "helicopter", "airplane flyby", "sea waves", "sheep bleating", "fireworks banging",
    "lions roaring", "bull bellowing", "machine gun shooting", "auto racing",
    "sailing", "skateboarding", "frog croaking", "cat meowing", "bird chirping",
    "horse clip-clop", "water burbling",
}
# The camera follows a PERSON in these, so a co-occurring street sound is off-frame.
SUBJECT_CLASSES = {
    "man speaking", "woman speaking", "kid speaking", "people laughing",
    "male singing", "female singing", "child singing", "people eating",
    "playing acoustic guitar", "playing piano", "playing violin", "playing drum kit",
    "people whispering", "typing on computer keyboard", "people coughing",
    "playing tennis", "playing badminton", "playing table tennis", "rope skipping",
    "basketball bounce", "tap dancing", "people slapping", "baby babbling",
}
# Crowd reactions are usually in the same shot as the thing being reacted to,
# so they rarely give a clean off-screen case. Capped, not excluded.
LOW_VALUE = {"people clapping", "people crowd", "people cheering", "people running",
             "people shouting", "people laughing"}


def load():
    return json.load(open(ANN, encoding="utf-8"))["database"]


def pick_overlap(v):
    """Return (secondary_label, primary_label, overlap_mid) or None.

    PRIMARY  = class with the most total airtime (the video's subject, on camera).
    SECONDARY= a different AMBIENT class overlapping it (the off-screen candidate).
    """
    anns = v.get("annotations", [])
    if len(anns) < 2:
        return None
    airtime = {}
    for a in anns:
        s, e = a["segment"]
        airtime[a["label"]] = airtime.get(a["label"], 0.0) + max(0.0, e - s)
    if len(airtime) < 2:
        return None
    primary = max(airtime, key=airtime.get)
    prim_spans = [a["segment"] for a in anns if a["label"] == primary]
    best = None
    for a in anns:
        lb = a["label"]
        if lb == primary or lb not in AMBIENT or lb in SPECTACLE:
            continue
        s, e = a["segment"]
        for ps, pe in prim_spans:
            lo, hi = max(s, ps), min(e, pe)
            if hi - lo >= 0.5:                     # genuinely concurrent
                mid = (lo + hi) / 2
                # brief secondaries are the most incidental, hence likeliest off-camera;
                # a person-centred primary is a further bonus (the camera is on them)
                score = (e - s) - (4.0 if primary in SUBJECT_CLASSES else 0.0)
                if best is None or score < best[0]:
                    best = (score, lb, primary, mid)
    return best[1:] if best else None


def fetch(ytid, mid, dur, name, meta):
    length = random.randint(MIN_LEN, MAX_LEN)
    start = max(0.0, min(mid - length * 0.45, max(0.0, dur - length)))
    out = STAGE / f"{name}.mp4"
    out.unlink(missing_ok=True)
    subprocess.run(
        ["yt-dlp", f"https://www.youtube.com/watch?v={ytid}",
         "--download-sections", f"*{start:.1f}-{start + length:.1f}",
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


def main(want=60):
    STAGE.mkdir(parents=True, exist_ok=True)
    db = load()
    cands = []
    for ytid, v in db.items():
        got = pick_overlap(v)
        if got:
            sec, prim, mid = got
            cands.append((ytid, sec, prim, mid, v.get("duration", 0)))
    print(f"[unav] {len(cands)} videos with an ambient event overlapping the subject",
          flush=True)
    random.Random(5).shuffle(cands)
    # high-value (typically off-screen) classes first
    cands.sort(key=lambda c: 0 if c[1] in HIGH_VALUE else 1 if c[1] not in LOW_VALUE else 2)

    from collections import Counter
    per_class, got, tried = Counter(), 0, 0
    for ytid, sec, prim, mid, dur in cands:
        if got >= want:
            break
        cap = 4 if sec in HIGH_VALUE else 2
        if per_class[sec] >= cap:          # keep the class mix diverse
            continue
        tried += 1
        slug = sec.replace(" ", "_")[:18]
        name = f"un_{slug}_{ytid[:8]}"
        print(f"  [{got}/{want}] {sec}  (over '{prim}')  {ytid}", flush=True)
        if fetch(ytid, mid, dur, name, {
                "source": "unav-100", "youtube_id": ytid,
                "secondary_event": sec, "primary_event": prim,
                "overlap_time": round(mid, 2),
                "license": "research use; not redistributed"}):
            got += 1
            per_class[sec] += 1
    print(f"\nunav: accepted {got} of {tried} tried")
    print("classes:", dict(per_class))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
