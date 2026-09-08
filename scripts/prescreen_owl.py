"""Pre-screen the tagging queue with OWLv2 open-vocabulary DETECTION.

Why a detector rather than CLIP/SigLIP: the question is "is the source of THIS sound
inside the frame?" -- an object-presence question. CLIP and SigLIP embed the whole
image, so a small ambulance in a corner is swamped by the rest of the scene. OWLv2
localises a named object and returns a per-object score, which is exactly the right
shape. Checked against the two clips Adam reported as mislabelled: sheep 0.77 and
dog 0.67 (both correctly ON screen), sea 0.54.

We know from the source datasets WHICH sound each clip was selected for (UnAV's
secondary_event, AudioSet's event_label), so the detector is asked one precise
question instead of a vague "what is here".

Verdicts:
  seen      -- the target source is detected -> set aside (the benchmark already has
               a surplus of seen_ambient, and these are what wasted Adam's time)
  mixed?    -- target NOT detected, but some OTHER sound-source object IS visible
               -> the selective-gating case the benchmark is shortest of
  unseen?   -- target not detected and no other source visible

METHOD NOTE for the write-up: this ENRICHES the queue for positives, so the enriched
clips are flagged in the source log and the 209 clips tagged before this filter
existed remain the unbiased sample for headline accuracy. The human tag is always
the ground truth; the filter only decides what is worth looking at.

Usage:
    python scripts/prescreen_owl.py            # dry run, prints the verdicts
    python scripts/prescreen_owl.py --apply    # move "seen" clips out of the queue
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH = ROOT / "data" / "input" / "benchmark"
UNS, DROP = BENCH / "unsorted", BENCH / "_dropped"
LOG = ROOT / "benchmark" / "sources_batch3.json"
CACHE = ROOT / "benchmark" / "prescreen_owl.json"

N_FRAMES = 2            # 2 frames keeps the whole queue under ~1 h on CPU
DETECT_THR = 0.20          # OWLv2 score at which we call the object present

# sound label (from either dataset) -> a natural detection phrase
QUERY = {
    # sirens / emergency
    "ambulance siren": "an ambulance", "police car siren": "a police car",
    "fire truck siren": "a fire truck", "Siren": "an emergency vehicle",
    "Ambulance (siren)": "an ambulance", "Police car (siren)": "a police car",
    "Fire engine, fire truck (siren)": "a fire truck",
    "Civil defense siren": "a warning siren", "Emergency vehicle": "an emergency vehicle",
    # road
    "car passing by": "a car", "vehicle honking": "a car", "driving buses": "a bus",
    "driving motorcycle": "a motorcycle", "skidding": "a car",
    "engine knocking": "a car engine", "auto racing": "a race car",
    "Vehicle": "a car or truck", "Truck": "a truck", "Bus": "a bus",
    "Motorcycle": "a motorcycle", "Traffic noise, roadway noise": "cars on a road",
    "Skidding": "a car", "Vehicle horn, car horn, honking": "a car",
    # rail / air / water
    "train horning": "a train", "train wheels squealing": "a train", "Train": "a train",
    "Train horn": "a train", "helicopter": "a helicopter", "Helicopter": "a helicopter",
    "airplane flyby": "an airplane", "Aircraft": "an airplane",
    "sailing": "a sailing boat", "Boat": "a boat", "sea waves": "the sea",
    "water burbling": "a river or stream", "Water": "water, a river or the sea",
    # animals
    "dog barking": "a dog", "dog howling": "a dog", "Dog": "a dog", "Bark": "a dog",
    "cat meowing": "a cat", "Cat": "a cat", "sheep bleating": "a sheep",
    "Sheep": "a sheep", "bull bellowing": "a cow", "Cattle": "a cow",
    "horse clip-clop": "a horse", "Horse": "a horse", "bird chirping": "a bird",
    "Bird": "a bird", "lions roaring": "a lion",
    # weather / nature
    "raining": "rain falling", "Rain": "rain falling", "thunder": "storm clouds",
    "Thunder": "storm clouds and lightning", "Thunderstorm": "storm clouds",
    # people
    "people crowd": "a crowd of people", "Crowd": "a crowd of people",
    "people cheering": "a crowd of people", "people clapping": "an audience clapping",
    "Applause": "an audience", "people shouting": "people shouting",
    "Screaming": "a person screaming", "people running": "people running",
    "Walk, footsteps": "a person's feet walking",
    # tools / machines
    "chainsawing trees": "a chainsaw", "Chainsaw": "a chainsaw",
    "lawn mowing": "a lawn mower", "Jackhammer": "a jackhammer",
    "hammering nails": "a hammer", "Drill": "a power drill",
    "Power tool": "a power tool", "Sawing": "a saw",
    "vacuum cleaner cleaning floors": "a vacuum cleaner",
    "hair dryer drying": "a hair dryer",
    # events / objects
    "church bell ringing": "a church bell tower", "Church bell": "a church tower",
    "Bell": "a bell", "fireworks banging": "fireworks in the sky",
    "Fireworks": "fireworks", "Firecracker": "fireworks",
    "machine gun shooting": "a person firing a gun", "Gunshot, gunfire": "a gun",
    "Machine gun": "a gun", "Explosion": "an explosion or fire",
    "Glass": "broken glass", "Shatter": "broken glass",
    "telephone bell ringing": "a telephone", "Telephone": "a telephone",
    "Alarm": "an alarm or siren", "Car alarm": "a car", "Fire alarm": "an alarm",
    "Smoke detector, smoke alarm": "a smoke detector",
    # remaining labels seen in the queue
    "Cheering": "a crowd of people", "Boom": "an explosion", "Jet engine": "an airplane",
    "Wind": "trees blowing in wind", "Bang": "an explosion", "Eruption": "an explosion",
    "Fusillade": "a gun", "Artillery fire": "artillery", "Air horn, truck horn": "a truck",
    "Breaking": "broken glass", "Smash, crash": "a crash", "Knock": "a door",
    "Slam": "a door", "Doorbell": "a door", "Creak": "a wooden floor",
    "Squeak": "a door", "Rustling leaves": "trees and leaves", "Stream": "a stream",
    "Waterfall": "a waterfall", "Raindrop": "rain falling", "Wood": "wood or sticks",
    "Run": "people running", "Sneeze": "a person", "Whip": "a whip",
    "Crack": "a broken branch", "Clatter": "falling objects",
    "Splash, splatter": "water splashing", "Thump, thud": "a falling object",
    "Tire squeal": "a car", "Train whistle": "a train", "Ding-dong": "a door",
}


def _label_from_name(name: str) -> str | None:
    """Some clips lost their metadata when two sourcing waves corrupted the log.
    Their filenames encode the sound they were selected for (ly_bark_..., lx_crowd_
    and_applause_...), so recover it from there."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    body = "_".join(parts[1:-1])           # strip source prefix and the video id
    body = body.split("_and_")[0]          # layered names: keep the first sound
    words = body.replace("_", " ").strip()
    for k in QUERY:
        if k.lower() == words or k.lower().split(",")[0] == words:
            return k
    for k in QUERY:                        # loose containment
        if words and words in k.lower():
            return k
    return None
# asked on every clip, to spot a DIFFERENT visible source (-> the mixed case)
CONTEXT_QUERIES = ["a car", "a person", "a crowd of people", "a dog", "a train",
                   "a boat", "water", "an airplane", "a building", "a road"]


def target_query(meta: dict, filename: str = "") -> tuple[str, str] | None:
    lab = meta.get("secondary_event") or meta.get("event_label") or ""
    for part in str(lab).split("+"):
        p = part.strip()
        if p in QUERY:
            return p, QUERY[p]
    k = _label_from_name(filename)
    return (k, QUERY[k]) if k else None


def main(apply: bool = False):
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from src.stage2_video_understanding import _sample_frames

    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    meta = {e["file"]: e for e in log}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    clips = [p for p in sorted(UNS.glob("*"))
             if p.suffix.lower() in (".mp4", ".webm", ".ogv")]

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    mdl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").eval()

    verdicts, moved = Counter(), []
    for i, p in enumerate(clips, 1):
        if p.name in cache:
            v = cache[p.name]
        else:
            tq = target_query(meta.get(p.name, {}), p.name)
            if tq is None:
                cache[p.name] = v = {"verdict": "no target label", "target": None}
            else:
                lab, q = tq
                queries = [q] + [c for c in CONTEXT_QUERIES if c != q]
                best = {}
                for img in _sample_frames(p, N_FRAMES):
                    inp = proc(text=[queries], images=img, return_tensors="pt")
                    with torch.no_grad():
                        out = mdl(**inp)
                    r = proc.post_process_grounded_object_detection(
                        out, threshold=0.08,
                        target_sizes=torch.tensor([[img.height, img.width]]))[0]
                    for sc, lb in zip(r["scores"], r["labels"]):
                        k = queries[int(lb)]
                        best[k] = max(best.get(k, 0.0), float(sc))
                tscore = best.get(q, 0.0)
                others = {k: s for k, s in best.items()
                          if k != q and s >= DETECT_THR and k not in ("a building", "a road")}
                verdict = ("seen" if tscore >= DETECT_THR
                           else "mixed?" if others else "unseen?")
                cache[p.name] = v = {"verdict": verdict, "target": lab, "query": q,
                                     "target_score": round(tscore, 3),
                                     "others": {k: round(s, 2) for k, s in others.items()}}
            CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        verdicts[v["verdict"]] += 1
        print(f"  [{i}/{len(clips)}] {p.name[:36]:36} {str(v.get('target'))[:18]:18} "
              f"score={v.get('target_score', 0):.2f}  {v['verdict']}", flush=True)
        if apply and v["verdict"] == "seen":
            shutil.move(str(p), str(DROP / p.name))
            moved.append(p.name)
        elif apply and p.name in meta:
            meta[p.name]["prescreen"] = "enriched"

    if apply:
        tmp = LOG.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(meta.values()), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, LOG)
    print("\n" + "=" * 58)
    for k, n in verdicts.most_common():
        print(f"  {n:4}  {k}")
    print(f"\nqueue after: {len(clips) - len(moved)}"
          f"{'' if apply else '   (dry run: pass --apply)'}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
