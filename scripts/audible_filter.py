"""Keep only clips where the target sound is CLEARLY AUDIBLE and not drowned.

Diagnosis (2026-08-24). After the OWLv2 pre-screen, Adam's tags on the surviving
clips were still ~0-13% useful, and the reason was not visibility: 9 of 31 came back
no_ambient and 9 more came back "I don't know". Checking the audio directly showed
the real failure -- the sound the clip was selected for is often faint, spurious, or
buried under speech/music, so there is simply nothing to judge.

Every previous filter asked "is the source on screen?". None asked the prior
question: "is there a clear ambient sound here at all?" This one does.

A clip survives only if:
  1. the target sound is detected in the ACTUAL CUT at >= MIN_CONF (not merely
     annotated somewhere in the source video -- segment timings drift, and a
     sound annotated at second 30 can be inaudible in the window we kept);
  2. it is not drowned: target confidence must be within reach of the loudest
     speech/music in the clip, otherwise a DHH viewer gets dialogue, not ambience;
  3. OWLv2 did not find its source on screen (reuses benchmark/prescreen_owl.json).

The label mapping matters: UnAV names ("fire truck siren") are not PANNs names, and
a naive lookup scored 0.00 on clips where PANNs clearly reports "Fire engine, fire
truck (siren)". ACCEPT lists the PANNs labels that count as each target.

Usage:
    python scripts/audible_filter.py            # dry run
    python scripts/audible_filter.py --apply    # move inaudible clips out
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.labels import canonical, is_music

BENCH = ROOT / "data" / "input" / "benchmark"
UNS, DROP = BENCH / "unsorted", BENCH / "_dropped"
LOG = ROOT / "benchmark" / "sources_batch3.json"
PRE = ROOT / "benchmark" / "prescreen_owl.json"
CACHE = ROOT / "benchmark" / "audible.json"

MIN_CONF = 0.25          # target must be clearly present, not a trace
DROWN_RATIO = 0.60       # target must reach this fraction of the speech/music level

# target label (UnAV or AudioSet) -> PANNs labels that count as that sound
ACCEPT = {
    "ambulance siren": ["Ambulance (siren)", "Emergency vehicle", "Siren"],
    "police car siren": ["Police car (siren)", "Emergency vehicle", "Siren"],
    "fire truck siren": ["Fire engine, fire truck (siren)", "Emergency vehicle", "Siren"],
    "car passing by": ["Car", "Vehicle", "Traffic noise, roadway noise",
                       "Motor vehicle (road)", "Car passing by"],
    "vehicle honking": ["Vehicle horn, car horn, honking", "Toot", "Car"],
    "driving buses": ["Bus", "Vehicle", "Motor vehicle (road)"],
    "driving motorcycle": ["Motorcycle", "Vehicle", "Accelerating, revving, vroom"],
    "engine knocking": ["Engine", "Idling", "Medium engine (mid frequency)", "Vehicle"],
    "skidding": ["Skidding", "Tire squeal", "Car"],
    "train horning": ["Train horn", "Train whistle", "Train", "Rail transport"],
    "train wheels squealing": ["Train wheels squealing", "Train", "Rail transport"],
    "helicopter": ["Helicopter"], "airplane flyby": ["Fixed-wing aircraft, airplane",
                                                     "Aircraft", "Jet engine"],
    "dog barking": ["Bark", "Dog", "Bow-wow"], "dog howling": ["Howl", "Dog", "Bark"],
    "cat meowing": ["Meow", "Cat"], "sheep bleating": ["Bleat", "Sheep", "Goat"],
    "bull bellowing": ["Moo", "Cattle, bovinae"], "horse clip-clop": ["Clip-clop", "Horse"],
    "bird chirping": ["Bird", "Chirp, tweet", "Bird vocalization, bird call, bird song"],
    "sea waves": ["Waves, surf", "Ocean", "Water"], "water burbling": ["Stream", "Gurgling", "Water"],
    "raining": ["Rain", "Raindrop", "Rain on surface"],
    "thunder": ["Thunder", "Thunderstorm"],
    "church bell ringing": ["Church bell", "Bell", "Chime"],
    "fireworks banging": ["Fireworks", "Firecracker", "Explosion"],
    "machine gun shooting": ["Machine gun", "Gunshot, gunfire", "Artillery fire"],
    "chainsawing trees": ["Chainsaw"], "lawn mowing": ["Lawn mower", "Chainsaw"],
    "hammering nails": ["Hammer"], "vacuum cleaner cleaning floors": ["Vacuum cleaner"],
    "hair dryer drying": ["Hair dryer"], "telephone bell ringing": ["Telephone bell ringing",
                                                                   "Telephone", "Ringtone"],
    "people crowd": ["Crowd", "Hubbub", "Cheering"], "people cheering": ["Cheering", "Crowd"],
    "people clapping": ["Applause", "Clapping"], "people shouting": ["Shout", "Yell", "Crowd"],
    "people running": ["Run", "Walk, footsteps"],
}


def accepted_labels(target: str) -> list[str]:
    if target in ACCEPT:
        return ACCEPT[target]
    c = canonical(target)
    return [target, c]


def main(apply: bool = False):
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events

    pre = json.loads(PRE.read_text(encoding="utf-8")) if PRE.exists() else {}
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    meta = {e["file"]: e for e in log}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    clips = [p for p in sorted(UNS.glob("*"))
             if p.suffix.lower() in (".mp4", ".webm", ".ogv")]

    verdict, moved = Counter(), []
    for i, p in enumerate(clips, 1):
        if p.name in cache:
            v = cache[p.name]
        else:
            m = meta.get(p.name, {})
            target = (pre.get(p.name, {}).get("target")
                      or m.get("secondary_event") or m.get("event_label"))
            if not target:
                cache[p.name] = v = {"verdict": "no target", "conf": 0.0}
            else:
                try:
                    with tempfile.TemporaryDirectory() as td:
                        media = extract_audio(p, Path(td) / "a.wav", config.SAMPLE_RATE)
                        evs = detect_events(Path(media.wav_path), threshold=0.05,
                                            min_dur=0.1)
                except Exception as e:
                    cache[p.name] = v = {"verdict": f"error {type(e).__name__}", "conf": 0.0}
                    evs = None
                if evs is not None:
                    ok = set(accepted_labels(target))
                    conf = max((e.confidence for e in evs if e.label in ok), default=0.0)
                    loud = max((e.confidence for e in evs
                                if e.label.startswith(("Speech", "Male speech", "Female speech"))
                                or is_music(e.label)), default=0.0)
                    if conf < MIN_CONF:
                        vd = "inaudible"
                    elif loud > 0 and conf < loud * DROWN_RATIO:
                        vd = "drowned by speech/music"
                    else:
                        vd = "KEEP"
                    cache[p.name] = v = {"verdict": vd, "target": target,
                                         "conf": round(conf, 3), "loud": round(loud, 3)}
            CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        verdict[v["verdict"]] += 1
        print(f"  [{i}/{len(clips)}] {p.name[:34]:34} {str(v.get('target'))[:18]:18} "
              f"conf={v.get('conf',0):.2f} vs speech/music {v.get('loud',0):.2f}  "
              f"{v['verdict']}", flush=True)
        if apply and v["verdict"] != "KEEP":
            shutil.move(str(p), str(DROP / p.name))
            moved.append(p.name)

    print("\n" + "=" * 58)
    for k, n in verdict.most_common():
        print(f"  {n:4}  {k}")
    print(f"\nqueue after: {len(clips) - len(moved)}"
          f"{'' if apply else '   (dry run: pass --apply)'}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
