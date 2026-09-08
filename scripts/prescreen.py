"""Pre-screen the tagging queue so Adam mostly sees USEFUL candidates.

The problem this solves: every source converted at only 16-18% useful, so four out
of five clips Adam tagged were seen_ambient or no_ambient -- cases the benchmark
already has in surplus. Human tagging time, not download bandwidth, is the
bottleneck.

The check is precise because the dataset already tells us WHICH sound the clip was
selected for (UnAV's secondary_event, AudioSet's event_label). So instead of asking
the vague "is anything visible", we ask exactly: *is the source of THAT sound on
screen?* -- using SigLIP, which scores each concept independently. Clips whose
target source is plainly visible are set aside; the rest go to the queue.

METHODOLOGICAL NOTE (must appear in the write-up): this ENRICHES the pool for
positives and therefore does not sample the natural distribution. Two safeguards:

  1. the already-tagged clips (209 at the time of writing) were selected WITHOUT any
     model filter and remain the unbiased sample for headline accuracy;
  2. enriched clips are flagged (prescreen: "enriched") in the source log so the
     evaluation can report natural-sample and enriched-sample numbers separately.

The human tag always remains the ground truth; the filter only decides what is worth
Adam's time to look at.

Usage:
    python scripts/prescreen.py            # report only
    python scripts/prescreen.py --apply    # move clearly-visible clips out of the queue
"""
from __future__ import annotations

import json
import sys
import shutil
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.labels import canonical
from src.stage2_video_understanding import VISIBLE_CONCEPTS

BENCH = ROOT / "data" / "input" / "benchmark"
UNS, DROP = BENCH / "unsorted", BENCH / "_dropped"
LOG = ROOT / "benchmark" / "sources_batch3.json"

# dataset label -> our canonical visible concept
UNAV_TO_CONCEPT = {
    "ambulance siren": "Siren", "police car siren": "Siren", "fire truck siren": "Siren",
    "car passing by": "Vehicle", "vehicle honking": "Vehicle", "driving buses": "Vehicle",
    "driving motorcycle": "Vehicle", "skidding": "Vehicle", "engine knocking": "Vehicle",
    "auto racing": "Vehicle", "train horning": "Train", "train wheels squealing": "Train",
    "dog barking": "Dog", "dog howling": "Dog", "cat meowing": "Cat",
    "sheep bleating": "Sheep", "bull bellowing": "Cattle", "horse clip-clop": "Horse",
    "bird chirping": "Bird", "helicopter": "Helicopter", "airplane flyby": "Aircraft",
    "sea waves": "Water", "water burbling": "Water", "raining": "Rain",
    "thunder": "Thunder", "church bell ringing": "Bell", "fireworks banging": "Fireworks",
    "machine gun shooting": "Gunshot", "people crowd": "Crowd", "people cheering": "Crowd",
    "people clapping": "Applause", "lions roaring": "Cat", "sailing": "Boat",
    "chainsawing trees": "Chainsaw", "lawn mowing": "Chainsaw",
}
# a couple of extra concepts the queue needs
EXTRA_CONCEPTS = {"Chainsaw": "a chainsaw or power tool"}


def target_concept(meta: dict) -> str | None:
    """Which visible concept corresponds to the sound this clip was selected for?"""
    lab = meta.get("secondary_event") or meta.get("event_label") or ""
    if not lab:
        return None
    if lab in UNAV_TO_CONCEPT:
        return UNAV_TO_CONCEPT[lab]
    for part in str(lab).split("+"):          # layered names like "Bark+Truck"
        c = canonical(part.strip())
        if c in VISIBLE_CONCEPTS or c in EXTRA_CONCEPTS:
            return c
    return None


def main(apply: bool = False):
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    meta = {e["file"]: e for e in log}
    clips = [p for p in sorted(UNS.glob("*"))
             if p.suffix.lower() in (".mp4", ".webm", ".ogv")]

    from src.stage2_video_understanding.siglip import analyze_video_siglip
    import src.stage2_video_understanding as s2
    s2.VISIBLE_CONCEPTS.update(EXTRA_CONCEPTS)

    verdict, moved = Counter(), []
    for i, p in enumerate(clips, 1):
        m = meta.get(p.name, {})
        concept = target_concept(m)
        if concept is None:
            verdict["no target label - kept"] += 1
            continue
        sc = analyze_video_siglip(p, num_frames=6, model=config.SIGLIP_MODEL,
                                  device=config.DEVICE,
                                  threshold=config.SIGLIP_THRESHOLD,
                                  candidates=[concept])
        score = sc.raw.get("scores", {}).get(concept, -99)
        # clearly on screen -> another seen_ambient, which the benchmark already has
        # in surplus. A comfortable margin above threshold keeps borderline cases,
        # since those are exactly the interesting ones.
        clearly_visible = score >= config.SIGLIP_THRESHOLD + 3.0
        tag = "SET ASIDE (source clearly on screen)" if clearly_visible else "keep"
        verdict[tag] += 1
        print(f"  [{i}/{len(clips)}] {p.name[:38]:38} {concept:10} "
              f"logit={score:6.1f}  {tag}", flush=True)
        if clearly_visible and apply:
            shutil.move(str(p), str(DROP / p.name))
            moved.append(p.name)
        elif apply:
            m["prescreen"] = "enriched"

    if apply:
        LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 60)
    for k, v in verdict.most_common():
        print(f"  {v:4}  {k}")
    print(f"\nqueue after: {len(clips) - len(moved)}"
          f"{'' if apply else '  (dry run -- pass --apply to move)'}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
