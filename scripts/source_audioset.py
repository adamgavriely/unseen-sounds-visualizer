"""Source benchmark clips from AudioSet's TEMPORALLY-STRONG subset.

Why this source (decided 2026-08-22, after the keyword-scraped ambience clips proved
weak): AudioSet-strong ships ~1M HUMAN-annotated sound events with exact start/end
times inside 120k 10-second YouTube segments. Compared with keyword search this gives

  * certainty that the sound is present (human label, not our own detector),
  * the exact moment it happens, so the clip can be cut around it,
  * class quotas, so the benchmark is diverse by construction rather than by luck,
  * citable provenance for the thesis,
  * and no circularity: the dataset never says whether the source is VISIBLE, which
    is precisely the judgement the human annotator supplies.

Selection (all criteria are dataset-label based, never model based):
  * the segment contains a DISCRETE narrative event (<= 4 s) from the target classes
    -- the sounds a captioner brackets as [siren wails], [glass shatters];
  * that event does not span the whole segment (it is an event *within* a scene);
  * the segment ALSO carries speech or human activity, which indicates a real scene
    with people present -- the case where an off-screen siren or explosion matters --
    and filters out bare sound-effect uploads;
  * at most one segment per source video.

Usage: python scripts/source_audioset.py [n_wanted] [--per-class N]
"""
from __future__ import annotations

import random
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import screen, _accept, STAGE

DATA = ROOT / "data" / "work" / "audioset"
ARCHIVE = STAGE / "yt_archive.txt"
CLIP_LEN = 20            # seconds of final clip (>= the 15 s admissibility floor)

# Narrative sounds: meaningful when off-screen, and what SDH captions bracket.
# Quotas keep any one class from dominating (Bark alone has 6.7k candidates).
TARGET_CLASSES = {
    "Siren": 8, "Ambulance (siren)": 4, "Police car (siren)": 5,
    "Fire engine, fire truck (siren)": 4, "Civil defense siren": 3,
    "Explosion": 8, "Gunshot, gunfire": 6, "Artillery fire": 3, "Machine gun": 3,
    "Glass": 6, "Shatter": 3, "Breaking": 4, "Smash, crash": 3,
    "Alarm": 5, "Fire alarm": 4, "Car alarm": 4, "Smoke detector, smoke alarm": 3,
    "Thunder": 5, "Thunderstorm": 4,
    "Helicopter": 5, "Train horn": 4, "Church bell": 4,
    "Bark": 6, "Screaming": 4, "Fireworks": 4, "Firecracker": 3,
    "Doorbell": 3, "Knock": 3, "Slam": 3, "Chainsaw": 3, "Jackhammer": 3,
    "Vehicle horn, car horn, honking": 4, "Skidding": 3,
}
# Human presence in the same segment -> a real scene, not a sound-effect upload
CONTEXT_CLASSES = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Child speech, kid speaking",
    "Crowd", "Cheering", "Applause", "Chatter", "Hubbub, speech noise, speech babble",
    "Laughter", "Children shouting", "Walk, footsteps", "Vehicle", "Car",
}


def load_index():
    mid2name = {}
    for ln in open(DATA / "mid_to_name.tsv", encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 2:
            mid2name[p[0]] = p[1]
    segs = defaultdict(list)
    for f in ("eval_strong.tsv", "train_strong.tsv"):
        fp = DATA / f
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as fh:
            next(fh)
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) == 4:
                    segs[p[0]].append((mid2name.get(p[3], p[3]), float(p[1]), float(p[2])))
    return segs


def candidates(segs):
    """Segments meeting every (label-based) inclusion criterion, grouped by class."""
    by_class = defaultdict(list)
    for seg, evs in segs.items():
        names = {n for n, _, _ in evs}
        if not (names & CONTEXT_CLASSES):
            continue                      # no human/scene context -> likely a SFX upload
        for name, st, en in evs:
            if name not in TARGET_CLASSES:
                continue
            dur = en - st
            if dur > 4.0 or dur <= 0.15:  # must be a discrete event
                continue
            if dur > 8.0:
                continue
            by_class[name].append((seg, st, en))
            break
    return by_class


def fetch(seg_id: str, ev_start: float, name: str, label: str) -> bool:
    """seg_id is '<ytid>_<segment_start_ms>'; cut CLIP_LEN around the event."""
    ytid, _, start_ms = seg_id.rpartition("_")
    try:
        seg_start = int(start_ms) / 1000.0
    except ValueError:
        return False
    abs_t = seg_start + ev_start                 # event position in the source video
    cut_from = max(0.0, abs_t - CLIP_LEN * 0.45)  # event ~45% in, reaction after it
    out = STAGE / f"{name}.mp4"
    out.unlink(missing_ok=True)
    subprocess.run(
        ["yt-dlp", f"https://www.youtube.com/watch?v={ytid}",
         "--download-sections", f"*{cut_from:.1f}-{cut_from + CLIP_LEN:.1f}",
         "-f", "mp4[height<=720]/best[height<=720]/best",
         "--force-keyframes-at-cuts", "--download-archive", str(ARCHIVE),
         "--print-to-file", "%(title)s", str(STAGE / f"{name}.meta"),
         "--no-warnings", "-o", str(out)],
        capture_output=True, text=True, timeout=300)
    if not out.exists():
        return False
    why = screen(out)
    if why:
        print(f"    reject ({why})")
        out.unlink(missing_ok=True)
        return False
    meta_f = STAGE / f"{name}.meta"
    title = meta_f.read_text(encoding="utf-8").strip() if meta_f.exists() else ""
    return _accept(out, name, {
        "source": "audioset-strong", "youtube_id": ytid, "segment_id": seg_id,
        "event_label": label, "event_abs_time": round(abs_t, 2), "title": title,
        "license": "research use; not redistributed"})


def main(want: int, per_class_cap: int | None = None):
    print("[audioset] loading strong labels...", flush=True)
    segs = load_index()
    by_class = candidates(segs)
    print(f"[audioset] {sum(len(v) for v in by_class.values()):,} candidate segments "
          f"across {len(by_class)} classes", flush=True)

    rng = random.Random(11)
    seen_videos = set()
    plan = []
    for cls, quota in TARGET_CLASSES.items():
        pool = by_class.get(cls, [])
        rng.shuffle(pool)
        take, n = [], per_class_cap or quota
        for seg, st, en in pool:
            ytid = seg.rpartition("_")[0]
            if ytid in seen_videos:
                continue                  # one clip per source video
            seen_videos.add(ytid)
            take.append((cls, seg, st))
            if len(take) >= n * 3:        # over-plan: many YouTube ids are dead
                break
        plan += take
    rng.shuffle(plan)

    got, tried, per_cls = 0, 0, defaultdict(int)
    for cls, seg, st in plan:
        if got >= want:
            break
        if per_cls[cls] >= TARGET_CLASSES[cls]:
            continue
        tried += 1
        slug = cls.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:22]
        name = f"as_{slug}_{seg.rpartition('_')[0][:8]}"
        print(f"  [{got}/{want}] {cls} <- {seg}", flush=True)
        if fetch(seg, st, name, cls):
            got += 1
            per_cls[cls] += 1
    print(f"\naudioset: accepted {got} of {tried} tried")
    print("per class:", dict(per_cls))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cap = None
    if "--per-class" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--per-class") + 1])
    main(n, cap)
