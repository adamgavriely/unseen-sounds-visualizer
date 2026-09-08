"""End-to-end FILTERED sourcing: only clips that pass every gate reach the queue.

The lesson from ~350 hand-tagged clips: yield sat at 16-18% no matter which source
was used, because filtering happened after the fact (or not at all). A clip is only
worth Adam's time if all three hold, so all three are now checked BEFORE it is kept:

  1. quality   -- video + audible audio, 16-20 s, moving, colour (scripts/source_batch3)
  2. audibility-- the target sound is actually detected in the CUT at >= 0.25 and is
                  not buried under speech/music. This was the missing one: 19 of 53
                  queued clips had no audible target at all.
  3. visibility-- OWLv2 does not find the sound's source in frame. If it does, the
                  clip is a seen_ambient case, of which the benchmark has a surplus.

Candidates come from the two annotated pools already downloaded (UnAV-100 overlaps
and AudioSet-strong moments), so the target sound and its timing are known from human
annotation rather than guessed.

Cost is ~60 s per candidate, which is the point: spend machine time so Adam's tagging
time is spent almost entirely on clips that count.

Usage: python scripts/source_filtered.py [n_wanted]
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from scripts.source_batch3 import screen, _accept, STAGE
from scripts.audible_filter import accepted_labels, MIN_CONF, DROWN_RATIO
from scripts.prescreen_owl import QUERY as OWL_QUERY, DETECT_THR
from scripts.source_unav import AMBIENT, SPECTACLE, SUBJECT_CLASSES
from src.labels import is_music

ARCHIVE = STAGE / "yt_archive.txt"
UNAV = ROOT / "data" / "work" / "unav" / "annotations.json"
CLIP_LEN = 18


def unav_candidates():
    db = json.loads(UNAV.read_text(encoding="utf-8"))["database"]
    out = []
    for ytid, v in db.items():
        anns = v.get("annotations", [])
        if len(anns) < 2:
            continue
        air = defaultdict(float)
        for a in anns:
            s, e = a["segment"]
            air[a["label"]] += max(0.0, e - s)
        if len(air) < 2:
            continue
        primary = max(air, key=air.get)
        pspans = [a["segment"] for a in anns if a["label"] == primary]
        for a in anns:
            lb = a["label"]
            if lb == primary or lb not in AMBIENT or lb in SPECTACLE:
                continue
            if lb not in OWL_QUERY:
                continue
            s, e = a["segment"]
            for ps, pe in pspans:
                lo, hi = max(s, ps), min(e, pe)
                if hi - lo >= 0.5:
                    bonus = 0 if primary in SUBJECT_CLASSES else 1
                    out.append((bonus, ytid, lb, (lo + hi) / 2, v.get("duration", 0)))
                    break
            else:
                continue
            break
    out.sort(key=lambda c: c[0])
    return [(c[1], c[2], c[3], c[4]) for c in out]


def audible(path: Path, target: str):
    """(conf, loudest speech/music) for the target sound in the actual cut."""
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(path, Path(td) / "a.wav", config.SAMPLE_RATE)
        evs = detect_events(Path(media.wav_path), threshold=0.05, min_dur=0.1)
    ok = set(accepted_labels(target))
    conf = max((e.confidence for e in evs if e.label in ok), default=0.0)
    loud = max((e.confidence for e in evs
                if e.label.startswith(("Speech", "Male speech", "Female speech"))
                or is_music(e.label)), default=0.0)
    return conf, loud


def visible(path: Path, target: str, mdl, proc):
    import torch
    from src.stage2_video_understanding import _sample_frames
    q = OWL_QUERY.get(target)
    if not q:
        return 0.0
    best = 0.0
    for img in _sample_frames(path, 2):
        inp = proc(text=[[q]], images=img, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**inp)
        r = proc.post_process_grounded_object_detection(
            out, threshold=0.05,
            target_sizes=torch.tensor([[img.height, img.width]]))[0]
        for sc in r["scores"]:
            best = max(best, float(sc))
    return best


def main(want=80):
    STAGE.mkdir(parents=True, exist_ok=True)
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    mdl = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").eval()

    cands = unav_candidates()
    random.Random(17).shuffle(cands[:0])          # keep the person-primary ordering
    print(f"[filtered] {len(cands)} annotated candidates available", flush=True)

    got, tried, per = 0, 0, Counter()
    reasons = Counter()
    for ytid, target, mid, dur in cands:
        if got >= want:
            break
        if per[target] >= 6:
            continue
        tried += 1
        name = f"fx_{target.replace(' ', '_')[:18]}_{ytid[:8]}"
        raw = STAGE / f"{name}.mp4"
        raw.unlink(missing_ok=True)
        start = max(0.0, min(mid - CLIP_LEN * 0.45, max(0.0, dur - CLIP_LEN)))
        subprocess.run(
            ["yt-dlp", f"https://www.youtube.com/watch?v={ytid}",
             "--download-sections", f"*{start:.1f}-{start + CLIP_LEN:.1f}",
             "-f", "mp4[height<=720]/best[height<=720]/best",
             "--force-keyframes-at-cuts", "--download-archive", str(ARCHIVE),
             "--no-warnings", "-o", str(raw)],
            capture_output=True, text=True, timeout=300)
        if not raw.exists():
            reasons["no download"] += 1
            continue
        why = screen(raw)
        if why:
            reasons[f"quality: {why.split()[0]}"] += 1
            raw.unlink(missing_ok=True)
            continue
        try:
            conf, loud = audible(raw, target)
        except Exception:
            reasons["audio error"] += 1
            raw.unlink(missing_ok=True)
            continue
        if conf < MIN_CONF:
            reasons["inaudible"] += 1
            raw.unlink(missing_ok=True)
            continue
        if loud > 0 and conf < loud * DROWN_RATIO:
            reasons["drowned"] += 1
            raw.unlink(missing_ok=True)
            continue
        vis = visible(raw, target, mdl, proc)
        if vis >= DETECT_THR:
            reasons["source on screen"] += 1
            raw.unlink(missing_ok=True)
            continue
        if _accept(raw, name, {"source": "unav-100/filtered", "youtube_id": ytid,
                               "secondary_event": target, "audible_conf": round(conf, 3),
                               "owl_score": round(vis, 3), "prescreen": "enriched",
                               "license": "research use; not redistributed"}):
            got += 1
            per[target] += 1
            print(f"    KEEP {got}/{want}  {target}  audible={conf:.2f} "
                  f"owl={vis:.2f}", flush=True)
    print(f"\nfiltered: kept {got} of {tried} tried")
    print("rejections:", dict(reasons.most_common()))
    print("classes:", dict(per))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
