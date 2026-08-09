"""Assisted benchmark curation helper.

For one candidate clip it: (optionally downloads and) trims to a 10-20 s window,
runs Stage 1 audio extraction + Stage 4 PANNs sound-event detection (with the
timeline plot), and extracts keyframes at each detected event's peak time plus a
few evenly-spaced frames. It writes a per-clip REVIEW BUNDLE under
data/work/benchmark/<clip_id>/ (frames + events + plot) and a DRAFT manifest
entry with the detected events pre-filled and ``source_visible`` left null.

A human (assisted-labeling workflow) then inspects the frames, sets
``source_visible`` per salient event and the clip ``category``, and the finalised
entry is merged into benchmark/manifest.json (the committed ground truth).

Usage:
    python -m benchmark.curate --input data/input/london_protest.webm \\
        --clip-id london_protest_01 --source wikimedia \\
        --url https://... --license CC-BY-SA-4.0 --attribution "..." \\
        --start 4 --end 20
    python -m benchmark.curate --input https://example.org/clip.webm --clip-id foo ...

Reuses src/stage1_audio_extraction and src/stage4_audio_event_detection; needs
ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

# make the repo root importable whether run as `-m benchmark.curate` or directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.stage1_audio_extraction import extract_audio, media_duration
from src.stage4_audio_event_detection import _infer, _extract_events, plot_timeline
from src.types import AudioEvent
from src.labels import SPEECH_LABELS, SCENE_LABELS, MUSIC_LABELS, is_music, merge_by_label as _merge_by_label

ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "data" / "input" / "benchmark"
WORK_DIR = ROOT / "data" / "work" / "benchmark"
MANIFEST = ROOT / "benchmark" / "manifest.json"
# Label sets + merge helper are shared with the pipeline via src/labels.py.


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "MscProject-research/1.0 (academic; adamgavriely@gmail.com)"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def _trim(src: Path, dest: Path, start: float, end: float) -> None:
    """Accurate re-encode trim to [start, end] seconds."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-to", str(end),
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(dest)],
        check=True, capture_output=True)


def _extract_frame(clip: Path, t: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(clip),
                    "-frames:v", "1", "-q:v", "3", str(dest)],
                   check=True, capture_output=True)


def _frame_times(events: List[AudioEvent], duration: float, every: float = 3.0) -> List[float]:
    """Peak-ish time of each non-speech event (midpoint) + evenly-spaced context frames."""
    times = [round((e.start + e.end) / 2, 1) for e in events if e.label not in SPEECH_LABELS]
    t = 0.0
    while t < duration:
        times.append(round(t, 1))
        t += every
    # de-duplicate times within 0.8 s
    times.sort()
    out: List[float] = []
    for x in times:
        if not out or x - out[-1] >= 0.8:
            out.append(x)
    return out


def curate(input_path: str, clip_id: str, source: str, url: str = "",
           license: str = "", attribution: str = "",
           start: Optional[float] = None, end: Optional[float] = None,
           scenario: str = "", threshold: float = None) -> dict:
    threshold = config.AED_THRESHOLD if threshold is None else threshold
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    work = WORK_DIR / clip_id
    (work / "frames").mkdir(parents=True, exist_ok=True)

    # 1. obtain source
    if input_path.startswith(("http://", "https://")):
        raw = CLIPS_DIR / f"{clip_id}_raw{Path(input_path).suffix or '.webm'}"
        if not raw.exists():
            print(f"downloading {url or input_path} ...")
            _download(input_path, raw)
    else:
        raw = Path(input_path)
        if not raw.exists():
            raise SystemExit(f"input not found: {raw}")

    orig_duration = media_duration(raw)

    # 2. trim to the benchmark window (or keep whole if short)
    if start is not None or end is not None:
        s = start or 0.0
        e = end if end is not None else min(orig_duration, s + 20.0)
        clip = CLIPS_DIR / f"{clip_id}.mp4"
        print(f"trimming to [{s}, {e}] s ...")
        _trim(raw, clip, s, e)
    else:
        clip = raw
        s, e = 0.0, orig_duration

    # 3. audio + detection
    media = extract_audio(clip, work / "audio.wav", config.SAMPLE_RATE)
    framewise, tvec, labels = _infer(Path(media.wav_path), config.DEVICE)
    events = _merge_by_label(
        _extract_events(framewise, tvec, labels, threshold, None, config.AED_MIN_DUR))
    plot_timeline(framewise, tvec, labels, work / "events_plot.png",
                  top_k=config.AED_PLOT_TOP_K, threshold=threshold)

    # 4. keyframes
    times = _frame_times(events, media.duration)
    for t in times:
        _extract_frame(clip, t, work / "frames" / f"t{t:04.1f}.jpg")

    # 5. draft manifest entry (source_visible left null for human review)
    draft_events = []
    for ev in events:
        non_salient = ev.label in SPEECH_LABELS or ev.label in SCENE_LABELS
        draft_events.append({
            "label": ev.label, "start": round(ev.start, 2), "end": round(ev.end, 2),
            "confidence": round(ev.confidence, 3),
            "salient": (not non_salient),        # speech + ambience default to non-salient
            "music": ev.label in MUSIC_LABELS,
            "source_visible": None,               # <-- HUMAN fills this from frames
            "note": "",
        })
    entry = {
        "clip_id": clip_id, "source": source, "url": url,
        "license": license, "attribution": attribution,
        "orig_duration": round(orig_duration, 1), "clip_start": s, "clip_end": e,
        "duration": round(media.duration, 1),
        "scenario": scenario, "category": None,   # <-- HUMAN sets category
        "events": draft_events,
        "gt_should_augment": [],                  # derived once labels are set
    }
    (work / "draft.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    n_ns = sum(1 for e in draft_events if e["salient"])
    print(f"[curate] {clip_id}: {len(events)} events ({n_ns} non-speech), "
          f"{len(times)} frames -> {work}")
    print(f"[curate] review frames + plot, then set source_visible/category in {work/'draft.json'}")
    return entry


def finalize_gt(entry: dict) -> dict:
    """Compute gt_should_augment = salient & not source_visible (call after labeling)."""
    entry["gt_should_augment"] = [
        e["label"] for e in entry["events"]
        if e.get("salient") and e.get("source_visible") is False
    ]
    return entry


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"schema_version": 1, "clips": []}


def save_entry(entry: dict) -> None:
    """Merge a finalised entry into benchmark/manifest.json (replace by clip_id)."""
    m = load_manifest()
    m["clips"] = [c for c in m["clips"] if c["clip_id"] != entry["clip_id"]]
    m["clips"].append(finalize_gt(entry))
    m["clips"].sort(key=lambda c: c["clip_id"])
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[curate] saved {entry['clip_id']} to {MANIFEST} "
          f"({len(m['clips'])} clips total)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="local path or http(s) URL")
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--source", default="user", help="wikimedia|unav100|vggsound|user")
    ap.add_argument("--url", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--attribution", default="")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--scenario", default="", help="ambient|acoustic_event|mixed")
    args = ap.parse_args()
    curate(args.input, args.clip_id, args.source, args.url, args.license,
           args.attribution, args.start, args.end, args.scenario)


if __name__ == "__main__":
    main()
