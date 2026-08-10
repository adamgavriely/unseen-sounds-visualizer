"""Screen candidate clips for the 'heard but not seen' property.

For each clip it runs the real pipeline analysis (PANNs detection + CLIP
visibility + the gate) WITHOUT retrieval/compositing, and reports which detected
non-speech sounds are OFF-SCREEN (source not visible). A clip is:
  - 'complex'  -> has >=1 off-screen sound (an audio-visual mismatch)  [what we want]
  - 'control'  -> sounds present but all sources visible               [negative control]
  - 'empty'    -> no salient non-speech sound

This is a candidate FILTER, not ground truth (labels are still human-set in
curate.py). Use it to keep the complex scenes and discard single-subject clips.

Usage:
    python -m benchmark.screen                 # screen every clip in data/input/benchmark/
    python -m benchmark.screen market_iwakuni station_kwasa1
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.stage1_audio_extraction import extract_audio
from src.stage4_audio_event_detection import detect_events
from src.stage2_video_understanding import analyze_video
from src.stage5_cross_modal_analysis import plan_augmentations

BENCH = _ROOT / "data" / "input" / "benchmark"


def screen_clip(path: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(path, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                               min_dur=config.AED_MIN_DUR)
        scene = analyze_video(path, num_frames=config.NUM_FRAMES, model=config.VIDEO_MODEL,
                              device=config.DEVICE, threshold=config.VISIBILITY_THRESHOLD)
        specs = plan_augmentations(scene, [], events, gate_enabled=True,
                                   display_threshold=config.DISPLAY_THRESHOLD)
    offscreen = [s.event_label for s in specs if s.augment]
    salient = [s.event_label for s in specs]
    verdict = "complex" if offscreen else ("control" if salient else "empty")
    return {"clip": path.stem, "duration": round(media.duration, 1),
            "visible": scene.visible_entities, "offscreen": offscreen, "verdict": verdict}


def main() -> None:
    ids = sys.argv[1:]
    if ids:
        clips = [next(BENCH.glob(i + ".*")) for i in ids]
    else:
        clips = sorted(p for p in BENCH.glob("*")
                       if p.suffix in (".webm", ".ogv", ".mp4") and "_raw" not in p.stem)
    rows = []
    for f in clips:
        try:
            r = screen_clip(f)
        except Exception as e:
            r = {"clip": f.stem, "verdict": "ERROR", "offscreen": [], "visible": [],
                 "duration": 0, "error": str(e)}
        rows.append(r)
        tag = {"complex": "* COMPLEX", "control": "  control", "empty": "  empty",
               "ERROR": "! ERROR"}[r["verdict"]]
        print(f"{tag}  {r['clip']:20} visible={r.get('visible')}  off-screen={r.get('offscreen')}")
    n = sum(1 for r in rows if r["verdict"] == "complex")
    print(f"\n{n}/{len(rows)} clips are COMPLEX (have off-screen sound).")


if __name__ == "__main__":
    main()
