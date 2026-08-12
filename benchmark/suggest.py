"""Compute a SUGGESTED tag + one-line reason for each clip, for the tagger to show.

Runs the real analysis (PANNs detection + CLIP visibility + the gate) and writes
benchmark/suggestions.json: {"<folder>/<file>": {"suggest": <tag>, "reason": "..."}}.
The tagger displays this after "not tagged" as a hint (you still decide).

Usage:
    python -m benchmark.suggest                 # all clips (default: unsorted/ only)
    python -m benchmark.suggest --all           # every folder
"""
from __future__ import annotations

import json
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
from src.labels import is_salient_nonspeech, consolidate_families

BENCH = _ROOT / "data" / "input" / "benchmark"
OUT = _ROOT / "benchmark" / "suggestions.json"
EXT = {".webm", ".ogv", ".mp4"}


def suggest_clip(path: Path):
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(path, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                               min_dur=config.AED_MIN_DUR)
        scene = analyze_video(path, num_frames=config.NUM_FRAMES, model=config.VIDEO_MODEL,
                              device=config.DEVICE, threshold=config.VISIBILITY_THRESHOLD)
    salient = [e for e in consolidate_families([x for x in events if is_salient_nonspeech(x.label)])
               if e.confidence >= config.DISPLAY_THRESHOLD]
    visible = {v.lower() for v in scene.visible_entities}
    offscreen = [e.label for e in salient if e.label.lower() not in visible]
    seen = [e.label for e in salient if e.label.lower() in visible]

    def j(xs):
        return ", ".join(xs[:2])
    if offscreen and seen:
        return {"suggest": "mixed",
                "reason": f"{j(seen)} visible on-screen, but {j(offscreen)} heard off-screen"}
    if offscreen:
        return {"suggest": "unseen_ambient",
                "reason": f"{j(offscreen)} heard, source not visible in frame"}
    if seen:
        return {"suggest": "seen_ambient",
                "reason": f"{j(seen)} visible on screen (source seen)"}
    return {"suggest": "no_ambient",
            "reason": "no clear ambient sound detected (mostly speech/music/quiet)"}


def main():
    folders = ["unseen_ambient", "seen_ambient", "no_ambient", "unsorted"] if "--all" in sys.argv \
        else ["unsorted"]
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    clips = [p for f in folders for p in sorted((BENCH / f).glob("*")) if p.suffix.lower() in EXT]
    for i, p in enumerate(clips, 1):
        key = f"{p.parent.name}/{p.name}"
        try:
            out[key] = suggest_clip(p)
        except Exception as e:
            out[key] = {"suggest": "?", "reason": f"analysis failed: {type(e).__name__}"}
        print(f"[{i}/{len(clips)}] {key} -> {out[key]['suggest']}: {out[key]['reason']}")
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")  # save as we go
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
