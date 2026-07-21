"""End-to-end pipeline: video -> illustrated video.

Walking skeleton. Run:
    py -3.11 main.py --input data/input/clip.mp4

Stages: [1] extract audio -> [2] ASR -> [3] plan -> [4] visualize -> [5] compose.
Intermediate artifacts (audio, segments.json, plans.json, images/) are written
under data/work/<clip-name>/ for inspection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import config
from src import audio, asr, planner, visualizer, compositor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to input video")
    ap.add_argument("--planner", default="rule", choices=["rule", "llm"])
    ap.add_argument("--visualizer", default="placeholder",
                    choices=["placeholder", "diffusion"])
    ap.add_argument("--whisper-model", default=config.WHISPER_MODEL)
    args = ap.parse_args()

    video = Path(args.input).resolve()
    if not video.exists():
        raise SystemExit(f"input not found: {video}")

    work = config.WORK_DIR / video.stem
    work.mkdir(parents=True, exist_ok=True)

    print("[1/5] extracting audio...")
    wav = audio.extract_audio(video, work / "audio.wav")
    duration = audio.media_duration(video)

    print("[2/5] transcribing...")
    segments = asr.transcribe(wav, model_size=args.whisper_model,
                              device=config.WHISPER_DEVICE,
                              compute_type=config.WHISPER_COMPUTE)
    (work / "segments.json").write_text(
        json.dumps([s.to_dict() for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"       {len(segments)} segments")

    print(f"[3/5] planning ({args.planner})...")
    plans = planner.plan(segments, backend=args.planner,
                         model=config.PLANNER_MODEL, context=config.PLANNER_CONTEXT)
    (work / "plans.json").write_text(
        json.dumps([p.to_dict() for p in plans], indent=2, ensure_ascii=False),
        encoding="utf-8")
    n_vis = sum(1 for p in plans if p.visualize)
    print(f"       {n_vis}/{len(plans)} segments to visualize")

    print(f"[4/5] visualizing ({args.visualizer})...")
    plans = visualizer.visualize(plans, work, backend=args.visualizer,
                                 size=config.RESOLUTION)

    print("[5/5] composing video...")
    out = config.OUTPUT_DIR / f"{video.stem}_illustrated.mp4"
    compositor.compose(plans, wav, out, total_duration=duration,
                       size=config.RESOLUTION, fps=config.FPS)
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
