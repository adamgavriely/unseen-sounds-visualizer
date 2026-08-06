"""Pipeline orchestrator: video -> (audio, scene, speech, events) -> gate -> augmentations.

Runs stages 1-6 and writes inspectable artifacts under ``data/work/<stem>/``:
    audio.wav, media.json, scene.json, segments.json, events.json,
    augmentations.json, augmentations/aug_*.png

Stage 7 (evaluation) runs separately over the benchmark, not here.
Stages 2/4/5/6 are currently stubs (see each module) so this runs end-to-end
without a GPU or downloaded models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import config
from src.types import (MediaInfo, SceneContext, SpeechSegment, AudioEvent,
                       AugmentationSpec)
from src.stage1_audio_extraction import extract_audio
from src.stage2_video_understanding import analyze_video
from src.stage3_speech_recognition import transcribe
from src.stage4_audio_event_detection import detect_events
from src.stage5_cross_modal_analysis import plan_augmentations
from src.stage6_visual_augmentation import generate_augmentations


@dataclass
class PipelineResult:
    media: MediaInfo
    scene: SceneContext
    segments: List[SpeechSegment]
    events: List[AudioEvent]
    specs: List[AugmentationSpec]
    work_dir: Path


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run(video_path: Path, work_root: Path = None) -> PipelineResult:
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise SystemExit(f"input not found: {video_path}")
    work_root = work_root or config.WORK_DIR
    work = work_root / video_path.stem
    work.mkdir(parents=True, exist_ok=True)

    print("[1/6] extracting audio (ffmpeg)...")
    media = extract_audio(video_path, work / "audio.wav", config.SAMPLE_RATE)
    _dump(work / "media.json", media.to_dict())
    print(f"       {media.duration:.1f}s @ {media.sample_rate} Hz")

    print("[2/6] video understanding...")
    scene = analyze_video(video_path, num_frames=config.NUM_FRAMES,
                          model=config.VIDEO_MODEL, device=config.DEVICE)
    _dump(work / "scene.json", scene.to_dict())

    print("[3/6] speech recognition (whisper)...")
    segments = transcribe(Path(media.wav_path), model_size=config.WHISPER_MODEL,
                          device=config.DEVICE, compute_type=config.WHISPER_COMPUTE)
    _dump(work / "segments.json", [s.to_dict() for s in segments])
    print(f"       {len(segments)} speech segment(s)")

    print("[4/6] audio event detection...")
    events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                           model=config.AED_MODEL, device=config.DEVICE)
    _dump(work / "events.json", [e.to_dict() for e in events])
    print(f"       {len(events)} non-speech event(s)")

    print("[5/6] cross-modal gating...")
    specs = plan_augmentations(scene, segments, events,
                               threshold=config.AED_THRESHOLD)
    _dump(work / "augmentations.json", [s.to_dict() for s in specs])
    n_aug = sum(1 for s in specs if s.augment)
    print(f"       {n_aug}/{len(specs)} event(s) selected to augment")

    print("[6/6] visual augmentation generation...")
    specs = generate_augmentations(specs, work, size=config.RESOLUTION,
                                   model=config.GEN_MODEL, device=config.DEVICE)
    _dump(work / "augmentations.json", [s.to_dict() for s in specs])

    print(f"\nDone -> artifacts in {work}")
    return PipelineResult(media, scene, segments, events, specs, work)
