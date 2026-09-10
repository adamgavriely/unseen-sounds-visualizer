"""Pipeline orchestrator: video -> audio/scene/speech/events -> gate -> images -> composited mp4.

v1 prototype (detect-all, no gating): stages 1 (ffmpeg), 3 (whisper) and 4 (PANNs)
are real; Stage 5 passes through every distinct non-speech sound; Stage 6 retrieves
a CC image per sound (Openverse) and composites it alongside the original video.
Stage 2 (video understanding) is still a stub -> the seen/not-seen gate is v2.

Writes inspectable artifacts under ``data/work/<stem>/`` (audio.wav, *.json,
events_plot.png, augmentations/aug_*.png, credits.json) and the final
``data/output/<stem>_augmented.mp4``.
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
from src.stage2_video_understanding import analyze
from src.stage3_speech_recognition import transcribe
from src.stage4_audio_event_detection import detect_events
from src.stage5_cross_modal_analysis import plan_augmentations
from src.stage6_visual_augmentation import generate_augmentations, composite_alongside


@dataclass
class PipelineResult:
    media: MediaInfo
    scene: SceneContext
    segments: List[SpeechSegment]
    events: List[AudioEvent]
    specs: List[AugmentationSpec]
    work_dir: Path
    output_path: Path = None


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run(video_path: Path, work_root: Path = None) -> PipelineResult:
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise SystemExit(f"input not found: {video_path}")
    work_root = work_root or config.WORK_DIR
    work = work_root / video_path.stem
    work.mkdir(parents=True, exist_ok=True)

    print("[1/7] extracting audio (ffmpeg)...")
    media = extract_audio(video_path, work / "audio.wav", config.SAMPLE_RATE)
    _dump(work / "media.json", media.to_dict())
    print(f"       {media.duration:.1f}s @ {media.sample_rate} Hz")

    print(f"[2/7] video understanding ({config.VIDEO_BACKEND})...")
    scene = analyze(video_path, backend=config.VIDEO_BACKEND,
                    num_frames=config.NUM_FRAMES, model=config.VIDEO_MODEL,
                    vlm_model=config.VLM_MODEL, siglip_model=config.SIGLIP_MODEL, owl_model=config.OWL_MODEL,
                        owl_threshold=config.OWL_THRESHOLD,
                        siglip_threshold=config.SIGLIP_THRESHOLD, device=config.DEVICE,
                    threshold=config.VISIBILITY_THRESHOLD)
    _dump(work / "scene.json", scene.to_dict())

    print("[3/7] speech recognition (whisper)...")
    segments = transcribe(Path(media.wav_path), model_size=config.WHISPER_MODEL,
                          device=config.DEVICE, compute_type=config.WHISPER_COMPUTE)
    _dump(work / "segments.json", [s.to_dict() for s in segments])
    print(f"       {len(segments)} speech segment(s)")

    print("[4/7] audio event detection...")
    events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                           min_dur=config.AED_MIN_DUR, model=config.AED_MODEL,
                           device=config.DEVICE, plot_path=work / "events_plot.png",
                           plot_top_k=config.AED_PLOT_TOP_K)
    _dump(work / "events.json", [e.to_dict() for e in events])
    print(f"       {len(events)} event span(s)")

    print("[5/7] cross-modal gating...")
    specs = plan_augmentations(scene, segments, events,
                               threshold=config.AED_THRESHOLD,
                               gate_enabled=config.GATE_ENABLED,
                               display_threshold=config.DISPLAY_THRESHOLD,
                               augment_threshold=config.AUGMENT_THRESHOLD)
    _dump(work / "augmentations.json", [s.to_dict() for s in specs])
    n_aug = sum(1 for s in specs if s.augment)
    print(f"       {n_aug}/{len(specs)} sound(s) selected to visualize")

    print(f"[6/7] visual augmentation ({config.GEN_BACKEND})...")
    # Decide WHAT to draw by looking at the video around each sound, before the
    # generator is loaded -- the reasoner and the image model do not co-fit.
    if getattr(config, 'DEPICTION_REASONING', True) and any(sp.augment for sp in specs):
        from src.stage5_cross_modal_analysis import reason
        try:
            reason.decide_subjects(video_path, specs,
                                   transcript=' '.join(sg.text for sg in segments),
                                   model=config.VLM_MODEL, device=config.DEVICE)
        finally:
            reason.unload()
    specs = generate_augmentations(specs, work, backend=config.GEN_BACKEND,
                                   size=config.RESOLUTION, model=config.GEN_MODEL,
                                   device=config.DEVICE)
    _dump(work / "augmentations.json", [s.to_dict() for s in specs])

    print(f"[7/7] compositing alongside the video (mode={config.RENDER_MODE})...")
    out_mp4 = config.OUTPUT_DIR / f"{video_path.stem}_augmented.mp4"
    composite_alongside(video_path, specs, out_mp4, duration=media.duration,
                        panel=config.PANEL_SIZE, fps=config.FPS,
                        mode=config.RENDER_MODE)
    print(f"\nDone -> {out_mp4}\n       artifacts in {work}")
    return PipelineResult(media, scene, segments, events, specs, work, out_mp4)
