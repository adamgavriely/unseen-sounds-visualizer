"""Stage 2 - Video Understanding.

Analyze the visual stream to determine what is already visible to the viewer.
Its output (SceneContext) is what the cross-modal gate (Stage 5) uses to decide
whether a sound is redundant (source on screen) or worth augmenting.

STUB: returns an empty SceneContext. TODO: sample ~NUM_FRAMES frames and query a
VLM (Qwen2.5-VL / InternVL / LLaVA) for a scene summary + a list of visible
entities (and, ideally, likely sound sources that are visible).
"""
from __future__ import annotations

from pathlib import Path

from src.types import SceneContext


def analyze_video(video_path: Path, num_frames: int = 8,
                  model: str = "", device: str = "cpu") -> SceneContext:
    # TODO(stage2): frame-sample the clip and run a VLM to fill this in.
    print("       [stage2] STUB - no video understanding yet "
          "(returning empty SceneContext).")
    return SceneContext(
        summary="",
        visible_entities=[],
        setting="",
        frames_analyzed=0,
        raw={"stub": True, "intended_model": model},
    )
