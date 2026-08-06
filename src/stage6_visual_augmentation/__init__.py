"""Stage 6 - Visual Augmentation Generation.

Render the selected augmentations as complementary visuals shown *alongside* the
original video (not replacing it). Static storyboard-style images first; short
motion clips as a stretch.

STUB: writes a labelled placeholder PNG per augmented event (PIL, no GPU) so the
pipeline produces inspectable output. TODO: generate with SDXL/FLUX conditioned
on the Stage-5 prompt; then composite alongside the source video (see
``composite_alongside``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.types import AugmentationSpec


def _placeholder_image(path: Path, caption: str, size=(1024, 1024)) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, color=(20, 22, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size[0] - 8, size[1] - 8], outline=(90, 100, 140), width=3)
    draw.text((28, 28), "[augmentation placeholder]", fill=(150, 160, 190))
    # naive word-wrap
    words, line, y = caption.split(), "", size[1] // 2 - 20
    for w in words:
        if len(line) + len(w) > 40:
            draw.text((28, y), line, fill=(220, 225, 235)); y += 22; line = ""
        line += w + " "
    if line:
        draw.text((28, y), line, fill=(220, 225, 235))
    img.save(path)


def generate_augmentations(specs: List[AugmentationSpec], work_dir: Path,
                           size=(1024, 1024), model: str = "",
                           device: str = "cpu") -> List[AugmentationSpec]:
    out_dir = work_dir / "augmentations"
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale images from previous runs
    for old in out_dir.glob("aug_*.png"):
        old.unlink()

    n = 0
    for spec in specs:
        if not spec.augment:
            continue
        path = out_dir / f"aug_{spec.index:03d}.png"
        # TODO(stage6): replace with SDXL/FLUX generation from spec.image_prompt.
        _placeholder_image(path, spec.image_prompt or spec.event_label, size=size)
        spec.image_path = str(path)
        n += 1
    print(f"       [stage6] STUB - wrote {n} placeholder image(s) "
          f"(TODO: SDXL/FLUX).")
    return specs


def composite_alongside(video_path: Path, specs: List[AugmentationSpec],
                        out_path: Path, fps: int = 25) -> Path:
    """Compose augmentations *alongside* the original video (time-aligned)."""
    # TODO(stage6): ffmpeg overlay / side-by-side (hstack) of a time-aligned
    # augmentation track next to the source video. The legacy compositor (in
    # archive/) *replaced* the footage; here we display augmentations beside/over
    # it, so the layout differs. Placement/timing is a design variable (see
    # docs/project_notes.tex sec:placement).
    raise NotImplementedError(
        "composite_alongside is a TODO - see archive/legacy_speech_pipeline/"
        "compositor.py for reusable ffmpeg concat logic.")
