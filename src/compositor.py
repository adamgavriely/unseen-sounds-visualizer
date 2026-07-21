"""Stage 5 — assemble timed images + original audio into a synced .mp4 (ffmpeg).

Each image tiles the timeline from its own segment start to the next segment's
start (gaps of silence simply extend the previous image). The full original
audio is muxed in, so the visuals stay locked to the speech.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

from PIL import Image

from .models import Plan


def _blank(path: Path, size):
    Image.new("RGB", size, color=(15, 15, 20)).save(path)


def compose(plans: List[Plan], audio_path: Path, out_path: Path,
            total_duration: float, size=(1280, 720), fps: int = 25) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent

    # Build (image, duration) timeline. Only segments that produced an image
    # get a slot; a leading blank covers any pre-speech gap.
    shots = [p for p in plans if p.image_path]
    if not shots:
        raise RuntimeError("no images to compose")

    timeline = []  # list of (image_path, duration_seconds)
    if shots[0].start > 0.05:
        blank = work / "_blank.png"
        _blank(blank, size)
        timeline.append((blank, shots[0].start))

    for i, p in enumerate(shots):
        end = shots[i + 1].start if i + 1 < len(shots) else total_duration
        dur = max(0.1, end - p.start)
        timeline.append((Path(p.image_path), dur))

    # ffmpeg concat demuxer needs the last entry repeated (its duration line is
    # otherwise ignored).
    concat_file = work / "_concat.txt"
    lines = []
    for img, dur in timeline:
        lines.append(f"file '{img.as_posix()}'")
        lines.append(f"duration {dur:.3f}")
    lines.append(f"file '{timeline[-1][0].as_posix()}'")
    concat_file.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(audio_path),
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
