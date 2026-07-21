"""Stage 1 — extract a mono 16 kHz WAV from the input video (ffmpeg)."""
from __future__ import annotations
import subprocess
from pathlib import Path


def extract_audio(video_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


def media_duration(path: Path) -> float:
    """Duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())
