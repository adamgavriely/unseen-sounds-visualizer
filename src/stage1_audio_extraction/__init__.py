"""Stage 1 - Audio Extraction.

Extract a standardized mono WAV from the input video via ffmpeg, and probe basic
media facts. No models required, only ffmpeg on PATH.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.types import MediaInfo


def media_duration(path: Path) -> float:
    """Duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: Path, out_wav: Path,
                  sample_rate: int = 16000) -> MediaInfo:
    """Extract mono WAV at ``sample_rate`` and return media facts."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return MediaInfo(
        video_path=str(video_path),
        wav_path=str(out_wav),
        duration=media_duration(video_path),
        sample_rate=sample_rate,
    )
