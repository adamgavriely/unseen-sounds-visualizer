"""Stage 2 — transcribe audio into timestamped segments (faster-whisper)."""
from __future__ import annotations
from pathlib import Path
from typing import List

from .models import Segment


def transcribe(wav_path: Path, model_size: str = "base",
               device: str = "cpu", compute_type: str = "int8",
               language: str = "en") -> List[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, _info = model.transcribe(str(wav_path), language=language)

    segments: List[Segment] = []
    for i, s in enumerate(raw_segments):
        text = s.text.strip()
        if not text:
            continue
        segments.append(Segment(index=i, start=float(s.start),
                                end=float(s.end), text=text))
    return segments
