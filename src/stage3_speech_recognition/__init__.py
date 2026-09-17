"""Stage 3 - Speech Recognition.

Transcribe speech into timestamped segments via faster-whisper. Real
implementation (salvaged from the legacy pipeline). Speech is a *secondary*
signal here (the star is non-speech sound); the transcript gives context to the
cross-modal gate and helps avoid augmenting content captions already cover.

Degrades gracefully: if faster-whisper is not installed, returns [] with a
warning so the skeleton still runs end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.types import SpeechSegment


def transcribe(wav_path: Path, model_size: str = "base",
               device: str = "cpu", compute_type: str = "int8",
               language: str = "en") -> List[SpeechSegment]:
    # v4: Granite Speech 4.1-2B (see granite.py) when config names it; Whisper otherwise
    if "granite" in str(model_size).lower():
        from src.stage3_speech_recognition.granite import transcribe_granite
        return transcribe_granite(Path(wav_path), device=device)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("       [stage3] faster-whisper not installed - skipping ASR "
              "(pip install faster-whisper). Returning no segments.")
        return []

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, _info = model.transcribe(str(wav_path), language=language)

    segments: List[SpeechSegment] = []
    for i, s in enumerate(raw_segments):
        text = s.text.strip()
        if not text:
            continue
        segments.append(SpeechSegment(index=i, start=float(s.start),
                                      end=float(s.end), text=text))
    return segments
