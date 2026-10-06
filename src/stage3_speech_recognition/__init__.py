"""Stage 3 - Speech Recognition.

Transcribe speech into timestamped segments via faster-whisper. Speech is a *secondary*
signal here (the star is non-speech sound); the transcript gives context to the
cross-modal gate and helps avoid augmenting content captions already cover.

Degrades gracefully: if faster-whisper is not installed, returns [] with a
warning so the rest of the pipeline still runs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.types import SpeechSegment


# Amendment 3 (2026-09-21, bug fix): both ASR back-ends emit non-speech as text ("Music",
# "[Music]", "(applause)", "You", "Thank you." on silence), and every consumer treats a
# non-empty transcript as "someone is speaking" (judge reference, speech-rescue band).
# A stoplist only -- real one-word speech ("Help!") is kept.
_NON_SPEECH = {"music", "applause", "laughter", "laughs", "noise", "silence", "you", "thank you",
               "thanks for watching", "bye", "the end", "subtitles by the amara.org community"}


def _is_speech(text: str) -> bool:
    t = text.strip().lower()
    if not re.search(r"[a-z]", t) or re.fullmatch(r"\s*(\[[^\]]*\]|\([^)]*\))\s*", t):
        return False                                    # no letters, or one bracketed tag "[music playing]"
    t = re.sub(r"^[\[\(\s]+|[\]\)\.\!\s]+$", "", t)          # strip brackets, parentheses, end marks
    return bool(t) and t not in _NON_SPEECH


def transcribe(wav_path: Path, model_size: str = "base",
               device: str = "cpu", compute_type: str = "int8",
               language: str = "en") -> List[SpeechSegment]:
    # v4: Granite Speech 4.1-2B (see granite.py) when config names it; Whisper otherwise
    if "granite" in str(model_size).lower():
        from src.stage3_speech_recognition.granite import transcribe_granite
        return [s for s in transcribe_granite(Path(wav_path), device=device) if _is_speech(s.text)]
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
        if not text or not _is_speech(text):
            continue
        segments.append(SpeechSegment(index=i, start=float(s.start),
                                      end=float(s.end), text=text))
    return segments
