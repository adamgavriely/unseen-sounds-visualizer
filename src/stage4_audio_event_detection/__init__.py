"""Stage 4 - Audio Event Detection.

Detect and classify non-speech sounds (ambient, environmental, acoustic events)
with time boundaries over the AudioSet ontology. This is the core signal the
system augments.

STUB: returns no events. TODO: run PANNs/CNN14 (fast, light first pass) then a
BEATs tagger (quality); keep events with confidence >= threshold; use a
frame-level/CRNN model only if precise onset/offset timing is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.types import AudioEvent


def detect_events(wav_path: Path, threshold: float = 0.3,
                  model: str = "", device: str = "cpu") -> List[AudioEvent]:
    # TODO(stage4): load PANNs/BEATs, run on the wav, map to AudioEvent list.
    print("       [stage4] STUB - no audio event detection yet "
          "(returning no events).")
    return []
