"""Shared data structures that flow between pipeline stages.

Every stage consumes and/or produces these typed objects, so stages stay
decoupled and independently testable/swappable. All are JSON-serialisable via
``to_dict`` for writing inspectable artifacts under ``data/work/<stem>/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class MediaInfo:
    """Stage 1 output: the standardized audio + basic media facts."""
    video_path: str
    wav_path: str
    duration: float          # seconds
    sample_rate: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeechSegment:
    """Stage 3 output: one timestamped chunk of transcribed speech."""
    index: int
    start: float             # seconds
    end: float               # seconds
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneContext:
    """Stage 2 output: what is already visible in the video.

    This is the evidence the cross-modal gate (Stage 5) uses to decide whether a
    sound is redundant (its source is already on screen) or worth augmenting.
    """
    summary: str = ""                          # one-paragraph scene description
    visible_entities: List[str] = field(default_factory=list)  # objects/agents on screen
    setting: str = ""                          # e.g. "city street, daytime"
    frames_analyzed: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)          # backend-specific extras

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudioEvent:
    """Stage 4 output: one detected non-speech sound with a time span.

    ``source_on_screen`` / ``on_screen_prob`` are filled by the optional
    on/off-screen localization signal (see docs/project_notes.tex sec:stage5);
    left as None when that signal is not used.
    """
    label: str
    start: float             # seconds
    end: float               # seconds
    confidence: float = 0.0
    source_on_screen: Optional[bool] = None
    on_screen_prob: Optional[float] = None
    detail: str = ""         # most specific detected sub-label, for depiction only
    # Every separate burst of this sound, as (start, end). ``start``/``end`` above are
    # the strongest burst only; a picture is shown during each burst, not across them.
    spans: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AugmentationSpec:
    """Stage 5 output: the gating + depiction decision for one candidate event.

    ``augment`` is the gate: False means "stay silent" (e.g. source already
    visible, or not salient). When True, the remaining fields drive generation
    (Stage 6) and placement.
    """
    index: int
    event_label: str
    start: float
    end: float
    augment: bool                    # the gate decision
    confidence: float = 0.0          # detector confidence; drives visual weight (opacity)
    reason: str = ""                 # why augment / why not (required for auditability)
    subject: str = ""                # short description of what to depict
    image_prompt: str = ""           # full prompt for the generator
    placement: str = "peripheral"    # "peripheral" | "anchored" (see sec:placement)
    backend: str = "generate"        # "generate" | "retrieve"
    image_path: Optional[str] = None  # filled in by the generator (Stage 6)
    detail: str = ""                 # most specific sub-label PANNs heard (e.g. Shatter under Glass)
    talked_about: bool = False       # people on the soundtrack are reacting to this sound
    spans: List[Tuple[float, float]] = field(default_factory=list)  # every burst; see AudioEvent

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
