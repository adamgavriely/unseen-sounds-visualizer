"""Shared data structures that flow between pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Segment:
    """One timestamped chunk of speech from ASR."""
    index: int
    start: float          # seconds
    end: float            # seconds
    text: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Plan:
    """The planner's decision for one segment: whether/what/how to visualize."""
    index: int
    start: float
    end: float
    text: str
    visualize: bool
    reason: str = ""
    subject: str = ""            # short description of what to depict
    image_prompt: str = ""       # full prompt for the visualizer
    backend: str = "generate"    # "generate" | "retrieve"
    fallback: str = "hold_previous"   # what to show when visualize=False
    image_path: Optional[str] = None  # filled in by the visualizer

    def to_dict(self):
        return asdict(self)
