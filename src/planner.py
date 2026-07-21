"""Stage 3 — decide, per segment, whether/what/how to visualize.

Two backends:
  - "rule": a trivial stub (visualize everything, subject = the text). Lets the
    whole pipeline run with no API key. Placeholder until the LLM backend is on.
  - "llm":  the real research component (Anthropic). Enabled once ANTHROPIC_API_KEY
    is set and `anthropic` is installed. Implemented in a later step.
"""
from __future__ import annotations
from typing import List

from .models import Segment, Plan


def plan_rule(segments: List[Segment]) -> List[Plan]:
    plans: List[Plan] = []
    for s in segments:
        plans.append(Plan(
            index=s.index, start=s.start, end=s.end, text=s.text,
            visualize=bool(s.text),
            reason="rule-stub: visualize every non-empty segment",
            subject=s.text,
            image_prompt=s.text,
            backend="generate",
        ))
    return plans


def plan_llm(segments: List[Segment], model: str, context: int = 3) -> List[Plan]:
    # TODO: implement once the Anthropic key is available.
    # One structured call per segment (with `context` previous segments as
    # context) returning: visualize, reason, subject, image_prompt, backend, fallback.
    raise NotImplementedError(
        "LLM planner not wired yet — set ANTHROPIC_API_KEY, `pip install anthropic`, "
        "then implement plan_llm. Use --planner rule for now."
    )


def plan(segments: List[Segment], backend: str = "rule", **kwargs) -> List[Plan]:
    if backend == "rule":
        return plan_rule(segments)
    if backend == "llm":
        return plan_llm(segments, **kwargs)
    raise ValueError(f"unknown planner backend: {backend}")
