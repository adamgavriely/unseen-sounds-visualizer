"""Stage 3 — decide, per segment, whether/what/how to visualize.

This is the research core. Two backends:
  - "rule": trivial stub (visualize every non-empty segment). No API needed.
  - "llm":  the real component (Gemini). For each segment it decides whether the
            speech is concrete/depictable, what to depict, writes a text-to-image
            prompt, and routes generate-vs-retrieve. Segments are batched per call
            (rate limits) with the full transcript as context.
"""
from __future__ import annotations
import json
import os
from typing import List

from .models import Segment, Plan


# ---------------------------------------------------------------- rule stub ---
def plan_rule(segments: List[Segment]) -> List[Plan]:
    plans: List[Plan] = []
    for s in segments:
        plans.append(Plan(
            index=s.index, start=s.start, end=s.end, text=s.text,
            visualize=bool(s.text),
            reason="rule-stub: visualize every non-empty segment",
            subject=s.text, image_prompt=s.text, backend="generate",
        ))
    return plans


# ---------------------------------------------------------------- LLM planner -
_INSTRUCTIONS = """\
You plan visuals for a system that lets Deaf viewers SEE spoken news instead of \
hearing it. You are given numbered transcript segments. For EACH segment to plan, \
decide how (or whether) to illustrate it.

Rules:
- visualize: true ONLY if the segment describes something concrete and depictable \
in a single image (a physical object, place, identifiable person, action, or \
event). Set false for abstract/procedural speech (statistics, opinions, vague \
references, filler, transitions) that no honest single image can depict. It is \
better to stay silent than to show a misleading image.
- reason: one short clause justifying the decision.
- subject: a short phrase naming what to depict (empty string if visualize=false).
- image_prompt: a vivid, concrete, neutral text-to-image prompt describing the \
scene photojournalistically, with NO text/words rendered in the image (empty \
string if visualize=false).
- backend: "retrieve" when the segment refers to a SPECIFIC real, named, \
identifiable event/person/place/organisation, where generating an image would \
fabricate reality (prefer a real retrieved photo). "generate" for generic or \
illustrative concepts.
- fallback: always "hold_previous".

Use the WHOLE transcript for context: resolve pronouns and follow the running story.
Return ONLY a JSON array, one object per segment TO PLAN, each with keys:
index, visualize, reason, subject, image_prompt, backend, fallback.
"""


def _build_prompt(chunk: List[Segment], context: List[Segment]) -> str:
    lines = [_INSTRUCTIONS, ""]
    if context:
        lines.append("Earlier context (do NOT output objects for these):")
        for s in context:
            lines.append(f"  [{s.index}] {s.text}")
        lines.append("")
    lines.append("Segments to plan:")
    for s in chunk:
        lines.append(f"[{s.index}] ({s.start:.1f}-{s.end:.1f}s) {s.text}")
    return "\n".join(lines)


def _coerce_list(data):
    """Gemini returns a JSON array; tolerate an accidental wrapper object."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"planner response was not a JSON array: {type(data)}")


def plan_llm(segments: List[Segment], model: str, context: int = 3,
             chunk_size: int = 25) -> List[Plan]:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (put it in .env; main.py loads it).")
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                      temperature=0.2)

    by_index: dict[int, dict] = {}
    for start in range(0, len(segments), chunk_size):
        chunk = segments[start:start + chunk_size]
        ctx = segments[max(0, start - context):start]
        resp = client.models.generate_content(
            model=model, contents=_build_prompt(chunk, ctx), config=cfg)
        if not resp.text:
            raise RuntimeError("empty planner response (possibly blocked/rate-limited)")
        for item in _coerce_list(json.loads(resp.text)):
            by_index[int(item["index"])] = item

    plans: List[Plan] = []
    for s in segments:
        item = by_index.get(s.index, {})
        vis = bool(item.get("visualize", False))
        plans.append(Plan(
            index=s.index, start=s.start, end=s.end, text=s.text,
            visualize=vis,
            reason=item.get("reason", ""),
            subject=item.get("subject", "") if vis else "",
            image_prompt=item.get("image_prompt", "") if vis else "",
            backend=item.get("backend", "generate"),
            fallback=item.get("fallback", "hold_previous"),
        ))
    return plans


# ---------------------------------------------------------------- dispatch ----
def plan(segments: List[Segment], backend: str = "rule", **kwargs) -> List[Plan]:
    if backend == "rule":
        return plan_rule(segments)
    if backend == "llm":
        return plan_llm(
            segments,
            model=kwargs.get("model", "gemini-2.5-flash"),
            context=kwargs.get("context", 3),
            chunk_size=kwargs.get("chunk_size", 25),
        )
    raise ValueError(f"unknown planner backend: {backend}")
