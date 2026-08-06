"""Stage 5 - Cross-Modal Semantic Analysis (the intellectual core).

Decide, per detected sound event: (1) is it salient? (2) is its source/meaning
already visible in the scene -> if so, STAY SILENT (the gate); (3) if augmenting,
what to depict. See docs/project_notes.tex sec:stage5 for the full design.

STUB: a transparent rule-based gate so the pipeline runs and its decisions are
inspectable. TODO: combine an on/off-screen sound-source-localization signal with
a GROUNDED LLM decision (structured JSON over the Stage-2 entity list, Stage-3
transcript, Stage-4 events) - never over raw media - to curb MLLM hallucination.
"""
from __future__ import annotations

from typing import List

from src.types import SceneContext, SpeechSegment, AudioEvent, AugmentationSpec


def plan_augmentations(scene: SceneContext,
                       segments: List[SpeechSegment],
                       events: List[AudioEvent],
                       threshold: float = 0.3) -> List[AugmentationSpec]:
    print("       [stage5] STUB - transparent rule-based gate "
          "(TODO: localization + grounded LLM).")
    visible = {e.lower() for e in scene.visible_entities}
    specs: List[AugmentationSpec] = []
    for i, ev in enumerate(events):
        # Gate rule (placeholder logic):
        #  - drop low-confidence events
        #  - if the source is known on-screen (or the label matches a visible
        #    entity), stay silent (visual redundancy).
        salient = ev.confidence >= threshold
        redundant = (ev.source_on_screen is True) or (ev.label.lower() in visible)
        augment = salient and not redundant
        if not salient:
            reason = f"below salience threshold ({ev.confidence:.2f} < {threshold})"
        elif redundant:
            reason = "source already visible on screen (stay silent)"
        else:
            reason = "salient non-speech sound, source not visible -> augment"
        specs.append(AugmentationSpec(
            index=i, event_label=ev.label, start=ev.start, end=ev.end,
            augment=augment, reason=reason,
            subject=ev.label if augment else "",
            image_prompt=(f"A clear, simple illustration of: {ev.label}"
                          if augment else ""),
        ))
    return specs
