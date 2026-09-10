"""Stage 5 - Cross-Modal Semantic Analysis (the intellectual core).

Decide, per detected sound event: (1) is it salient? (2) is its source/meaning
already visible in the scene -> if so, STAY SILENT (the gate); (3) if augmenting,
what to depict. See docs/project_notes.tex sec:stage5 for the full design.

STUB: a transparent rule-based gate so the pipeline runs and its decisions are
inspectable. TODO: combine an on/off-screen sound-source-localization signal with
a GROUNDED LLM decision (structured JSON over the Stage-2 entity list, Stage-3
transcript, Stage-4 events) - never over raw media - to curb MLLM hallucination.

Even in PASS-THROUGH (gate disabled, v1 prototype) we still drop speech, ambience
and music, and merge PANNs' label families to one entry per real source, so we
visualize a handful of distinct non-speech sounds rather than 20 near-duplicates.
"""
from __future__ import annotations

from typing import List

from src.types import SceneContext, SpeechSegment, AudioEvent, AugmentationSpec
from src.labels import is_salient_nonspeech, consolidate_families, min_confidence, depiction_query, disambiguate, contextual_subject


def plan_augmentations(scene: SceneContext,
                       segments: List[SpeechSegment],
                       events: List[AudioEvent],
                       threshold: float = 0.3,
                       gate_enabled: bool = False,
                       display_threshold: float = 0.15,
                       augment_threshold: float | None = None) -> List[AugmentationSpec]:
    if augment_threshold is None:
        augment_threshold = display_threshold
    mode = "rule-based gate" if gate_enabled else "PASS-THROUGH (detect-all)"
    print(f"       [stage5] {mode} (TODO: localization + grounded LLM).")
    visible = {e.lower() for e in scene.visible_entities}
    setting = getattr(scene, "setting", "") or ""
    setting_group = (scene.raw or {}).get("setting_group", "") if scene.raw else ""

    # Keep only discrete non-speech sounds, collapse label families to one/source.
    candidates = consolidate_families([e for e in events if is_salient_nonspeech(e.label)])

    specs: List[AugmentationSpec] = []
    for i, ev in enumerate(candidates):
        if not gate_enabled:
            # v1 prototype: visualize every distinct non-speech sound above a
            # light display threshold (no seen/not-seen gating yet).
            augment = ev.confidence >= display_threshold
            reason = ("detect-all mode: distinct non-speech sound"
                      if augment else
                      f"below display threshold ({ev.confidence:.2f} < {display_threshold})")
        else:
            salient = ev.confidence >= min_confidence(ev.label, display_threshold)
            redundant = (ev.source_on_screen is True) or (ev.label.lower() in visible)
            augment = (salient and not redundant
                       and ev.confidence >= min_confidence(ev.label, augment_threshold))
            if not salient:
                reason = f"below display threshold ({ev.confidence:.2f} < {display_threshold})"
            elif redundant:
                reason = "source already visible on screen (stay silent)"
            else:
                reason = "salient non-speech sound, source not visible -> augment"
        # Gate on the FAMILY, depict the SPECIFIC sound. The family is what visibility
        # concepts are keyed on, but drawing it loses everything the detector knew: a
        # fire engine became a generic "Siren", whose query hint is "ambulance", so the
        # system showed an ambulance for a fire truck. 85 of 100 clips had a more
        # specific label available than the one being drawn. See labels.depiction_query.
        # Cross-modal depiction (proposal, Stage 2/5): the scene decides both WHAT the
        # sound most likely is and HOW to draw it. Disambiguation fires only for the
        # handful of genuinely confusable pairs; phrasing applies whenever the setting
        # is known, so "Water" becomes a stream in a forest or a tap in a kitchen.
        if augment:
            label_for_image = disambiguate(ev.label, setting)
            detail_for_image = ev.detail if label_for_image == ev.label else ""
            subject = contextual_subject(label_for_image, detail_for_image,
                                         setting, setting_group)
            if label_for_image != ev.label:
                reason += f" (scene says {label_for_image.lower()}, not {ev.label.lower()})"
        else:
            subject = ""
        specs.append(AugmentationSpec(
            index=i, event_label=ev.label, start=ev.start, end=ev.end,
            augment=augment, confidence=ev.confidence, reason=reason,
            subject=subject,
            image_prompt=(f"A clear, simple photograph of: {subject}" if augment else ""),
        ))
    return specs
