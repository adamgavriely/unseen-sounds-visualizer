"""Stage 5 - Cross-Modal Semantic Analysis (the intellectual core).

Decide, per detected sound event: (1) is it salient? (2) is its source/meaning
already visible in the scene -> if so, STAY SILENT (the gate); (3) if augmenting,
what to depict. See docs/project_notes.tex sec:stage5 for the full design.

The rule-based gate here is the cheap first pass (salience, Stage-2 concept list);
the per-sound decisions -- visibility from the frames, what kind of sound, what to
draw, speech context -- are made in reason.py by the VLM. Stage 2's verdict is
deferred to it when it is available (see `defer` below).

Even in PASS-THROUGH (gate disabled, v1 prototype) we still drop speech, ambience
and music, and merge PANNs' label families to one entry per real source, so we
visualize a handful of distinct non-speech sounds rather than 20 near-duplicates.
"""
from __future__ import annotations

from typing import List

import config
from src.types import SceneContext, SpeechSegment, AudioEvent, AugmentationSpec
from src.labels import is_salient_nonspeech, consolidate_families, min_confidence, depiction_query


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
    print(f"       [stage5] {mode}.")
    visible = {e.lower() for e in scene.visible_entities}
    setting = getattr(scene, "setting", "") or ""
    setting_group = (scene.raw or {}).get("setting_group", "") if scene.raw else ""

    # Keep only discrete non-speech sounds, collapse label families to one/source.
    # Amendment 3 (2026-09-21, bug fix): families are built from the firings at or above the
    # display bar as before, and a family with NO firing at the bar is added from the band
    # [bar/2, bar) so it reaches the specs with the reason "below display threshold" -- the
    # speech-rescue band in reason.py had been unreachable since v3 because everything
    # below the bar was dropped here. A marginal firing never extends or details a strong
    # family (the "reversing tractor" case), because the two sets are consolidated apart.
    drawable = [e for e in events if is_salient_nonspeech(e.label)]
    # Round 65 RETURN: a RET row (marked by a zero-length break at its start; only config.PERC_RETURN makes one) is its own
    # candidate -- never merged into its family's spec, so the gate judges its stretch alone
    from src.stage4_audio_event_detection import is_return_row as _isret
    rets = [e for e in drawable if _isret(e.start, getattr(e, "breaks", None))]
    drawable = [e for e in drawable if not _isret(e.start, getattr(e, "breaks", None))]
    strong_bar = min_confidence("", display_threshold)
    candidates = consolidate_families(drawable, threshold=strong_bar)
    strong_families = {c.label for c in candidates}
    from src.labels import canonical as _canon
    marginal = [e for e in drawable if 0.5 * strong_bar <= e.confidence < strong_bar
                and _canon(e.label) not in strong_families]
    candidates += consolidate_families(marginal, threshold=0.5 * strong_bar)
    for e in rets:                                   # Round 65 RETURN: one spec each (consolidate_families of one row)
        candidates += consolidate_families([e], threshold=0.0)

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
            # Stage 2's concept list is a whole-clip, object-level pass: it can say a
            # thing is somewhere in the video, not that the viewer sees it making this
            # sound now. When the per-sound VLM check is available it gets the final
            # word -- it silenced a fire alarm because the pull station was on screen,
            # and the VLM question that would have caught that never ran. Without the
            # VLM (the baselines, or no GPU) Stage 2's verdict stands as before.
            defer = redundant and getattr(config, "VLM_VISIBILITY", False)                 and getattr(config, "DEPICTION_REASONING", False)
            augment = (salient and (not redundant or defer)
                       and ev.confidence >= min_confidence(ev.label, augment_threshold))
            if not salient:
                reason = f"below display threshold ({ev.confidence:.2f} < {display_threshold})"
                if redundant:
                    # recorded so a later rescue-by-speech cannot revive a sound whose
                    # source is on screen; the visibility rule beats every other signal
                    reason += "; source visible on screen anyway"
            elif redundant and defer:
                reason = "stage 2 saw the source somewhere in the clip; VLM to confirm"
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
        # Placeholder only. What is actually drawn is decided by the VLM in reason.py,
        # which sees frames spanning the sound; the hand-written setting and phrasing
        # tables that used to fill this in are gone, because a fixed category list
        # cannot describe an arbitrary scene and mislabelled the ones it could not fit.
        subject = depiction_query(ev.label, ev.detail) if augment else ""
        specs.append(AugmentationSpec(
            index=i, event_label=ev.label, start=ev.start, end=ev.end,
            augment=augment, confidence=ev.confidence, reason=reason,
            subject=subject, detail=ev.detail or "", spans=list(ev.spans),
            source=getattr(ev, "source", "") or "",
            breaks=list(getattr(ev, "breaks", None) or []),
            rescued=bool(getattr(ev, "rescued", False)),
            arbiter=bool(getattr(ev, "arbiter", False)),
            image_prompt=(f"A clear, simple photograph of: {subject}" if augment else ""),
        ))
    # Deduplication now happens in reason.py, on the depictions the VLM chose,
    # rather than on a hand-written synonym table.
    return specs
