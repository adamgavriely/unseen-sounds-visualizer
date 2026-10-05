"""Stage 2 backend: SigLIP visibility check (default from 2026-08-23).

Why SigLIP replaces CLIP here. Our question is "is source X present in this frame?"
-- an INDEPENDENT yes/no for each candidate source, several of which can be true at
once. CLIP answers a different question: it is trained with a softmax over a batch,
so its scores form a "pick one of N" competition. Three concrete failures followed
from that mismatch:

  * absolute scores shrank as the concept list grew (20 -> 33 concepts pushed a clip
    that is entirely sea below a fixed threshold -- "water not seen");
  * concepts stole probability from each other ("people walking" scored 0.67 on open
    sea, outranking Water);
  * we had to compensate with a relative threshold, which is a workaround, not a fix.

SigLIP is trained with a pairwise SIGMOID loss instead, so every image-text pair is
scored on its own. Adding a concept cannot dilute the others, and an absolute
threshold becomes meaningful again -- exactly the semantics the gate needs. (The
threshold is on the logit rather than the probability; see analyze_video_siglip.)
Verified 6/6 on the cases where CLIP failed, including two clips that are entirely
sea and which CLIP reported as "water not seen".

Same contract as the CLIP backend: returns a SceneContext whose visible_entities
names the sound-source concepts actually on screen.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.types import SceneContext
from src.stage2_video_understanding import VISIBLE_CONCEPTS, _sample_frames

_MODEL = None


def _short_prompt(clip_prompt: str) -> str:
    """CLIP prompts are long and descriptive; SigLIP prefers terse alt-text style."""
    t = clip_prompt.replace("a photo of ", "").split(",")[0].split(" or ")[0]
    return t.strip()


def _load(model_name: str, device: str):
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        proc = AutoProcessor.from_pretrained(model_name)
        _MODEL = (model, proc)
    return _MODEL


def analyze_video_siglip(video_path: Path, num_frames: int = 6,
                         model: str = "google/siglip-base-patch16-224",
                         device: str = "cpu", threshold: float = -7.5,
                         candidates: Optional[List[str]] = None) -> SceneContext:
    """Per-concept score that the source is visible in a frame.

    ``threshold`` is on the raw LOGIT, not the sigmoid probability. SigLIP's absolute
    probabilities are tiny by construction (a frame that is entirely sea scores ~0.003
    for "the sea"), because the sigmoid is calibrated against the whole space of
    possible captions. The separation is nevertheless clean: true matches land around
    logit -1 to -7 while unrelated concepts sit below -10, so the logit is the usable
    decision variable. Prompts are kept SHORT for the same reason -- SigLIP was
    trained on terse alt-text, and an elaborate CLIP-style prompt scored -10.8 on the
    same sea frame that a plain "the sea" scored -5.9.

    A concept counts as visible if ANY sampled frame clears the threshold: a source
    can enter and leave the shot, and the gate only needs to know it was shown.
    """
    try:
        import torch
        labels = [c for c in (candidates or list(VISIBLE_CONCEPTS)) if c in VISIBLE_CONCEPTS]
        if not labels:
            return SceneContext(summary="no candidate concepts", visible_entities=[],
                                frames_analyzed=0, raw={"backend": "siglip"})
        frames = _sample_frames(Path(video_path), num_frames)
        if not frames:
            raise RuntimeError("no frames sampled")
        mdl, proc = _load(model, device)
        prompts = [_short_prompt(VISIBLE_CONCEPTS[l]) for l in labels]

        scores = {l: -1e9 for l in labels}
        for img in frames:
            inputs = proc(text=prompts, images=img, return_tensors="pt",
                          padding="max_length", truncation=True).to(device)
            with torch.no_grad():
                # per-pair logits: sigmoid loss means each pair is scored on its own,
                # so adding a concept cannot dilute the others (unlike CLIP's softmax)
                logits = mdl(**inputs).logits_per_image[0]
            for i, l in enumerate(labels):
                scores[l] = max(scores[l], float(logits[i]))

        visible = sorted(l for l, p in scores.items() if p >= threshold)
        print(f"       [stage2/siglip] {len(frames)} frames, visible: "
              f"{visible or 'none'}")
        return SceneContext(
            summary=("visible sources: " + ", ".join(visible)) if visible
            else "no target sources visible",
            visible_entities=visible, frames_analyzed=len(frames),
            raw={"backend": "siglip", "model": model, "threshold": threshold,
                 "scores": {k: round(v, 4) for k, v in scores.items()}})
    except Exception as e:
        print(f"       [stage2/siglip] unavailable ({type(e).__name__}: {e}); "
              f"no visibility info (gate will augment all).")
        return SceneContext(summary="", visible_entities=[], frames_analyzed=0,
                            raw={"backend": "siglip", "error": str(e)})
