"""Stage 2 backend: SAM 3 concept detection (v4, docs/history/preregistrations/prereg_v4.md, release v1.2.0).

Meta, November 2025: Promptable Concept Segmentation -- a short noun phrase in, every
matching instance out with a presence score. It answers the same question OWLv2 did
("is a named object in this frame?") with the recognised successor model; the concept
phrases are OWLv2's (owl.DETECT_QUERY) so the two are comparable one for one. The
score bar is SAM 3's own default (0.5, a calibrated presence probability); nothing is
tuned on our clips. Weights are gated on the Hub (accept the licence once, then
export HF_TOKEN).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.types import SceneContext
from src.stage2_video_understanding import VISIBLE_CONCEPTS, _sample_frames
from src.stage2_video_understanding.owl import DETECT_QUERY

MODEL = "facebook/sam3"
_MODEL = None


def _load(model_name: str, device: str):
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import Sam3Model, Sam3Processor
        proc = Sam3Processor.from_pretrained(model_name)
        mdl = Sam3Model.from_pretrained(model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        _MODEL = (mdl, proc)
    return _MODEL


def analyze_video_sam3(video_path: Path, num_frames: int = 4, model: str = MODEL,
                       device: str = "cuda", threshold: float = 0.5,
                       candidates: Optional[List[str]] = None) -> SceneContext:
    """Best presence score per candidate concept over the sampled frames."""
    try:
        import torch
        labels = [c for c in (candidates or list(VISIBLE_CONCEPTS)) if c in DETECT_QUERY]
        if not labels:
            return SceneContext(summary="no candidate concepts", visible_entities=[],
                                frames_analyzed=0, raw={"backend": "sam3"})
        frames = _sample_frames(Path(video_path), num_frames)
        if not frames:
            raise RuntimeError("no frames sampled")
        mdl, proc = _load(model, device)
        queries = [DETECT_QUERY[l] for l in labels]
        best = {l: 0.0 for l in labels}
        for img in frames:
            # one text per image: the frame repeated once per concept
            inputs = proc(images=[img] * len(queries), text=queries, return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(mdl.dtype)
            with torch.no_grad():
                out = mdl(**inputs)
            res = proc.post_process_instance_segmentation(
                out, threshold=0.05, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())
            for l, r in zip(labels, res):
                if len(r["scores"]):
                    best[l] = max(best[l], float(r["scores"].max()))
        visible = sorted(l for l, s in best.items() if s >= threshold)
        print(f"       [stage2/sam3] {len(frames)} frames, visible: {visible or 'none'}")
        return SceneContext(
            summary=("visible sources: " + ", ".join(visible)) if visible else "no target sources visible",
            visible_entities=visible, frames_analyzed=len(frames),
            raw={"backend": "sam3", "model": model, "threshold": threshold,
                 "scores": {k: round(v, 4) for k, v in best.items()}})
    except Exception as e:
        print(f"       [stage2/sam3] unavailable ({type(e).__name__}: {e}); "
              f"no visibility info (gate will augment all).")
        return SceneContext(summary="", visible_entities=[], frames_analyzed=0,
                            raw={"backend": "sam3", "error": str(e)})
