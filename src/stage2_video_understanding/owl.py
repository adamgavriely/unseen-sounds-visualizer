"""Stage 2 backend: OWLv2 open-vocabulary DETECTION (v2-c).

Why this exists. Swapping CLIP for SigLIP barely moved gating accuracy (49.3% ->
50.2%), and the confusion matrix showed why: of 103 clips tagged seen_ambient,
67 were predicted unseen or mixed. The failure is not the embedding model, it is the
QUESTION. CLIP and SigLIP embed a whole frame, so "is the source of this sound in
shot?" gets answered by overall scene gist -- and a small ambulance at the end of a
street, a dog at the edge of frame, or sheep across a field are swamped.

OWLv2 is an open-vocabulary detector: it localises a named object and returns a
score for that object. That is precisely the gate's question. Spot checks on the
clips reported as misjudged: sheep 0.77, dog 0.67, sea 0.54 -- all correctly ON
screen, where both CLIP and SigLIP had called them off-screen.

Cost: ~12 s per frame on CPU, so this backend is intended for the GPU, where it is
roughly real-time. The same model is used by scripts/prescreen_owl.py (release v1.2.0) to filter the
tagging queue.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.types import SceneContext
from src.stage2_video_understanding import VISIBLE_CONCEPTS, _sample_frames

_MODEL = None

# canonical sound-source concept -> the phrase to detect
DETECT_QUERY = {
    "Vehicle": "a car or truck", "Train": "a train", "Aircraft": "an airplane",
    "Helicopter": "a helicopter", "Boat": "a boat", "Siren": "an emergency vehicle",
    "Dog": "a dog", "Cat": "a cat", "Bird": "a bird", "Horse": "a horse",
    "Sheep": "a sheep", "Cattle": "a cow", "Pig": "a pig",
    "Water": "water, a river or the sea", "Rain": "rain falling",
    "Thunder": "storm clouds and lightning", "Fire": "fire or flames",
    "Crowd": "a crowd of people", "Applause": "an audience",
    "Bell": "a church bell tower", "Insect": "an insect",
    "Fireworks": "fireworks in the sky", "Gunshot": "a person firing a gun",
    "Explosion": "an explosion", "Glass": "broken glass",
    "Alarm": "an alarm device", "Saxophone": "a musician playing an instrument",
}


def _load(model_name: str, device: str):
    global _MODEL
    if _MODEL is None:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        proc = Owlv2Processor.from_pretrained(model_name)
        mdl = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()
        _MODEL = (mdl, proc)
    return _MODEL


def analyze_video_owl(video_path: Path, num_frames: int = 4,
                      model: str = "google/owlv2-base-patch16-ensemble",
                      device: str = "cpu", threshold: float = 0.20,
                      candidates: Optional[List[str]] = None) -> SceneContext:
    """Detect whether each candidate sound-source object is present in any frame.

    ``candidates`` restricts the queries to the sounds actually detected in the audio,
    which is both faster and more precise; without it every known concept is queried.
    """
    try:
        import torch
        labels = [c for c in (candidates or list(VISIBLE_CONCEPTS))
                  if c in DETECT_QUERY]
        if not labels:
            return SceneContext(summary="no candidate concepts", visible_entities=[],
                                frames_analyzed=0, raw={"backend": "owlv2"})
        frames = _sample_frames(Path(video_path), num_frames)
        if not frames:
            raise RuntimeError("no frames sampled")
        mdl, proc = _load(model, device)
        queries = [DETECT_QUERY[l] for l in labels]

        best = {l: 0.0 for l in labels}
        for img in frames:
            inputs = proc(text=[queries], images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, threshold=0.05,
                target_sizes=torch.tensor([[img.height, img.width]]).to(device))[0]
            for score, lab in zip(res["scores"], res["labels"]):
                l = labels[int(lab)]
                best[l] = max(best[l], float(score))

        visible = sorted(l for l, s in best.items() if s >= threshold)
        print(f"       [stage2/owlv2] {len(frames)} frames, visible: {visible or 'none'}")
        return SceneContext(
            summary=("visible sources: " + ", ".join(visible)) if visible
            else "no target sources visible",
            visible_entities=visible, frames_analyzed=len(frames),
            raw={"backend": "owlv2", "model": model, "threshold": threshold,
                 "scores": {k: round(v, 4) for k, v in best.items()}})
    except Exception as e:
        print(f"       [stage2/owlv2] unavailable ({type(e).__name__}: {e}); "
              f"no visibility info (gate will augment all).")
        return SceneContext(summary="", visible_entities=[], frames_analyzed=0,
                            raw={"backend": "owlv2", "error": str(e)})
