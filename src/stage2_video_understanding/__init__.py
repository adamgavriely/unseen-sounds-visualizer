"""Stage 2 - Video Understanding (lightweight, CLIP-based).

Answers "which candidate sound-sources are actually visible on screen?" so the
Stage-5 gate can stay silent on sounds whose source is already visible (the
seen/not-seen gate). Uses CLIP zero-shot over sampled frames -- small and
CPU-friendly. This is the v2-(a) lightweight version; a full VLM (Qwen2.5-VL)
is a later upgrade.

Sounds whose source is not a persistent visible object (thunder, explosion,
gunshot, wind, siren) are simply absent from VISIBLE_CONCEPTS, so they are never
marked visible -> always eligible to augment.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

from src.types import SceneContext
from src.stage1_audio_extraction import media_duration

# Candidate visible sources, keyed by the (consolidated) sound label so the gate's
# `ev.label.lower() in visible_entities` check matches directly.
VISIBLE_CONCEPTS = {
    "Dog": "a photo of a dog", "Cat": "a photo of a cat",
    # machinery is acoustically indistinguishable from traffic at this model scale, so
    # the VISIBILITY side is made forgiving: a visible machine suppresses the augmentation
    # instead of yielding a phantom off-screen car (notes sec:annotation)
    "Vehicle": "a photo of a car, bus, truck or machinery, engine or heavy equipment",
    "Water": "a photo of water, a river, waterfall or the sea",
    "Bird": "a photo of a bird", "Crowd": "a photo of a crowd of people",
    "Train": "a photo of a train", "Aircraft": "a photo of an airplane in the sky",
    "Helicopter": "a photo of a helicopter", "Horse": "a photo of a horse",
    "Fire": "a photo of fire or flames", "Rain": "a photo of rain falling",
    "Applause": "a photo of an audience clapping", "Bell": "a photo of a church, bell or bell tower",
    "Insect": "a photo of an insect", "Boat": "a photo of a boat or ship on water",
    "Siren": "a photo of an ambulance, police car or fire truck",
    "Fireworks": "a photo of fireworks exploding in the sky",
    # Added 2026-08-23: 54% of off-screen claims named a label with NO concept here,
    # so the gate could never mark them visible -- e.g. a video showing nothing but
    # sheep was still reported as "sheep heard, source not visible".
    "Sheep": "a photo of sheep or goats",
    "Cattle": "a photo of cows or cattle",
    "Pig": "a photo of pigs",
    "Gunshot": "a photo of a person firing a gun, soldiers shooting",
    "Explosion": "a photo of an explosion, blast or fireball",
    "Glass": "a photo of broken glass or a shattered window",
    "Alarm": "a photo of an alarm device, smoke detector or warning light",
    "Telephone": "a photo of a telephone or a person holding a phone",
    "Dishes": "a photo of dishes, plates, pots or cutlery",
    "Cooking": "a photo of food cooking in a pan on a stove",
    "Door": "a photo of a door",
    "Footsteps": "a photo of people walking or running",
    "Engine": "a photo of a machine or engine running",
    "Thunder": "a photo of a dark stormy sky, storm clouds or lightning",
    "Saxophone": "a photo of a person playing a saxophone or brass instrument",
}
_DISTRACTORS = ["a photo of an indoor scene", "a photo of an empty street",
                "a photo of the sky", "a random photo of something else"]

_CLIP = None  # lazy (model, processor)


def _get_clip(model_name: str, device: str):
    global _CLIP
    if _CLIP is None:
        import torch  # noqa
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        _CLIP = (model, processor)
    return _CLIP


def _sample_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    dur = media_duration(video_path)
    imgs = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(num_frames):
            t = dur * (i + 0.5) / num_frames
            fp = Path(td) / f"f{i}.jpg"
            try:
                subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video_path),
                                "-frames:v", "1", "-q:v", "3", str(fp)],
                               check=True, capture_output=True)
                imgs.append(Image.open(fp).convert("RGB").copy())
            except Exception:
                continue
    return imgs


def analyze_video(video_path: Path, num_frames: int = 6,
                  model: str = "openai/clip-vit-base-patch32", device: str = "cpu",
                  threshold: float = 0.30) -> SceneContext:
    try:
        import torch
        frames = _sample_frames(Path(video_path), num_frames)
        if not frames:
            raise RuntimeError("no frames sampled")
        clip_model, processor = _get_clip(model, device)
        labels = list(VISIBLE_CONCEPTS.keys())
        prompts = [VISIBLE_CONCEPTS[l] for l in labels] + _DISTRACTORS

        visible = set()
        scores = {l: 0.0 for l in labels}
        for img in frames:
            inputs = processor(text=prompts, images=img, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
            for i, l in enumerate(labels):
                p = float(probs[i])
                scores[l] = max(scores[l], p)
                if p >= threshold:
                    visible.add(l)
        vis = sorted(visible)
        summary = ("visible sources: " + ", ".join(vis)) if vis else "no target sources visible"
        print(f"       [stage2] CLIP: {summary}")
        return SceneContext(summary=summary, visible_entities=vis,
                            frames_analyzed=len(frames),
                            raw={"scores": {k: round(v, 3) for k, v in scores.items()},
                                 "threshold": threshold, "model": model})
    except Exception as e:
        print(f"       [stage2] CLIP unavailable ({type(e).__name__}: {e}); "
              f"no visibility info (gate will augment all).")
        return SceneContext(summary="", visible_entities=[], frames_analyzed=0,
                            raw={"error": str(e)})


def analyze(video_path, backend: str = "clip", **kw) -> SceneContext:
    """Backend dispatcher: 'clip' (CPU, v2-a) or 'vlm' (Qwen2.5-VL, GPU, v2-b).

    Keeps callers (pipeline, benchmark/evaluate) agnostic of which gate is in use
    so the two can be compared by flipping one config flag.
    """
    if backend == "vlm":
        from src.stage2_video_understanding.vlm import analyze_video_vlm  # lazy: GPU deps
        return analyze_video_vlm(
            video_path, num_frames=kw.get("num_frames", 6),
            model=kw.get("vlm_model", "Qwen/Qwen2.5-VL-7B-Instruct"),
            device=kw.get("device", "cuda"), candidates=kw.get("candidates"))
    return analyze_video(video_path, num_frames=kw.get("num_frames", 6),
                         model=kw.get("model", "openai/clip-vit-base-patch32"),
                         device=kw.get("device", "cpu"),
                         threshold=kw.get("threshold", 0.30))
