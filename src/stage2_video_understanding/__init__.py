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
#
# DEMOTED 2026-09-10 to a cheap first pass. This table can only ever mark a sound
# visible if somebody wrote that sound into it, so laughter, footsteps, a telephone and
# an owl were structurally incapable of being gated and got a picture beside a video
# that was already showing the source -- which is the system's central claim failing,
# not a gap in coverage. The real check is now stage5/reason.py, which asks the VLM to
# NAME the thing making each sound in the frames spanning that sound, with no list of
# sounds anywhere. What survives here is a fast, model-free suppression that runs before
# the VLM is loaded, and is still what the benchmark's seen/not-seen numbers are
# computed from. See notes sec:vlmvisibility.
VISIBLE_CONCEPTS = {
    "Dog": "a photo of a dog", "Cat": "a photo of a cat",
    # machinery is acoustically indistinguishable from traffic at this model scale, so
    # the VISIBILITY side is made forgiving: a visible machine suppresses the augmentation
    # instead of yielding a phantom off-screen car (notes sec:annotation)
    "Vehicle": "a photo of a car, bus, truck or machinery, engine or heavy equipment",
    "Water": "a photo of the sea, ocean waves, a river or a waterfall",
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
    "Gunshot": "a photo of soldiers or police firing guns in combat",
    "Explosion": "a photo of a large explosion with a fireball and smoke",
    "Glass": "a photo of broken glass or a shattered window",
    "Alarm": "a photo of an alarm device, smoke detector or warning light",
    "Thunder": "a photo of a dark stormy sky, storm clouds or lightning",
    "Saxophone": "a photo of a person playing a saxophone or brass instrument",
}
_DISTRACTORS = ["a photo of an indoor scene", "a photo of an empty street",
                "a photo of the sky", "a random photo of something else"]
# Relative visibility rule (see analyze_video): a concept counts as visible when it
# reaches VIS_RATIO of the best-scoring concept for that frame, and clears VIS_FLOOR.
VIS_RATIO = 0.55
VIS_FLOOR = 0.12
NEGATIVE_TEMPLATE = "a photo with no {thing} anywhere in it"
_NEG_NOUN = {
    "Vehicle": "car, truck or machine", "Water": "water, sea or river",
    "Crowd": "crowd of people", "Siren": "emergency vehicle",
    "Bell": "bell or bell tower", "Applause": "audience",
    "Gunshot": "gun or shooting", "Explosion": "explosion or fire",
    "Glass": "broken glass", "Alarm": "alarm device", "Dishes": "dishes or cutlery",
    "Cooking": "food cooking", "Footsteps": "people walking", "Engine": "machine or engine",
    "Fireworks": "fireworks", "Saxophone": "musician", "Insect": "insect",
    "Sheep": "sheep or goats", "Cattle": "cows", "Telephone": "telephone",
}

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


def _sample_frames_at(video_path: Path, times) -> List[Image.Image]:
    """Frames at specific timestamps, clamped to the clip.

    Stage 5 needs the moments AROUND a sound -- just before it starts, while it sounds,
    just after it ends -- because a sound is an event in time and its cause is often
    only legible from what changed. Even sampling across the whole clip cannot answer
    that; a door slam at second 3 is invisible in a frame from second 12.
    """
    dur = media_duration(video_path) or 0.0
    imgs = []
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(times):
            t = max(0.0, min(float(t), max(0.0, dur - 0.05)))
            fp = Path(td) / f"w{i}.jpg"
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
        # Each concept gets its OWN yes/no pair. Scoring a concept against generic
        # distractors made CLIP prefer any specific prompt (a sea clip came back with
        # "Pig" and "Explosion" visible); scoring it against a softmax over all
        # concepts made the answer depend on how many concepts exist. A matched
        # negative fixes both: the comparison is "X present" vs "X absent" only.
        prompts = [VISIBLE_CONCEPTS[l] for l in labels] + _DISTRACTORS

        # Visibility is a per-concept YES/NO question, not a "pick one of N" choice.
        # A softmax over every concept made the score depend on HOW MANY concepts
        # exist: growing the list from 20 to 33 diluted each probability and pushed
        # obviously-visible sources (a clip that is entirely sea) below threshold.
        # Instead score each concept against the distractors alone, so the result is
        # independent of the size of VISIBLE_CONCEPTS.
        # A softmax over all prompts ranks the concepts well, but its ABSOLUTE values
        # shrink as concepts are added (20 -> 33 pushed an all-sea clip below a fixed
        # 0.30 bar). Two alternatives failed: scoring against generic distractors made
        # every specific prompt win, and matched negatives fail because CLIP does not
        # represent negation. So keep the softmax and threshold RELATIVELY: a concept
        # is visible when it is a top contender for this frame, with a small absolute
        # floor to reject frames where nothing matches.
        visible = set()
        scores = {l: 0.0 for l in labels}
        for img in frames:
            inputs = processor(text=prompts, images=img, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
            lab_probs = [float(probs[i]) for i in range(len(labels))]
            top = max(lab_probs) if lab_probs else 0.0
            bar = max(VIS_FLOOR, VIS_RATIO * top)
            for i, l in enumerate(labels):
                p = lab_probs[i]
                scores[l] = max(scores[l], p)
                if p >= bar and top >= VIS_FLOOR:
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


def _analyze_visibility(video_path, backend: str = "clip", **kw) -> SceneContext:
    """Backend dispatcher: 'siglip' (default), 'clip' (baseline) or 'vlm' (GPU).

    Keeps callers (pipeline, benchmark/evaluate) agnostic of which gate is in use
    so the two can be compared by flipping one config flag.
    """
    if backend == "siglip":
        from src.stage2_video_understanding.siglip import analyze_video_siglip
        return analyze_video_siglip(
            video_path, num_frames=kw.get("num_frames", 6),
            model=kw.get("siglip_model", "google/siglip-base-patch16-224"),
            device=kw.get("device", "cpu"),
            threshold=kw.get("siglip_threshold", 0.30),
            candidates=kw.get("candidates"))
    if backend == "owlv2":
        from src.stage2_video_understanding.owl import analyze_video_owl
        return analyze_video_owl(
            video_path, num_frames=kw.get("num_frames", 4),
            model=kw.get("owl_model", "google/owlv2-base-patch16-ensemble"),
            device=kw.get("device", "cpu"),
            threshold=kw.get("owl_threshold", 0.20),
            candidates=kw.get("candidates"))
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


def analyze(video_path, backend: str = "clip", **kw) -> SceneContext:
    """Scene context: what is visible, AND what kind of place this is.

    The proposal asks Stage 2 for "high-level scene context ... contextual cues that
    help interpret the semantic meaning of the accompanying audio". Only the first half
    was implemented -- a detector answering "is this object on screen", which the gate
    consumes. For an OFF-SCREEN sound, the only kind this system depicts, that answer is
    always no, so the video contributed nothing to what actually got drawn.

    The setting is the missing half. The fireplace is not in frame, but the living room
    is, and that is what separates a crackle that is a fire from one that is a stream.
    Classifying it costs one extra pass and is what makes Stage 5 cross-modal for the
    sounds that matter, rather than only for the ones already visible.
    """
    scene = _analyze_visibility(video_path, backend=backend, **kw)
    if not kw.get("with_setting", True):
        return scene
    try:
        from src.stage2_video_understanding.scene import classify_setting, GROUP
        frames = _sample_frames(Path(video_path), kw.get("num_frames", 6))
        setting, conf, top = classify_setting(
            frames, device=kw.get("device", "cpu"),
            backend=kw.get("setting_backend", "siglip"))
        scene.setting = setting
        scene.raw = dict(scene.raw or {})
        scene.raw["setting_conf"] = round(float(conf), 3)
        scene.raw["setting_scores"] = top
        scene.raw["setting_group"] = GROUP.get(setting, "")
        if setting:
            print(f"       [stage2] setting: {setting} ({conf:.2f})")
    except Exception as e:
        print(f"       [stage2] setting unavailable ({type(e).__name__}: {e})")
    return scene
