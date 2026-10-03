"""Classify the SETTING of a clip, independently of which objects are visible.

A design-review point, and it is the project's own thesis applied where it had not been: the
video constrains what an unseen sound plausibly is. A crackle indoors in a living room
is a fire; the same crackle beside a forest stream is water. PANNs confuses the two
routinely -- they are spectrally similar -- and nothing in the pipeline corrected it,
because Stage 2's output was only ever consulted to answer "is the source ON SCREEN",
which for an off-screen sound is always no.

Setting is a different question from visibility and needs its own classifier: the
fireplace is not in frame, but the living room is. CLIP zero-shot over a small list of
settings is enough, costs one extra model on top of whichever visibility backend is in
use, and gives Stage 5 something to reason with for exactly the sounds the system
exists to depict.
"""
from __future__ import annotations

from typing import List, Tuple

# Short name -> the CLIP prompt that detects it. Kept small and mutually distinct;
# a long list mostly splits probability mass between near-synonyms.
SETTINGS = {
    "living room": "a photo inside a living room at home",
    "kitchen": "a photo inside a kitchen",
    "bathroom": "a photo inside a bathroom",
    "office": "a photo inside an office or classroom",
    "shop": "a photo inside a shop, market or cafe",
    "city street": "a photo of a city street with buildings and traffic",
    "road": "a photo of a road or highway",
    "forest": "a photo of a forest or woodland",
    "field": "a photo of an open field, farm or countryside",
    "mountain": "a photo of mountains or hills",
    "river": "a photo of a river, stream or lake",
    "sea": "a photo of the sea, a beach or the coast",
    "garden": "a photo of a garden or park",
    "crowd venue": "a photo of a stadium, concert or large crowd",
    "workshop": "a photo of a workshop, factory or construction site",
    "transport hub": "a photo inside a train station or airport",
    "vehicle interior": "a photo taken inside a moving vehicle",
    "night outdoors": "a photo taken outdoors at night",
}

# Coarse groups, which is the granularity the depiction rules actually need.
GROUP = {
    "living room": "home", "kitchen": "home", "bathroom": "home", "office": "indoor",
    "shop": "indoor", "transport hub": "indoor", "vehicle interior": "indoor",
    "city street": "urban", "road": "urban", "crowd venue": "urban",
    "workshop": "work",
    "forest": "nature", "field": "nature", "mountain": "nature", "river": "nature",
    "sea": "nature", "garden": "nature", "night outdoors": "nature",
}

MIN_CONF = 0.22          # below this the setting is not worth acting on

# CLIP ViT-B/32 is a 2021 model and the weakest thing in this pipeline; the setting
# decides what an unseen sound gets depicted AS, so it deserves a stronger classifier.
# SigLIP so400m is markedly better at zero-shot scene recognition and is already used
# elsewhere in the project. CLIP stays as a cheap fallback.
MODELS = {
    "siglip": "google/siglip-so400m-patch14-384",
    "clip": "openai/clip-vit-base-patch32",
}


def classify_setting(frames, device: str = "cpu", backend: str = "siglip",
                     model: str = "") -> Tuple[str, float, dict]:
    """Best-matching setting across the sampled frames, its confidence, and top scores."""
    if not frames:
        return "", 0.0, {}
    repo = model or MODELS.get(backend, MODELS["siglip"])
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        names: List[str] = list(SETTINGS)
        prompts = [SETTINGS[n] for n in names]
        proc = AutoProcessor.from_pretrained(repo)
        net = AutoModel.from_pretrained(repo).to(device).eval()
        totals = {n: 0.0 for n in names}
        for img in frames:
            inputs = proc(text=prompts, images=img, return_tensors="pt",
                          padding="max_length" if "siglip" in repo else True,
                          truncation=True).to(device)
            with torch.no_grad():
                # both families expose logits_per_image; softmax picks the best of a
                # mutually exclusive set, which is what a setting is
                probs = net(**inputs).logits_per_image.softmax(dim=1)[0]
            for i, n in enumerate(names):
                totals[n] += float(probs[i])
        for n in totals:
            totals[n] /= len(frames)
        best = max(totals, key=totals.get)
        conf = totals[best]
        top = {k: round(v, 3) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:6]}
        return (best if conf >= MIN_CONF else ""), conf, top
    except Exception as e:
        print(f"       [stage2] setting classifier unavailable ({type(e).__name__}: {e})")
        return "", 0.0, {}
