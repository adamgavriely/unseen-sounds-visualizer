"""Decide what to depict: the SOUND, made specific by the scene.

The rule, in Adam's words: the video should CONTRIBUTE to the audio classification, not
replace it. What the system depicts is always the sound that was detected; the video's
only job is to make that depiction specific to this place.

The previous version broke that rule structurally. It handed a vision model four frames
plus one line of text naming the sound, and a vision model describes images -- so the
pixels won and the sound line lost. A clip from Penguins of Madagascar, where a glass
breaks off screen, came back as "penguin in cockpit". That was not a hallucination: the
model reported exactly what it saw. It had been asked the wrong question.

So audio and video are handled separately and only then combined:

  1. the scene is described ONCE per clip, as TEXT
  2. each sound stays what the detector said it is
  3. a TEXT-ONLY step combines them, with the sound as subject and the scene as
     modifier, so a sentence about the scene cannot swamp a sentence about the sound
  4. the result is rejected unless it is still about that sound, falling back to the
     plain label -- which is what enforces "contribute, not replace"
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

SCENE_PROMPT = (
    "Describe this scene in one short sentence: where it is, and what is happening. "
    "No more than 15 words."
)

DEPICT_PROMPT = (
    "A deaf viewer is watching a video and cannot hear it.\n"
    "A sound detector heard: {label}.\n"
    "The video scene is: {scene}.\n"
    "{speech}\n\n"
    "Describe ONE picture showing {label}, made specific to that scene. The picture "
    "must be of {label} itself. The scene only tells you what kind of {label} it is and "
    "where it would be. Do not describe the scene instead.\n"
    "Example: sound Water, scene a forest path, gives: a stream running through a "
    "forest.\n"
    "Answer with a short phrase of at most 8 words. No punctuation, no explanation."
)

_VLM = None


def _load(model: str, device: str):
    global _VLM
    if _VLM is None:
        # The generator is cached across clips, so from the second clip onwards it is
        # still on the card when the reasoner wants it. That cascade failed 7 of 8
        # demos with CUDA out of memory after the first one succeeded.
        try:
            from src.stage6_visual_augmentation import unload_generator
            unload_generator()
        except Exception:
            pass
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(model)
        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None).eval()
        _VLM = (mdl, proc)
    return _VLM


def unload():
    """Free the reasoner before Stage 6 loads the generator: they do not co-fit."""
    global _VLM
    if _VLM is None:
        return
    del _VLM
    _VLM = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ask(mdl, proc, prompt, images=None, max_new: int = 40) -> str:
    import torch
    content = [{"type": "image"} for _ in (images or [])]
    content.append({"type": "text", "text": prompt})
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)
    kw = {"text": [text], "return_tensors": "pt"}
    if images:
        kw["images"] = images
    inputs = proc(**kw).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)[0].strip()


def _about_the_sound(phrase: str, label: str) -> bool:
    """Is this still a picture of the detected sound, or has it become the scene?

    The guard that makes the video contribute rather than replace. A depiction sharing
    no word with the sound it is supposed to depict has changed the subject, and the
    plain label is used instead: a generic correct picture beats a specific wrong one
    when a deaf viewer is relying on it to know what they cannot hear.
    """
    stop = {"a", "an", "the", "of", "or", "and", "in", "on", "with"}
    want = [w for w in "".join(c if c.isalnum() else " " for c in label.lower()).split()
            if w not in stop and len(w) > 2]
    got = "".join(c if c.isalnum() else " " for c in phrase.lower()).split()
    if not want:
        return True
    for w in want:
        stem = w[:4]
        if any(g.startswith(stem) for g in got):
            return True
    return False


CHECK_PROMPT = (
    "Sound: {label}\nPicture: {phrase}\n"
    "Would this picture tell a deaf viewer that they are hearing {label}? "
    "Answer yes or no."
)


def _still_the_sound(phrase: str, label: str, mdl, proc) -> bool:
    """Word overlap first, then ask the model about the rest.

    Overlap alone is too strict: "a stream running through a forest" is a correct
    depiction of Water and shares no word with it, and rejecting it would throw away
    exactly the specificity the video is there to add. Overlap alone is also enough on
    its own when it hits, so the model is only consulted when it does not -- which is
    the small minority of cases, and the ones where meaning rather than spelling
    decides.
    """
    if _about_the_sound(phrase, label):
        return True
    reply = _ask(mdl, proc, CHECK_PROMPT.format(label=label, phrase=phrase),
                 max_new=6).strip().lower()
    return reply.startswith("y")


def decide_subjects(video_path, specs, transcript: str = "",
                    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    device: str = "cuda", frames_per_sound: int = 4) -> None:
    """Fill in spec.subject for every augmented sound, in place."""
    from src.stage2_video_understanding import _sample_frames
    active = [s for s in specs if s.augment]
    if not active:
        return
    mdl, proc = _load(model, device)

    frames = _sample_frames(Path(video_path), 4)
    scene = _ask(mdl, proc, SCENE_PROMPT, images=frames, max_new=40) if frames else ""
    scene = " ".join(scene.split())[:160] or "an unknown place"
    print("       [stage5] scene: " + scene, flush=True)

    speech = ("Someone is speaking: " + transcript.strip()[:120]) if transcript.strip()         else "Nobody is speaking."

    for spec in active:
        prompt = DEPICT_PROMPT.format(label=spec.event_label, scene=scene, speech=speech)
        phrase = " ".join(_ask(mdl, proc, prompt, max_new=24).split())
        phrase = phrase.strip(' ."' + chr(39))
        if phrase and _still_the_sound(phrase, spec.event_label, mdl, proc):
            spec.subject = phrase
            spec.reason += " | depiction: " + phrase
        else:
            if phrase:
                print("       [stage5] rejected (not about " + spec.event_label + "): "
                      + phrase, flush=True)
            spec.subject = spec.subject or spec.event_label
        spec.image_prompt = spec.subject
        print("       [stage5] " + spec.event_label + " -> " + spec.subject, flush=True)

    seen = set()
    for spec in sorted(active, key=lambda x: -x.confidence):
        key = " ".join((spec.subject or "").lower().split())
        if key and key in seen:
            spec.augment = False
            spec.subject = ""
            spec.image_prompt = ""
            spec.reason = "identical depiction to a louder sound"
            print("       [stage5] merged: " + spec.event_label, flush=True)
        elif key:
            seen.add(key)
