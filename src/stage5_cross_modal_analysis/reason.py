"""Decide what to depict by looking at the video, not by consulting a table.

The first version of cross-modal depiction used three hand-written tables: a list of
settings, a table of acoustically confusable pairs, and a table of phrasings per
setting. Adam's objection is correct and the proposal already said so -- Stage 5 is
specified as "combine the extracted scene context, spoken language, and detected audio
events into a unified semantic representation", with an LLM, not a lookup.

Presets failed exactly where presets fail. A motorcycle POV shot was classified
"vehicle interior" at 0.98 confidence -- true of the camera, false of the scene -- and
every depiction for that clip was then phrased "indoors" while the video showed an
outdoor street. No amount of extra categories fixes that; the category list was the bug.

So the model is shown FRAMES SPANNING THE SOUND -- before it starts, while it sounds,
and after it ends -- together with everything else known about the moment, and asked
what a deaf viewer should be shown. Frames from around the event rather than one still
because the sound is an event in time: a door slamming, a glass breaking and a crowd
reacting all look like their aftermath a second later.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

PROMPT = (
    "These frames are consecutive moments from one video, spanning a few seconds "
    "around a sound.\n"
    "A sound detector heard: {label}.\n"
    "Other sounds present: {others}.\n"
    "{speech}\n"
    "The thing making this sound is NOT visible in these frames -- that is why it needs "
    "illustrating for a deaf viewer.\n\n"
    "Using what the video shows about where this is happening, describe the ONE picture "
    "we should display beside the video so a deaf viewer understands what they are "
    "hearing. Be concrete and specific to this scene. If the detector's label looks "
    "wrong given what you can see, describe what the sound most likely actually is.\n"
    "Answer with a short phrase of at most 8 words. No punctuation, no explanation."
)

_VLM = None


def _load(model: str, device: str):
    global _VLM
    if _VLM is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(model)
        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None).eval()
        _VLM = (mdl, proc)
    return _VLM


def unload():
    """Free the VLM before Stage 6 loads the image generator: they do not co-fit."""
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


def _window(video_path: Path, start: float, end: float, n: int = 4,
            pad: float = 1.5) -> List:
    """Frames spanning the sound, from just before it to just after."""
    from src.stage2_video_understanding import _sample_frames_at
    lo = max(0.0, start - pad)
    hi = end + pad
    times = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]
    return _sample_frames_at(Path(video_path), times)


def decide_subjects(video_path, specs, transcript: str = "",
                    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    device: str = "cuda", frames_per_sound: int = 4) -> None:
    """Fill in spec.subject for every augmented sound, in place."""
    import torch
    active = [s for s in specs if s.augment]
    if not active:
        return
    mdl, proc = _load(model, device)
    others = ", ".join(dict.fromkeys(s.event_label for s in active)) or "none"
    speech = (f'Someone says: "{transcript.strip()[:200]}"' if transcript.strip()
              else "Nobody is speaking.")
    for spec in active:
        frames = _window(Path(video_path), spec.start, spec.end, frames_per_sound)
        if not frames:
            continue
        prompt = PROMPT.format(label=spec.event_label, others=others, speech=speech)
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=frames, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=24, do_sample=False)
        phrase = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip()
        phrase = phrase.strip(' ."\'\n').replace("\n", " ")
        if phrase:
            spec.subject = phrase
            spec.image_prompt = phrase
            spec.reason += f" | depiction from video: {phrase}"
            print(f"       [stage5] {spec.event_label} -> {phrase}", flush=True)
    # Two sounds the model described the same way are one thing to a viewer; this
    # replaces the hand-written synonym table with the model's own judgement.
    seen = set()
    for spec in sorted(active, key=lambda s: -s.confidence):
        key = (spec.subject or "").lower()
        if key and key in seen:
            spec.augment = False
            spec.subject = ""
            spec.image_prompt = ""
            spec.reason = "same depiction as a louder sound already shown"
        elif key:
            seen.add(key)
