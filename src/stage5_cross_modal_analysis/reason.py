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
    "Describe ONLY this sound, ignoring any others in the clip: {label}.\n"
    "{speech}\n"
    "The thing making this sound is NOT visible in these frames -- that is why it "
    "needs illustrating for a deaf viewer.\n\n"
    "Describe the ONE picture to display beside the video so a deaf viewer understands "
    "THE SOUND. The picture must show the thing MAKING the sound, or the action that "
    "produces it -- NOT the scene the viewer can already see. Use the video only to "
    "make that thing specific to this place. If the label looks wrong given what you "
    "can see, describe what the sound most likely actually is.\n"
    "Good: laughing people at a bus stop. Bad: a man walking near cars.\n"
    "Answer with a short phrase of at most 8 words. No punctuation, no explanation."
)

_VLM = None


def _load(model: str, device: str):
    global _VLM
    if _VLM is None:
        # The generator is cached across clips, so from the second clip onwards it is
        # still on the card when the reasoner wants it. That cascade failed 7 of 8
        # demos with "CUDA out of memory" after the first one succeeded.
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
    speech = (f'Someone says: "{transcript.strip()[:200]}"' if transcript.strip()
              else "Nobody is speaking.")
    for spec in active:
        frames = _window(Path(video_path), spec.start, spec.end, frames_per_sound)
        if not frames:
            continue
        prompt = PROMPT.format(label=spec.event_label, speech=speech)
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
    _drop_duplicates(active, mdl, proc)


DEDUP_PROMPT = (
    "A deaf viewer will see these pictures beside a video, one per sound:\n{items}\n\n"
    "Some may tell the viewer the same thing. List the numbers of the ones to REMOVE, "
    "keeping the single most informative of each duplicated group. Two entries are "
    "duplicates when a viewer would learn nothing new from the second.\n"
    "Answer with numbers separated by commas, or the word none."
)


def _drop_duplicates(active, mdl, proc) -> None:
    """Ask the model which depictions say the same thing, and drop those.

    String matching cannot do this. The motorcycle clip produced "people laughing in a
    car", "people laughing and clapping" and "children laughing and playing on
    sidewalk": three slots spent telling a viewer that people are laughing, sharing too
    few words for any similarity threshold to catch without also merging things that
    differ. Whether two pictures say the same thing is a judgement about meaning, which
    is what the model is for -- and it is the reason the synonym table was removed.
    """
    if len(active) < 2:
        return
    # Identical text first, deterministically. Three sounds came back as the very same
    # sentence -- "Woman laughing with hands on face" -- and all three were rendered,
    # because the only dedup was a model call that answered "none". No judgement is
    # needed to see that the same sentence twice is the same picture twice.
    seen, survivors = set(), []
    for spec in sorted(active, key=lambda x: -x.confidence):
        key = " ".join((spec.subject or "").lower().split())
        if key and key in seen:
            spec.augment = False
            spec.subject = ""
            spec.image_prompt = ""
            spec.reason = "identical depiction to a louder sound"
            print(f"       [stage5] merged (identical): {spec.event_label}", flush=True)
        else:
            seen.add(key)
            survivors.append(spec)
    active = survivors
    if len(active) < 2:
        return
    import torch
    items = "\n".join(f"{i + 1}. {s.subject}" for i, s in enumerate(active))
    msgs = [{"role": "user",
             "content": [{"type": "text", "text": DEDUP_PROMPT.format(items=items)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=24, do_sample=False)
    reply = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)[0].strip().lower()
    print(f"       [stage5] dedup judge said: {reply[:60]!r}", flush=True)
    if "none" in reply:
        return
    drop = set()
    for tok in reply.replace(".", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and 1 <= int(tok) <= len(active):
            drop.add(int(tok) - 1)
    # never empty the panel over a parsing surprise
    if not drop or len(drop) >= len(active):
        return
    for i in sorted(drop):
        spec = active[i]
        spec.augment = False
        spec.subject = ""
        spec.image_prompt = ""
        spec.reason = "same thing shown by another sound already"
        print(f"       [stage5] merged: {spec.event_label}", flush=True)
