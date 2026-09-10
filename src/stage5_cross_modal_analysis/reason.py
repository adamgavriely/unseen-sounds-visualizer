"""Decide, per sound: is its source already on screen, and if not, what to depict.

The rule, in Adam's words: the video should CONTRIBUTE to the audio classification, not
replace it. What the system depicts is always the sound that was detected; the video's
only job is to make that depiction specific to this place, and to say when the sound
needs no picture because the viewer can already see what is making it.

An earlier version broke that rule structurally. It handed a vision model four frames
plus one line of text naming the sound, and a vision model describes images -- so the
pixels won and the sound line lost. A clip from Penguins of Madagascar, where a glass
breaks off screen, came back as "penguin in cockpit". That was not a hallucination: the
model reported exactly what it saw. It had been asked the wrong question.

So audio and video are handled separately and only then combined:

  1. the scene is described ONCE per clip, as TEXT
  2. each sound stays what the detector said it is
  3. the video is asked ONE question about pixels -- "can you see the thing making this
     sound?" -- which is the only question whose answer genuinely lives in the image
  4. a TEXT-ONLY step combines sound and scene, with the sound as subject and the scene
     as modifier, so a sentence about the scene cannot swamp a sentence about the sound
  5. the result is kept only if it is still closer to the sound than to the scene,
     falling back to the plain label -- which is what enforces "contribute, not replace"

Nothing here consults a list of known sounds. Every judgement is a question asked at run
time about the sound that was actually detected, so an unseen sound is handled the same
way as a familiar one.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import config

SCENE_PROMPT = (
    "Describe this scene in one short sentence: where it is, and what is happening. "
    "No more than 15 words."
)

# Asked over the frames spanning ONE sound, not the whole clip: a door slamming at
# second 3 is not visible in a frame from second 12. Phrased as "name it" rather than
# "is it visible?" because a yes/no question to a VLM collects agreement rather than
# evidence -- a name can be checked against the sound, a "yes" cannot.
VISIBLE_PROMPT = (
    "These frames are from the moment a sound of {label} was heard.\n"
    "Name the thing in these frames that is making that sound. Answer with a short "
    "noun phrase of at most 5 words. If nothing that could make that sound is visible "
    "in these frames, answer exactly: nothing."
)

DEPICT_PROMPT = (
    "A deaf viewer is watching a video and cannot hear it.\n"
    "A sound detector heard: {label}.\n"
    "The video scene is: {scene}.\n\n"
    "Describe ONE picture showing {label}, made specific to that scene. The picture "
    "must be of {label} itself. The scene only tells you what kind of {label} it is and "
    "where it would be. Do not describe the scene instead.\n"
    "Example: sound Water, scene a forest path, gives: a stream running through a "
    "forest.\n"
    "Answer with a short phrase of at most 8 words. No punctuation, no explanation."
)

MAX_WORDS = 8

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


def _ask(mdl, proc, prompt, images=None, max_new: int = 48) -> str:
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
    new = out[:, inputs["input_ids"].shape[1]:]
    text = proc.batch_decode(new, skip_special_tokens=True)[0].strip()
    # A generation that used its whole budget was CUT, not finished, and the cut lands
    # mid-word: "a stream running through a for" reached the image generator once. The
    # model cannot tell us this and the string does not look wrong, so the only reliable
    # signal is the token count, and the only safe repair is to drop the fragment.
    if new.shape[1] >= max_new and " " in text:
        text = text.rsplit(" ", 1)[0]
    return text


def _clean_phrase(text: str, max_words: int = MAX_WORDS) -> str:
    """One short phrase, cut at a word boundary.

    Two failures this fixes, both general rather than particular to a clip. The token
    budget used to land mid-word, so a verbose answer reached the generator as "Air
    horn: loud, sharp, metallic tone; tr". And a model that starts explaining or quoting
    runs past the first clause, so only the first clause is kept -- a picture is one
    thing, and everything after the first comma or colon is a second thing.
    """
    text = " ".join((text or "").split())
    for cut in (". ", "; ", ": ", ", ", " - ", " -- "):
        if cut in text:
            text = text.split(cut)[0]
    text = text.strip(" .\"" + chr(39))
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip(" ,.;:-")


def _sim(a: str, b: str, device: str = "cpu") -> float:
    """Cosine similarity of two phrases, 0.0 if the encoder is unavailable."""
    from src import text_similarity
    try:
        return float(text_similarity.similarity(
            [a], [b], model=getattr(config, "SIGLIP_MODEL", ""), device=device)[0][0])
    except Exception as e:                       # encoder missing -> fall back to text
        print("       [stage5] text encoder unavailable ("
              + type(e).__name__ + "): " + str(e), flush=True)
        return 0.0


def _about_the_sound(phrase: str, label: str) -> bool:
    """Cheap word-stem overlap: does the phrase name the sound in so many words?

    Kept as a fast path in front of the encoder, not as the decision. It is right when
    it fires and silent when it does not: "a stream running through a forest" is a
    correct depiction of Water and shares no word with it.
    """
    stop = {"a", "an", "the", "of", "or", "and", "in", "on", "with"}
    want = [w for w in "".join(c if c.isalnum() else " " for c in label.lower()).split()
            if w not in stop and len(w) > 2]
    got = "".join(c if c.isalnum() else " " for c in phrase.lower()).split()
    if not want:
        return True
    return any(any(g.startswith(w[:4]) for g in got) for w in want)


def _still_the_sound(phrase: str, label: str, scene: str, device: str = "cpu") -> bool:
    """Is this still a picture of the sound, or has it become a picture of the scene?

    The test is COMPARATIVE, which is what makes it general and what removes the need to
    pick a threshold: a depiction is kept only if it is closer to the sound than the
    bare scene description already was. In other words the depiction has to say
    something about the sound that the scene did not -- if it does not, it has not
    contributed to the audio, it has replaced it.

    That is Adam's rule made mechanical, and it is the only formulation tested here that
    gets every measured case right. An absolute bar on depiction-to-sound similarity
    does not: "a stream running through a forest" scores 0.76 against *Water* while "a
    cowboy hat" scores 0.65 against *Vehicle*, so no single cut separates them. Against
    their own scenes they separate cleanly, because the stream adds water to a forest
    path (0.76 > 0.69) and "Man on motorcycle drives past silver van" adds nothing at
    all to *Laughter* (0.20 < 0.44) -- which is exactly the depiction the previous
    yes/no adjudicator waved through, because as a sentence about the scene it was
    perfectly good.
    """
    if _about_the_sound(phrase, label):
        return True
    to_sound = _sim(phrase, label, device)
    scene_to_sound = _sim(scene, label, device) if scene else 0.0
    print("       [stage5] check '" + phrase + "': depiction=" + format(to_sound, ".3f")
          + " vs scene=" + format(scene_to_sound, ".3f") + " against " + label,
          flush=True)
    return to_sound > scene_to_sound


MAKES_SOUND_PROMPT = (
    "Does a {named} make a {label} sound? Answer yes or no."
)


def _sound_is_visible(label: str, frames, mdl, proc, device: str = "cpu"):
    """Ask the video whether the source of THIS sound can be seen at THIS moment.

    Visibility is the one judgement that genuinely belongs to the pixels, so it is the
    one question the vision model is asked about them. It used to be answered by a
    hand-written table of about thirty concepts in Stage 2, which meant any sound
    outside the table -- laughter, footsteps, a telephone, an owl -- could never be
    marked visible, and got a picture beside a video that was already showing it.

    Two questions, not one, because they are two different kinds of question. What is in
    the frame is perceptual and is asked of the frames; whether that thing makes this
    sound is world knowledge and is asked as plain text. Splitting them is what stops a
    loosely-related object from counting as the source: a rodeo clip answers "a cowboy
    hat" for *Vehicle*, and the second question rejects it without any list of which
    objects make which sounds.

    Returns (visible, what_it_named).
    """
    if not frames:
        return False, ""
    named = _clean_phrase(_ask(mdl, proc, VISIBLE_PROMPT.format(label=label),
                               images=frames, max_new=24), max_words=5)
    low = named.lower()
    if not named or low.startswith(("nothing", "none", "no ", "not ")):
        return False, named
    if _about_the_sound(named, label):
        return True, named
    reply = _ask(mdl, proc, MAKES_SOUND_PROMPT.format(named=named, label=label),
                 max_new=6).strip().lower()
    print("       [stage5] visible? " + label + " <- '" + named + "' -> " + reply,
          flush=True)
    return reply.startswith("y"), named


SAME_PICTURE_PROMPT = (
    "A deaf viewer is watching a video. They will be shown one picture per sound.\n"
    "Picture A, for the sound {label_a}: {a}\n"
    "Picture B, for the sound {label_b}: {b}\n"
    "Would these two pictures look the same and tell the viewer the same thing? "
    "Answer yes or no."
)


def _dedup(active, mdl, proc, device: str = "cpu") -> None:
    """Merge sounds that would be drawn as the same picture.

    PANNs emits label families -- Laughter, Snicker, Chuckle; Laughter, Belly laugh,
    Giggle -- and each one used to take a slot, so the viewer got three near-identical
    pictures of people laughing beside one clip. Comparing depiction strings for
    equality never caught it, because the strings differ.

    The comparison is between the DEPICTIONS, never the labels. Labels are single words,
    and SigLIP embeds single words too tightly to separate: measured on this project's
    own vocabulary, *Dog* and *Cat* score 0.90 while *Laughter* and *Snicker* score
    0.80, so a label-level threshold would merge the wrong pairs in the wrong order.
    Depictions are short descriptive phrases -- exactly what the encoder was trained on
    -- and they separate: near-synonyms land at 0.78-0.91 and genuinely different
    sounds in the SAME scene at 0.58-0.81.

    Those two ranges overlap, so the embedding is used as a cheap filter and not as the
    verdict. Above DEDUP_SIM the pair is merged outright; between DEDUP_ASK and
    DEDUP_SIM the model is asked, in words, whether the two pictures would say the same
    thing; below DEDUP_ASK nothing is asked at all. So the decision in the ambiguous
    band is made at run time by a model that knows what laughing and giggling are,
    rather than by a synonym table that would only ever list the families we happened to
    see. What the filter buys is that the question is asked a couple of times per clip
    instead of for every pair.
    """
    from src import text_similarity
    order = sorted(active, key=lambda x: -x.confidence)
    subs = [" ".join((s.subject or "").lower().split()) for s in order]
    sure = float(getattr(config, "DEDUP_SIM", 0.88))
    ask = float(getattr(config, "DEDUP_ASK", 0.60))
    mat = None
    try:
        mat = text_similarity.similarity(subs, subs,
                                         model=getattr(config, "SIGLIP_MODEL", ""),
                                         device=device)
    except Exception as e:
        print("       [stage5] dedup fell back to exact match ("
              + type(e).__name__ + "): " + str(e), flush=True)

    kept: List[int] = []
    for i, spec in enumerate(order):
        if not subs[i]:
            continue
        dup, score = None, 1.0
        for j in kept:
            if subs[i] == subs[j]:
                dup, score = j, 1.0
                break
            if mat is None:
                continue
            sim = mat[i][j]
            if sim >= sure:
                dup, score = j, sim
                break
            if sim >= ask:
                reply = _ask(mdl, proc, SAME_PICTURE_PROMPT.format(
                    label_a=order[j].event_label, a=subs[j],
                    label_b=spec.event_label, b=subs[i]), max_new=6).strip().lower()
                print("       [stage5] same picture? " + spec.event_label + " / "
                      + order[j].event_label + " sim=" + format(sim, ".2f")
                      + " -> " + reply, flush=True)
                if reply.startswith("y"):
                    dup, score = j, sim
                    break
        if dup is None:
            kept.append(i)
            continue
        spec.augment = False
        spec.subject = ""
        spec.image_prompt = ""
        spec.reason = ("same picture as " + order[dup].event_label
                       + " (similarity " + format(score, ".2f") + ")")
        print("       [stage5] merged " + spec.event_label + " into "
              + order[dup].event_label + " (" + format(score, ".2f") + ")", flush=True)


def decide_subjects(video_path, specs, transcript: str = "",
                    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    device: str = "cuda", frames_per_sound: int = 4) -> None:
    """Fill in spec.subject for every augmented sound, and drop the visible ones.

    ``transcript`` is accepted for interface stability but deliberately NOT fed to the
    depiction step. It was, and dialogue leaked straight into the pictures: a Hiccup
    became "Hiccup on the phone, slow down, how many". The transcript is evidence for
    the gate, not material for the illustrator, and any talky clip reproduces the bug.
    """
    from src.stage2_video_understanding import _sample_frames, _sample_frames_at
    active = [s for s in specs if s.augment]
    if not active:
        return
    mdl, proc = _load(model, device)
    sim_device = device if device == "cuda" else "cpu"

    frames = _sample_frames(Path(video_path), 4)
    scene = _ask(mdl, proc, SCENE_PROMPT, images=frames, max_new=40) if frames else ""
    scene = " ".join(scene.split())[:160] or "an unknown place"
    print("       [stage5] scene: " + scene, flush=True)

    # 1. visibility, per sound, on the frames spanning that sound
    if getattr(config, "VLM_VISIBILITY", True):
        for spec in active:
            span = max(0.4, spec.end - spec.start)
            times = [spec.start - 0.4 + span * k / max(1, frames_per_sound - 1)
                     for k in range(frames_per_sound)]
            win = _sample_frames_at(Path(video_path), times)
            seen, named = _sound_is_visible(spec.event_label, win, mdl, proc, sim_device)
            if seen:
                spec.augment = False
                spec.subject = ""
                spec.image_prompt = ""
                spec.reason = "source visible on screen (" + named + ") - stay silent"
                print("       [stage5] silent: " + spec.event_label + " is visible ("
                      + named + ")", flush=True)
        active = [s for s in specs if s.augment]
        if not active:
            print("       [stage5] every sound was already visible; nothing to add",
                  flush=True)
            return

    # 2. depiction: text-only, sound as subject, scene as modifier
    for spec in active:
        prompt = DEPICT_PROMPT.format(label=spec.event_label, scene=scene)
        phrase = _clean_phrase(_ask(mdl, proc, prompt, max_new=48))
        if phrase and _still_the_sound(phrase, spec.event_label, scene, sim_device):
            spec.subject = phrase
            spec.reason += " | depiction: " + phrase
        else:
            if phrase:
                print("       [stage5] rejected (not about " + spec.event_label + "): "
                      + phrase, flush=True)
            spec.subject = spec.subject or spec.event_label
        spec.image_prompt = spec.subject
        print("       [stage5] " + spec.event_label + " -> " + spec.subject, flush=True)

    # 3. one picture per picture, not one per label family
    _dedup(active, mdl, proc, sim_device)
