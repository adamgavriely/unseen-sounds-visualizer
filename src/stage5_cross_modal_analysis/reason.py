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
  5. the result is kept only if the model, made to CHOOSE between this clip's sounds,
     reads the picture as this one -- which is what enforces "contribute, not replace"

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

# The depiction step gets the PLACE, not the scene sentence, and the difference is the
# whole reason this prompt exists. Handed "A man on a motorcycle drives past a silver
# van", the model glued that van onto every sound in the clip: Air horn became "Air horn
# blaring from silver van" and rendered as a picture of a van, which tells a deaf viewer
# nothing about a horn. The scene sentence names incidental objects, and a model asked to
# be specific will reach for them. A place cannot be glued on the same way -- it can only
# say what KIND of sound this is and where it would be, which is exactly the contribution
# the video is supposed to make: a stream in a forest, a tap in a kitchen.
PLACE_PROMPT = (
    "What kind of place is this? Answer with at most four words, naming the place only. "
    "Do not name any people or objects in it."
)

# Asked over the frames spanning ONE sound, not the whole clip: a door slamming at
# second 3 is not visible in a frame from second 12. Phrased as "name it" rather than
# "is it visible?" because a yes/no question to a VLM collects agreement rather than
# evidence -- a name can be checked against the sound, a "yes" cannot.
#
# The wording is load-bearing and was arrived at the hard way. Tightened to "name the
# thing you can SEE making that sound, and only if you can see it actually making it",
# the model answered "nothing" for every sound in all eight demo clips -- no suppression
# at all, which is the complaint this check exists to answer. Leading with the question
# instead of the instruction did the same. Asking plainly for a name, and letting the
# follow-up question reject the loose ones, is what produces evidence to work with.
VISIBLE_PROMPT = (
    "These frames are from the moment a sound of {label} was heard.\n"
    "Name the thing in these frames that is making that sound. Answer with a short "
    "noun phrase of at most 5 words. If nothing that could make that sound is visible "
    "in these frames, answer exactly: nothing."
)

# The one thing the frames ARE asked about the sound itself, and the bounds on it. The
# detector said Crowd; the picture was "people cheering" because the model guessed,
# while the frames plainly showed a crowd chanting with torches. Adam: "why not try to
# understand from the video?" So the frames are asked what KIND of the detected sound
# this is -- and only that. The label is fixed in the question, the answer is a
# qualifier that goes into the depiction prompt as the specific kind, and "unknown" is
# an explicit option. The frames can turn a Crowd into a chanting crowd; they cannot turn
# a Glass into a penguin, which is the failure this whole file exists to prevent.
KIND_PROMPT = (
    "A sound detector heard: {label}. These frames are from that moment."
    + chr(10) +
    "If the frames show what KIND of {label} this is, or who or what is making it, say "
    "so in at most 4 words. Name the kind of {label} only; do not describe anything "
    "else in the frames."
    + chr(10) +
    "If the frames do not show it, answer exactly: unknown."
)


# v2 (2026-09-24, panel of three, two rounds). The v1 question -- do the frames SHOW what kind it
# is -- is nearly always "no" for the sounds this pipeline draws, because the gate has already
# decided the source is off screen. What the frames do show is the setting, and the setting is
# what separates a steam locomotive from a subway train. The first clause keeps the partly visible
# case. The era cue is stated as a rule, not an example: an example sentence in a prompt leaks its
# content (see the note under DEPICT_PROMPT).
KIND_PROMPT_V2 = (
    "A sound detector heard: {label}. These frames are from that moment."
    + chr(10) +
    "If the frames show what kind of {label} this is, say so. Otherwise, the thing making "
    "the sound is out of view: from the place and the era the frames show, which kind of "
    "{label} would it be? An old or a modern setting means an old or a modern kind."
    + chr(10) +
    "Answer with at most 4 words naming the kind of {label} only. Do not name the place, a "
    "person, or anything that is not the source of {label}."
    + chr(10) +
    "If the frames tell you nothing about it, answer exactly: unknown."
)


def _kind_from_frames(label: str, frames, mdl, proc, seen_only: bool = False) -> str:
    if not frames or not getattr(config, "KIND_FROM_FRAMES", True):
        return ""
    # seen_only: the corroboration step needs evidence -- the frames SHOWING the source (v1). The v2
    # question INFERS a kind from the setting, and a guess must never count as proof that a faint
    # sound is real (picture panel, P2, round 1).
    kp = KIND_PROMPT if seen_only else (KIND_PROMPT_V2 if getattr(config, "KIND_ALWAYS", False) else KIND_PROMPT)
    ans = _clean_phrase(_ask(mdl, proc, kp.format(label=label), images=frames,
                             max_new=16), max_words=4)
    low = ans.lower()
    if not ans or low.startswith(("unknown", "none", "not ", "no ")):
        return ""
    # No name check here: it discarded "airplane" as a kind of Vehicle because the word
    # "vehicle" is not in it. The answer is only a hint in brackets; the depiction that
    # comes out of it still has to pass the forced choice against the clip's other
    # sounds, and that is where drift into the scene is caught.
    print("       [stage5] kind from frames: " + label + " -> " + ans, flush=True)
    return ans


# The label is a SOURCE noun and the sound is an EVENT, and the prompt has to ask for
# the event. AudioSet names things -- Glass, Crowd, Dog, Vehicle -- but what a deaf viewer
# needs to see is glass BREAKING, a crowd APPLAUDING, a dog BARKING. Asked for "a picture
# showing Glass", the model described a window pane, the generator drew a window pane,
# and Adam, correctly: "pictures of glass alone, which means nothing if it should be glass
# breaking". So the question is now what is HAPPENING to make the sound, at the moment it
# happens. The detector's most specific sub-label goes in too when there is one, because
# it often IS the event: Shatter under Glass, Applause under Crowd, Bark under Dog.
DEPICT_PROMPT = (
    "A deaf viewer is watching a video and cannot hear it."
    + chr(10) +
    "A sound detector heard: {label}{detail}."
    + chr(10) +
    "The video is set in: {scene}."
    + chr(10) + chr(10) +
    "Describe ONE picture of that sound HAPPENING: show the action that makes the sound, "
    "at the moment it makes it. Not the thing at rest -- glass breaking, not a glass; a "
    "dog barking, not a dog; rain falling, not clouds."
    + chr(10) +
    "Use the place only if it changes what the thing looks like; otherwise ignore it. "
    "Never describe the place instead of the action."
    + chr(10) +
    "Answer with 2 to 5 plain words naming the action. No adjectives, no adverbs, no "
    "scenery, no poetry, no punctuation."
    + chr(10) +
    "Name the SOURCE doing it, so the viewer sees where the sound comes from. If a "
    "specific kind is given in brackets, draw that kind and not a guess at another."
    + chr(10) +
    "Do not name the place in your answer unless the sound cannot be drawn without it. "
    "Do not add any object that is not the source of the sound."
)
# v2 (2026-09-24, docs/NIGHT_REPORT_2026-09-24.md). Of the 33 DEV pictures, 2 were blank ("Sky
# rumbles"), 1 was the mythological Siren ("Siren spinning"), 2 left out the object ("Fingers strike
# keyboard", "Man shaves face") and 3 were the wrong kind for the scene. The v1 prompt asks for the
# ACTION and bans adjectives, and between them the object that makes the sound -- and the one word
# saying which kind it is -- were prompted out. v2 puts the object first. Still no examples: the
# rule below this prompt holds.
DEPICT_PROMPT_V2 = (
    "A deaf viewer is watching a video and cannot hear it."
    + chr(10) +
    "A sound detector heard: {label}{detail}."
    + chr(10) +
    "The video is set in: {scene}."
    + chr(10) + chr(10) +
    "Describe ONE picture of that sound being made. Name FIRST the solid object that makes "
    "the sound -- something a person could point at -- and then what it is doing."
    + chr(10) +
    "The object is never the sky, the air, the weather, a feeling, or a body part on its "
    "own. For a sound made by weather, name the visible thing in the weather that makes it."
    + chr(10) +
    "If the name of the sound can also mean something else, name the device, animal or "
    "machine that makes this sound, so the picture cannot be read the other way."
    + chr(10) +
    "If the brackets say which kind of object it is, keep that one word."
    + chr(10) +
    "Answer with 3 to 6 plain words. No scenery, no poetry, no punctuation. Do not name "
    "the place unless the sound cannot be drawn without it. Do not add any object that is "
    "not the source of the sound."
)

# v3 (2026-09-24, picture panel of five, two rounds; Adam's blind ratings). v2's "name FIRST the
# solid object... do not add any object that is not the source" drew PARTS: an engine block for a car,
# a wheelset for a train, two pairs of hands for a clapping crowd, a desk bell for a fire alarm -- 4 of
# the 5 pictures Adam rated worse. What he rewarded was the whole recognisable source caught in the act
# (a rooster with its beak open yes, roosters with beaks closed no). v3 is given the SPECIFIC sound the
# detector heard (labels.choose_source) with its ontology chain, and never the place: the place is what
# painted palaces and streets into the pictures, and what the strip functions then had to scrub.
DEPICT_PROMPT_V3 = (
    "A deaf viewer is watching a video and cannot hear it."
    + chr(10) +
    "A sound detector heard: {source}. Where that sits among kinds of sound: {chain}."
    + chr(10) + chr(10) +
    "Describe ONE picture that shows this sound being made. Name the whole thing a person would "
    "point at as making it -- the whole animal, machine, vehicle, instrument or group of people, "
    "never one part of it -- and what it is doing at the moment it makes this sound."
    + chr(10) +
    "The action must be the one that makes THIS sound, not something else the same thing can do."
    + chr(10) +
    "If what the detector heard is the name of a sound rather than of a thing, name the thing that "
    "makes it. The thing is never the sky, the air, or a body part on its own. If it is weather, "
    "name the visible sign that appears at the same instant as the sound, not the weather around it. "
    "If a word could mean something else, add the word that says which kind of thing it is."
    + chr(10) +
    "Answer with 3 to 6 plain words. No place, no scenery, no other objects, no punctuation."
)
DEPICT_V3_RESTATE = ("Your answer must name the thing that makes the sound, which is filed under: "
                     "{chain}.")

# No example sentences, on purpose. They were there to teach a 7B model the format, but
# an example carries content as well as format and the content leaks: "a police car
# with its siren on" was copied over the detector's own "Civil defense siren". Adam's
# rule: the prompt is built from the detector, the audio and the video, not from
# anything written here in advance.

# Asked when the first answer drifted into the scene. Same event framing, place kept --
# a retry that dropped the place produced "People gathered together moving around" for a
# crowd in a talk-show studio, which is worse than the place-specific answer it replaced.
# Its result is accepted as it stands: this prompt cannot produce a scene sentence, and
# the old path of re-checking it and falling back to a bare label is exactly how "Glass"
# on its own reached the generator.
RETRY_PROMPT = (
    "A deaf viewer is watching a video and cannot hear it. A sound detector heard: "
    "{label}{detail}. The video is set in: {scene}."
    + chr(10) +
    "Describe ONE picture of {label} HAPPENING -- the action that makes that sound, at "
    "the moment it makes it. The picture must be of the sound being made, nothing else."
    + chr(10) +
    "Answer with 2 to 5 plain words naming the action. No adjectives, no scenery, no "
    "punctuation."
)

# Speech as gate context, done the way the proposal meant and not the way it was first
# tried. The transcript never reaches a picture. It answers one question per sound that
# has speech near it: are the people on the soundtrack reacting to this sound? A "yes"
# means the sound matters to what is happening -- "did you hear that?", "turn that alarm
# off", "is that a siren?" -- and a sound that matters is (a) never the one dropped when
# more sounds overlap than the panel can carry, and (b) shown even if the detector's
# confidence was marginal, because confidence measures loudness and people reacting to a
# quiet sound is better evidence that it matters than the decibel level is.
#
# It does NOT override visibility. Adam's rule: if the siren is on screen, the viewer can
# see it, and it does not matter that they are also talking about it.
SPEECH_PROMPT = (
    "A sound of {label} was heard in a video. Around that moment, someone said:"
    + chr(10) +
    "\"{speech}\""
    + chr(10) +
    "Are they (a) {opt_a}, or (b) {opt_b}? Answer with the letter only."
)
SPEECH_YES = "reacting to that sound or talking about it"
SPEECH_NO = "not referring to that sound"
SPEECH_WINDOW = 3.0   # seconds of speech before and after the sound that count as "around"
# Fewer words than this and the question is not asked. Whisper transcribes non-speech
# as syllables -- "Lachon. La." for laughter, "oh" for an owl -- and the model then
# matched them lexically, calling "La." a reaction to Laughter and "oh" a reaction to a
# Hoot, twice each through the double-ask. An interjection is not evidence that anyone
# is reacting to anything; a sentence might be.
SPEECH_MIN_WORDS = 4

MAX_WORDS = 6

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
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        dmap = "auto" if device == "cuda" else None
        if "Qwen2.5-VL" in model:
            from transformers import Qwen2_5_VLForConditionalGeneration
            mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, torch_dtype=dtype, device_map=dmap).eval()
        else:
            # Qwen3.x and other native vision-language models (transformers >= 5.8)
            from transformers import AutoModelForImageTextToText
            mdl = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype, device_map=dmap).eval()
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
    # Thinking (Qwen3.x): the model reasons first, then answers; the pipeline reads only the
    # answer. Off by default; config.VLM_THINKING = True turns it on (several times slower).
    thinking = bool(getattr(config, "VLM_THINKING", False))
    if thinking:
        prompt = prompt + " Think it through, then end with only the final answer on the last line."
    content = [{"type": "image"} for _ in (images or [])]
    content.append({"type": "text", "text": prompt})
    try:
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:                                   # templates without the switch (Qwen2.5)
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    kw = {"text": [text], "return_tensors": "pt"}
    if images:
        kw["images"] = images
    inputs = proc(**kw).to(mdl.device)
    budget = max(max_new, int(getattr(config, "VLM_THINKING_TOKENS", 1024))) if thinking else max_new
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=budget, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    text = proc.batch_decode(new, skip_special_tokens=True)[0].strip()
    if thinking:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        # the answer is the last non-empty line; "Final answer:" prefixes are dropped
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = lines[-1] if lines else text
        for pre in ("final answer:", "answer:"):
            if text.lower().startswith(pre):
                text = text[len(pre):].strip()
        max_new = budget
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
    # A multilingual model occasionally finishes a phrase in another script -- "Hoot owl
    # perched branch宫殿" reached the generator once. The image model reads English, so
    # anything outside the Latin range is dropped rather than passed through.
    text = "".join(c if ord(c) < 0x250 else " " for c in (text or ""))
    text = " ".join(text.split())
    for cut in (". ", "; ", ": ", ", ", " - ", " -- "):
        if cut in text:
            text = text.split(cut)[0]
    text = text.strip(" .\"" + chr(39))
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    # A word cap can land on a word that cannot end a phrase, and the result is a prompt
    # that trails off: "Owl perched on a tree branch outside the", "A large group of
    # people gathered outside a". Whatever followed is gone either way, so the dangling
    # connective goes too -- it adds nothing to the picture and reads as a bug.
    while words and words[-1].lower().strip(",.;:") in _DANGLING:
        words.pop()
    return " ".join(words).strip(" ,.;:-")


_DANGLING = {"a", "an", "the", "of", "in", "on", "at", "to", "with", "and", "or",
             "for", "from", "by", "near", "over", "under", "into", "onto", "as",
             "that", "this", "its", "their", "his", "her", "while", "during"}


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
    # Whole-word match with plain inflections, not a 4-letter stem: the stem let Rain
    # pass on "raincoat", Wind on "window", Bird on "birthday", skipping validation.
    def same(g, w):
        return g == w or g == w + "s" or g == w + "es" or g == w + "ing" or g == w + "ed"             or (w.endswith("e") and g == w[:-1] + "ing") or (len(w) > 4 and g == w[:-1] + "ies")
    return any(any(same(g, w) for g in got) for w in want)


CHOOSE_PROMPT = (
    "A deaf viewer is shown this picture: {phrase}"
    + chr(10) +
    "Which sound does the picture tell them is happening?"
    + chr(10) + "{options}" + chr(10) +
    "Answer with the letter only."
)


def _reads_as(phrase: str, label: str, others, mdl, proc) -> bool:
    """Would a viewer seeing this picture know which sound it is?

    A FORCED CHOICE, not a yes/no. Asking a model "would this picture tell a deaf viewer
    they are hearing Laughter?" is an agreement task, and it approved "Man on motorcycle
    drives past silver van". Asking which of several sounds the picture shows is a
    discrimination task, and the wrong answer is available to be chosen -- including
    "none of them", so the model is not forced to pick something.

    The options are the other sounds detected in this same clip, which is what makes the
    question hard in the right way: a depiction that has drifted into the scene tends to
    be equally true of every sound in it, and one that names the wrong source loses to
    the sound it actually depicts. Nothing is listed in advance; the alternatives come
    from what the detector heard in this video.
    """
    base = [label] + [o for o in others if o != label][:4]
    letters = "abcdef"
    # Asked twice with the correct option in two different positions, and both must
    # pick it. One rotated ask let "gun recoil" through as a picture of Sigh, and the
    # real Explosion in that clip was then merged into the phantom. Every other
    # two-way question here already requires agreement across orderings; this one
    # has more options but the same 7B model and the same letter habits.
    picks = []
    for k in (0, 1):
        options = list(base)
        slot = (sum(ord(c) for c in label) + k) % len(options)
        options[0], options[slot] = options[slot], options[0]
        body = chr(10).join("(" + letters[i] + ") " + o for i, o in enumerate(options))
        body += chr(10) + "(" + letters[len(options)] + ") none of them"
        reply = _ask(mdl, proc, CHOOSE_PROMPT.format(phrase=phrase, options=body),
                     max_new=6).strip().lower().lstrip("(")
        picks.append(reply[:1] == letters[slot])
    ok = all(picks)
    print("       [stage5] reads as " + label + "? '" + phrase + "' -> "
          + ("yes" if ok else "no") + " (votes "
          + "/".join("y" if v else "n" for v in picks) + ")", flush=True)
    return ok


def _still_the_sound(phrase, label, others, mdl, proc) -> bool:
    """Is this still a picture of the sound, or has it become a picture of the scene?

    Adam's rule made mechanical: the scene may shape the depiction, but the moment the
    scene IS the depiction, the video has replaced the audio instead of contributing to
    it.

    Two steps. If the phrase names the sound outright, keep it; otherwise the model is
    made to CHOOSE which of this clip's sounds the picture shows.

    The overlap step was removed once, on the reasoning that a spelling test should not
    overrule a meaning test -- the depiction step does sometimes write the sound's own
    word into a sentence about the scene, and "Man on motorcycle drives past LAUGHING
    silver van" was skipping validation as a picture of Laughter. Measured, that change
    was a clear loss and it was put back. It cost two good depictions in eight clips --
    "A corded telephone on a bedside table" rejected as a picture of Telephone, "Owl
    perched on a tree branch outside" rejected as a picture of Owl, both falling back to
    bare labels -- and it bought nothing, because the forced choice endorsed the laughing
    van anyway. A phrase that names its own sound is about that sound often enough to be
    worth trusting, and asking a 7B model a question it can get wrong is not free.

    A third formulation was tried and discarded, and it is worth recording because it
    looked right: score the depiction against the sound with SigLIP, and keep it only if
    it beat the bare scene description's own score against that sound. It gets the first
    eight cases right and then falls apart -- 11 of 17 on a harder set, missing almost
    every depiction that should have been rejected. The reason is that it was not
    measuring aboutness at all. A short, specific phrase scores higher against a
    one-word label than a long scene sentence does, whatever either one means, so the
    test was mostly reading phrase length. "Penguin in cockpit" beat its own scene as a
    picture of Glass (0.586 against 0.496); "a group of people laughing in a courtroom"
    lost as a picture of Laughter (0.428 against 0.459). Embeddings still do the
    deduplication, where both sides of every comparison are the same kind of string and
    the measurement is sound.
    """
    if _about_the_sound(phrase, label):
        return True
    if mdl is None:
        return False
    return _reads_as(phrase, label, others or [], mdl, proc)


# Written as two labelled lines rather than as one sentence. Interpolating a noun phrase
# into "does a {named} make a {label} sound?" produced "Does a The woman in white make a
# Laughter sound?", and the model answered "no" to the grammar rather than to the
# question -- so a talk-show clip kept three pictures of laughter beside a video of the
# people doing the laughing. Labelled fields cannot be mangled by whatever the naming
# step happens to return.
MAKES_SOUND_PROMPT = (
    "Sound heard: {label}"
    + chr(10) +
    "Thing visible in the video: {named}"
    + chr(10) +
    "Could that thing be what is making that sound? Answer yes or no."
)


def _ab(mdl, proc, question: str, opt_yes: str, opt_no: str, frames=None):
    """One a/b question asked in both orderings. Returns True / False / None (split).

    The single-letter answer format has a position bias in this model (it answered "(b)"
    22 of 23 times on one question, whatever (b) was), so the options are swapped
    between the two asks and only an answer that survives the swap counts. A split
    means the model cannot tell from these frames, and the caller decides what to do
    with "cannot tell" -- usually ask an open question instead.
    """
    votes = []
    for flip in (False, True):
        a, b = (opt_no, opt_yes) if flip else (opt_yes, opt_no)
        want = "b" if flip else "a"
        reply = _ask(mdl, proc, question + chr(10) + "(a) " + a + chr(10) + "(b) " + b
                     + chr(10) + "Answer with the letter only.",
                     images=frames, max_new=6).strip().lower().lstrip("(")
        votes.append(reply[:1] == want)
    if all(votes):
        return True
    if not any(votes):
        return False
    return None


DESCRIBE_PROMPT = (
    "Describe what is happening in these frames in one sentence. Name what you see "
    "and what it is doing. No more than 25 words."
)


LAST_VOTES: dict = {}
VOTE_LOG: list = []          # every (sound, stretch) verdict of the current clip, see decide_subjects


def _sound_is_visible(label: str, frames, mdl, proc, device: str = "cpu"):
    """Can the viewer SEE this sound happening? Three questions, combined.

    Adam: "combine both questions, a/b and an open question, to help determine in case
    of disagreement." So:

      open    name the thing making the sound (or "nothing")
      a/b     is the sound happening on screen, asked in both orderings
      open    if those two disagree, describe the frames in a sentence, and see
              whether the description names the sound

    The open naming question alone missed rain: asked what was making the rain, the
    model named the sky, the world-knowledge check said a sky does not make rain, and a
    video of rain got a picture of rain beside it. The a/b question alone has the letter
    bias. Each covers the other's blind spot, and the description settles the rest.

    Returns (visible, evidence).
    """
    if not frames:
        return False, ""
    named = _clean_phrase(_ask(mdl, proc, VISIBLE_PROMPT.format(label=label),
                               images=frames, max_new=24), max_words=5)
    low = named.lower()
    by_name = None
    if named and not low.startswith(("nothing", "none", "no ", "not ")):
        if _about_the_sound(named, label):
            by_name = True
        else:
            reply = _ask(mdl, proc, MAKES_SOUND_PROMPT.format(named=named, label=label),
                         max_new=6).strip().lower()
            by_name = reply.startswith("y")
    else:
        by_name = False
        named = "nothing"

    by_ab = _ab(mdl, proc,
                "These frames are from the moment a sound of " + label + " was heard. "
                "Judge from the frames alone.",
                "you can SEE " + label + " happening on screen -- the source is in the "
                "frames and visibly making that sound",
                label + " is not visibly happening in these frames",
                frames=frames)

    # Three independent readings of the same frames, majority wins. A cascade (decide
    # on two, consult the third only on a split) flipped between runs on a half-second
    # change in frame timing; three votes every time is steadier, and costs one call.
    desc = _clean_phrase(_ask(mdl, proc, DESCRIBE_PROMPT, images=frames, max_new=48),
                         max_words=25)
    # The description names things, not sounds: "a police car with its lights" is the
    # siren's source and "a gun on the back seat" is the gunshot's, and neither contains
    # the sound's word. So a description that does not name the sound outright is put to
    # the world-knowledge question -- could what it describes be making this sound?
    by_desc = _about_the_sound(desc, label) or (
        named != "nothing" and by_name is True and _about_the_sound(desc, named))
    if not by_desc and desc:
        reply = _ask(mdl, proc, MAKES_SOUND_PROMPT.format(named=desc, label=label),
                     max_new=6).strip().lower()
        by_desc = reply.startswith("y")
    votes = [by_name, by_ab, by_desc]
    # exposed for benchmark/gate_dev_sweep.py, which re-decides from the raw votes;
    # nothing on the inference path reads it
    global LAST_VOTES
    LAST_VOTES = {"name": by_name, "ab": by_ab, "desc": by_desc, "named": named}
    yes = sum(1 for v in votes if v is True)
    no = sum(1 for v in votes if v is False)
    # the rule is set once on the dev split (benchmark/gate_dev_sweep.py): majority (v3)
    # or unanimous -- config.VISIBILITY_RULE
    verdict = (yes == 3) if getattr(config, "VISIBILITY_RULE", "majority") == "unanimous" else yes > no
    how = ("name=" + ("yes:" + named if by_name else "no") + " a/b="
           + {True: "yes", False: "no", None: "split"}[by_ab] + " desc="
           + ("yes" if by_desc else "no") + " ['" + desc[:60] + "']")
    print("       [stage5] visible? " + label + " -> " + ("yes" if verdict else "no")
          + " (" + how + ")", flush=True)
    return verdict, named


def _subtract(fam, mem, min_len: float = 1.0):
    """The family's bursts with the member's bursts cut out of them.

    Overlap was the wrong test: birdsong ran 0.5-26 s as one burst, an owl hooted at
    4-5 s inside it, and one second of overlap "explained" twenty-five seconds of bird.
    Interval subtraction keeps 0.5-4 and 5-26 as Bird. Pieces shorter than `min_len`
    are dropped as edge slivers.
    """
    pieces = list(getattr(fam, "spans", None) or [(fam.start, fam.end)])
    for b0, b1 in list(getattr(mem, "spans", None) or [(mem.start, mem.end)]):
        nxt = []
        for a0, a1 in pieces:
            if b1 <= a0 or b0 >= a1:
                nxt.append((a0, a1))
            else:
                if b0 - a0 >= min_len:
                    nxt.append((a0, b0))
                if a1 - b1 >= min_len:
                    nxt.append((b1, a1))
        pieces = nxt
    return pieces


def _overlap(a, b, slack: float = 1.0) -> bool:
    """Do any bursts of these two sounds coincide (within `slack` seconds)?"""
    sa = list(getattr(a, "spans", None) or [(a.start, a.end)])
    sb = list(getattr(b, "spans", None) or [(b.start, b.end)])
    return any(x0 - slack <= y1 and y0 - slack <= x1 for x0, x1 in sa for y0, y1 in sb)


def _dedup(active, mdl, proc, device: str = "cpu") -> None:
    """Merge sounds that would be drawn as the same picture.

    PANNs emits label families -- Laughter, Snicker, Chuckle; Laughter, Belly laugh,
    Giggle -- and consolidate_families does not catch them all, so a single clip produced
    three slots holding three near-identical pictures of people laughing. The previous
    deduplication compared depiction strings for equality and never fired, because the
    strings differ.

    Similarity is measured between DEPICTIONS, never labels. Labels are single words and
    SigLIP embeds them too tightly to separate: measured on this project's own
    vocabulary, Dog/Cat scores 0.90 while Laughter/Snicker scores 0.80, so a label-level
    threshold merges the wrong pairs first. Depictions are descriptive phrases, which is
    what the encoder was trained on.

    THE THRESHOLD DOES THIS ALONE, and that conclusion was expensive. The design was for
    the embedding to filter and a model to decide the ambiguous band, because the ranges
    overlap. Over three demo runs the model was asked 40-odd times, in two different
    framings, and it answered "different" to roughly 90% of pairs -- including an owl on
    a branch against the hoot of an owl (0.82) and a woman laughing against a woman
    snickering (0.76). It was not being agreeable and it was not favouring a letter: two
    differently worded pictures ARE two different pictures, and Owl and Hoot are not two
    names for one sound if you read the words strictly. Its handful of "same" answers
    were arbitrary, and one of them merged "Baby laughing in a silver van's shadow" into
    "Dog barking near a silver van" because they share a van.

    So the band is decided by the number, with the bar set from what three runs actually
    produced:

        true duplicates      0.57 - 0.87   (Chuckle/Laughter 0.87, Owl/Hoot 0.82)
        true non-duplicates  0.46 - 0.63   (Siren/Hoot 0.63, Dog/Laughter 0.59)

    At 0.70 that is three of five duplicates merged and NO false merges. The two misses
    both sit at 0.57, inside the non-duplicate range, so no threshold can reach them and
    the model could not either -- it called both of them different too. Pairs above
    DEDUP_REPORT are logged rather than merged, so the next run's numbers keep arriving
    and the bar can be moved on evidence instead of taste.
    """
    from src import text_similarity
    from src.labels import same_source
    order = sorted(active, key=lambda x: -x.confidence)
    subs = [" ".join((s.subject or "").lower().split()) for s in order]
    sure = float(getattr(config, "DEDUP_SIM", 0.70))
    report = float(getattr(config, "DEDUP_REPORT", 0.45))
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
        dup, score, why = None, 1.0, "identical"
        for j in kept:
            # The detector's own taxonomy first: if one label is a more specific kind of
            # the other AND they overlap in time, they are one source. Time matters: an
            # Owl at 2 s and a Bird at 19-27 s are two events, and merging them by name
            # alone swallowed seven seconds of birdsong into a picture shown at second 2.
            if same_source(spec.event_label, order[j].event_label):
                # A family and one of its members. Two cases the ontology cannot tell
                # apart, and the depictions can:
                #   Owl / Hoot  -- one sound under two names ("Owl hoots", "Owl hooting
                #                  in tree"): one picture, shown at BOTH sounds' times
                #   Bird / Owl  -- two sounds in one family ("Bird chirping", "Owl
                #                  hoots"): the member keeps its bursts, the family keeps
                #                  whatever is left after subtracting them
                # Adam: "bird is bird and owl is owl; one is hooting and one is chirping."
                from src.labels import is_descendant
                fam, mem = ((spec, order[j]) if is_descendant(order[j].event_label, spec.event_label)
                            else (order[j], spec))
                alike = mat is not None and mat[i][j] >= sure
                if alike:
                    dup, score, why = j, mat[i][j], "same sound under two names"
                    break
                left = _subtract(fam, mem)
                if left:
                    fam.spans = left
                    fam.start, fam.end = left[0]
                    print("       [stage5] " + fam.event_label + " keeps "
                          + str(len(left)) + " burst(s) outside " + mem.event_label,
                          flush=True)
                    continue              # both stay; nothing merged
                dup, score, why = j, 1.0, "fully explained by " + mem.event_label
                break
            if subs[i] == subs[j]:
                dup, score, why = j, 1.0, "identical depiction"
                break
            if mat is None:
                continue
            sim = mat[i][j]
            # Two depictions that read alike are one picture only if they are heard at
            # the same time. An owl at 1 s absorbed a crow at 8 s on wording alone.
            if sim >= sure and _overlap(spec, order[j]):
                dup, score, why = j, sim, "similarity " + format(sim, ".2f")
                break
            if sim >= report:
                print("       [stage5] near-duplicate? " + spec.event_label + " / "
                      + order[j].event_label + " sim=" + format(sim, ".2f")
                      + " (bar " + format(sure, ".2f") + ")", flush=True)
        if dup is None:
            kept.append(i)
            continue
        # the survivor shows during BOTH sounds' bursts, not just its own
        keep_spec = order[dup]
        mine = list(getattr(spec, "spans", None) or [(spec.start, spec.end)])
        theirs = list(getattr(keep_spec, "spans", None) or [(keep_spec.start, keep_spec.end)])
        keep_spec.spans = sorted(set(theirs + mine))
        spec.augment = False
        spec.subject = ""
        spec.image_prompt = ""
        spec.reason = ("same picture as " + order[dup].event_label + " (" + why + ")")
        print("       [stage5] merged " + spec.event_label + " into "
              + order[dup].event_label + " (" + why + ")", flush=True)


# Cross-modal disambiguation among the DETECTOR'S OWN candidates -- Adam's original
# example, "fire crackling can sound like water if you don't consider the video", in
# the one form that keeps the rule that the video contributes and never replaces.
#
# When the detector gives two different labels to what is plainly one acoustic event --
# same start, same end, neither a kind of the other -- it is hedging between things it
# cannot tell apart by ear: bleating and a baby crying, crackling and running water.
# The audio has proposed the candidates; the video may pick between them. It may NOT
# add a candidate the detector did not hear, and if the frames do not settle it, both
# stay. So the label is never changed by the pixels, only chosen from what the audio
# already said.
#
# "Same event" is decided by timing, not by meaning: two spans whose starts and ends
# both fall within SAME_EVENT_TOL of each other. Genuinely simultaneous sounds -- a
# siren over a crowd -- rarely share both boundaries; one sound with two names always
# does, because both names came from the same windows.
DISAMBIG_PROMPT = (
    "A sound detector heard ONE sound at this moment and could not decide between two "
    "labels. Look at the frames from that moment."
    + chr(10) +
    "Which is it? (a) {a}  (b) {b}  (c) cannot tell from the frames"
    + chr(10) +
    "Answer with the letter only."
)
SAME_EVENT_TOL = 0.6


def _disambiguate(specs, video_path, mdl, proc, frames_per_sound: int = 4) -> None:
    from src.labels import same_source
    from src.stage2_video_understanding import _sample_frames_at
    live = [s for s in specs if s.augment]
    done = set()
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if id(a) in done or id(b) in done:
                continue
            if not a.augment or not b.augment:
                continue
            if same_source(a.event_label, b.event_label):
                continue                       # one source, handled by dedup
            if (abs(a.start - b.start) > SAME_EVENT_TOL
                    or abs(a.end - b.end) > SAME_EVENT_TOL):
                continue                       # two events, not one with two names
            span = max(0.4, a.end - a.start)
            times = [a.start - 0.4 + span * k / max(1, frames_per_sound - 1)
                     for k in range(frames_per_sound)]
            win = _sample_frames_at(Path(video_path), times)
            if not win:
                continue
            # both orderings, agreement required -- the 7B model has a letter bias
            picks = []
            for first, second in ((a, b), (b, a)):
                reply = _ask(mdl, proc, DISAMBIG_PROMPT.format(
                    a=first.event_label, b=second.event_label), images=win,
                    max_new=6).strip().lower().lstrip("(")[:1]
                picks.append({"a": first, "b": second}.get(reply))
            winner = picks[0] if picks[0] is not None and picks[0] is picks[1] else None
            print("       [stage5] one sound, two names? " + a.event_label + " / "
                  + b.event_label + " -> "
                  + (winner.event_label if winner else "kept both"), flush=True)
            if winner is None:
                continue
            loser = b if winner is a else a
            loser.augment = False
            loser.subject = ""
            loser.image_prompt = ""
            loser.reason = ("same sound as " + winner.event_label
                            + "; the frames say it is that one")
            done.add(id(loser))


_PREPS = {"in", "on", "at", "near", "inside", "outside", "by", "beside", "across", "through", "along"}


def _source_chain(source: str, family: str) -> list:
    """The source and its ontology ancestors up to (and including) the family it is gated as."""
    from src.labels import ancestors, canonical
    chain = [source]
    # stop at the family, at the first ancestor that is itself a gated family, or after three links:
    # walking to the ontology root put "Sounds of things" and "Livestock, farm animals" in the prompt
    # (P1, round 3)
    for a in ancestors(source)[:3]:
        if a == family or canonical(a) == a:
            break
        chain.append(a)
    if family not in chain:
        chain.append(family)
    return chain


def _depict_v3(spec, place: str, mdl, proc) -> str:
    """v3 depiction: the specific source, its chain, no place. Never drops a sound -- dropping is a
    gate decision and v3 changes only what is drawn."""
    src = getattr(spec, "source", "") or spec.event_label
    chain = _source_chain(src, spec.event_label)
    chain_txt = " > ".join(chain)
    keep = " ".join(chain + [spec.detail or ""])
    prompt = DEPICT_PROMPT_V3.format(source=src, chain=chain_txt)

    def one(p):
        ph = _clean_phrase(_ask(mdl, proc, p, max_new=48))
        ph = _without_place(ph, place, keep=keep)             # backstops only: no place was given
        return _drop_place_phrase(ph, place, spec.event_label, keep)
    phrase = one(prompt)
    # backstop, no model judgement: an answer that names nothing in the source's chain is asked once
    # more with the chain restated; the restated answer is taken only if it does name it
    if phrase and not any(_about_the_sound(phrase, x) for x in chain):
        again = one(prompt + chr(10) + DEPICT_V3_RESTATE.format(chain=chain_txt))
        if again and any(_about_the_sound(again, x) for x in chain):
            print("       [stage5] v3 restated: " + phrase + " -> " + again, flush=True)
            phrase = again
    return phrase


# V3.1, PICTURE_SCENE (2026-09-25, panel of five, two rounds). V3's subjects on 50 never-annotated clips,
# read before any picture was drawn, showed two failures. It INVENTED what the audio never established
# (a Whoosh became "a person swinging a sword", a bare Siren "a police car", a Thunk "a heavy door") and
# it LOST what the scene had established (a Vehicle on a farm had been a tractor, now "a car driving
# down the road"). The scene is therefore let back in, but only as a qualifier on the word the detector
# heard -- "car door", "farm vehicle" -- never as a new noun, and every check on it is a list, not a
# question to the model that wrote the answer (P1, P3, P4: the model's own yes/no passes every sword).
RESOLVE_PROMPT = (
    "A sound detector heard: {source}. What makes it may be out of view. These frames show where "
    "the video is."
    + chr(10) +
    "If these frames and this place make one kind of {head} clearly more likely than any other, "
    "answer with one or two words that say which kind, followed by the word {head}. Otherwise answer "
    "exactly: {head}."
    + chr(10) +
    "Never name a person, and never replace the word {head} with another thing."
)

DEPICT_PROMPT_V31 = (
    "A deaf viewer is watching a video and cannot hear it."
    + chr(10) +
    "A sound detector heard: {thing}. Where that sits among kinds of sound: {chain}."
    + chr(10) + chr(10) +
    "Describe ONE picture that shows this sound being made. Name the whole thing that makes it -- the "
    "whole animal, machine, vehicle or instrument, never one part of it -- and what it is doing at the "
    "moment it makes this sound."
    + chr(10) +
    "The action must be the one that makes THIS sound, not something else the same thing can do."
    + chr(10) +
    "If {thing} is the name of a sound and the words above do not say what makes it, show the visible "
    "effect of the sound instead of guessing what makes it. Show a person only if the sound comes from "
    "a person's own body or voice. The thing is never the sky or the air on its own. If it is weather, "
    "name the visible sign that appears at the same instant as the sound."
    + chr(10) +
    "Never name a more specific kind than {thing}. If a word could mean something else, choose the "
    "meaning that makes a sound."
    + chr(10) +
    "Answer with 3 to 6 plain words. No place, no scenery, no other objects, no punctuation."
)
DEPICT_V31_GUARD = "Do not name any more specific kind than: {thing}."


def _scene_thing(source: str, family: str, frames, place: str, fired, mdl, proc) -> str:
    """RESOLVE: the heard source, with at most two qualifier words from the scene. Accepted only when
    the answer still contains the heard word and uses no kind the audio did not establish."""
    from src.labels import label_names, names_forbidden, is_descendant
    head = (label_names(source) or [source.lower()])[0]
    if not frames:
        return head
    ans = _clean_phrase(_ask(mdl, proc, RESOLVE_PROMPT.format(source=source, head=head),
                             images=frames, max_new=16), max_words=4)
    low = " ".join("".join(c if c.isalnum() else " " for c in ans.lower()).split())
    # a fired kind UNDER the source was not chosen for a reason (a tie, or under the floor): the
    # scene may not settle it either (P1, round 2)
    ok_fired = [x for x in (fired or []) if not is_descendant(x, source)]
    bad = names_forbidden(low, source, ok_fired)
    words = low.split()
    if (not low or not low.endswith(head) or len(words) > len(head.split()) + 2 or bad
            or any(w in {"person", "man", "woman", "people", "someone", "hand", "hands"} for w in words)):
        if low and low != head:
            print("       [stage5] scene kind refused: " + source + " -> " + ans
                  + (" (names " + ", ".join(bad) + ")" if bad else ""), flush=True)
        return head
    if getattr(config, "PICTURE_SCENE_GUARD2", False) and low != head:
        from src.labels import other_branch_makers
        alien = other_branch_makers(words[: len(words) - len(head.split())], source)
        if alien:
            # a qualifier that is itself a sound source in another branch of the ontology is a new maker,
            # not a kind of this one: Screaming -> "cat screaming", found on slice B (prereg amendment 2b)
            print("       [stage5] scene kind refused (other-branch maker " + ", ".join(alien) + "): "
                  + source + " -> " + ans, flush=True)
            return head
    if low != head:
        print("       [stage5] scene kind: " + source + " -> " + low + " (place: " + place + ")", flush=True)
    return low


def _depict_v31(spec, place: str, frames, fired, mdl, proc) -> str:
    """V3.1 depiction: RESOLVE's thing, the source's chain, no place; the same list guard on the answer."""
    from src.labels import names_forbidden, is_descendant
    src = getattr(spec, "source", "") or spec.event_label
    thing = _scene_thing(src, spec.event_label, frames, place, fired, mdl, proc)
    chain = _source_chain(src, spec.event_label)
    chain_txt = " > ".join(chain)
    keep = " ".join(chain + [thing])
    ok_fired = [x for x in (fired or []) if not is_descendant(x, src)]
    prompt = DEPICT_PROMPT_V31.format(thing=thing, chain=chain_txt)

    def one(p):
        ph = _clean_phrase(_ask(mdl, proc, p, max_new=48))
        ph = _without_place(ph, place, keep=keep)
        return _drop_place_phrase(ph, place, spec.event_label, keep)
    phrase = one(prompt)
    # a person is drawn only for a human sound: "A person whooshing a flag" for a Whoosh on the 50
    # in-sample clips (the prompt's own "show a person only if" was ignored), so it is a list check too
    from src.labels import ancestors
    human = "Human sounds" in ancestors(src) or src == "Human sounds"
    people = {"person", "man", "woman", "people", "someone", "boy", "girl", "child", "hand", "hands"}

    def has_person(ph):
        return (not human) and bool(people & set("".join(c if c.isalnum() else " " for c in ph.lower()).split()))
    if phrase and has_person(phrase):
        again = one(prompt + chr(10) + "Do not show any person.")
        print("       [stage5] v3.1 person guard: " + phrase + " -> " + again, flush=True)
        phrase = again if again and not has_person(again) else ""
    bad = names_forbidden(phrase, src, ok_fired) if phrase else []
    if phrase and bad:
        again = one(prompt + chr(10) + DEPICT_V31_GUARD.format(thing=thing))
        print("       [stage5] v3.1 guard (" + ", ".join(bad) + "): " + phrase + " -> " + again, flush=True)
        phrase = again if again and not names_forbidden(again, src, ok_fired) else ""
    if phrase and not any(_about_the_sound(phrase, x) for x in chain + [thing]):
        again = one(prompt + chr(10) + DEPICT_V3_RESTATE.format(chain=chain_txt))
        ok = again and not names_forbidden(again, src, ok_fired)
        if ok and any(_about_the_sound(again, x) for x in chain + [thing]):
            print("       [stage5] v3.1 restated: " + phrase + " -> " + again, flush=True)
            phrase = again
    if phrase and getattr(config, "PICTURE_SCENE_GUARD2", False):
        # the drawn phrase must still name the heard thing: "fireworks artillery fire" became "Fireworks
        # exploding in the night sky" and the artillery vanished (slice B, prereg amendment 2b)
        # any word of the SOURCE's own name counts (not the qualifier, not only the last word: "A bird
        # singing" names Bird vocalization; "Fireworks exploding" names nothing of Artillery fire)
        from src.labels import label_names
        names = [w for x in chain for w in (label_names(x) or [x.lower()])]
        if not any(_about_the_sound(phrase, n) for n in names):
            print("       [stage5] v3.1 head word lost (" + "/".join(names) + "): " + phrase + " -> " + thing,
                  flush=True)
            phrase = ""
    return phrase or thing


# PICTURE_MAKER (Adam, 28 Sept 2026): "If the sound is already an object (ambulance, bird, frog, car) it is easy. If it
# is an action (honk, chirp, knock, bang...), the picture must show the OBJECT that makes the sound. If several objects
# could make it, use context from the video to choose." Audit of the 82 shipped pictures: 5 subjects named only the
# action ("laughter" x2, "applause", "run", "typing"). Cause: every guard in _depict_v31 that empties the phrase falls
# back to the bare head word, and the head-word guard misses inflections and synonyms ("A man laughing" does not name
# "laughter", "People clapping" does not name "applause"). Fix: for the action labels below, a subject that names none
# of the makers gets one. One maker -> its fixed phrase. Several makers (ontology or word lists) -> the VLM picks one
# from the sound's own frames (same model, same frames as RESOLVE; one closed question, asked only here), exact match
# against the list; unsure / no match -> the maker the subject already names, else the default (first in the list).
# Key: the drawn source (spec.source) or the event label. Value: [(maker words, fixed subject)], default first.
_PERSON = ("person", "man", "woman", "people", "child", "boy", "girl", "men", "women", "children")
MAKERS = {
    "Laughter": [(_PERSON, "a person laughing")],
    "Giggle": [(_PERSON, "a person giggling")],
    "Chuckle, chortle": [(_PERSON, "a person chuckling")],
    "Belly laugh": [(_PERSON, "a person laughing loudly")],
    "Snicker": [(_PERSON, "a person snickering")],
    "Baby laughter": [(("baby", "infant"), "a baby laughing")],
    "Applause": [(_PERSON + ("audience", "crowd", "hands"), "an audience clapping their hands")],
    "Clapping": [(_PERSON + ("hands",), "two hands clapping")],
    "Run": [(_PERSON + ("runner", "feet"), "a person running")],
    "Walk, footsteps": [(_PERSON + ("feet", "shoes", "walker"), "a person walking")],
    "Typing": [(("keyboard",), "hands typing on a computer keyboard"),
               (("typewriter",), "hands typing on a typewriter")],
    # AudioSet "Honk" is the GOOSE's call (Goose > Fowl > Animal); "honk" is also the word lists' car horn
    # (verify.AMBIGUOUS "horn"). Two makers, so the frames choose. Default: the ontology's goose.
    "Honk": [(("goose", "geese"), "a goose honking with its beak open"),
             (("car", "bus", "truck", "vehicle", "van", "taxi"), "a car sounding its horn")],
    "Toot": [(("car",), "a car sounding its horn"), (("bus",), "a bus sounding its horn"),
             (("truck", "lorry"), "a truck sounding its horn"), (("motorcycle", "scooter"), "a motorcycle sounding its horn")],
    "Vehicle horn, car horn, honking": [(("car",), "a car sounding its horn"), (("bus",), "a bus sounding its horn"),
                                        (("truck", "lorry"), "a truck sounding its horn"),
                                        (("motorcycle", "scooter"), "a motorcycle sounding its horn")],
}
MAKER_PROMPT = (
    "A sound detector heard: {sound}. What makes it may be out of view. These frames show where the video is."
    + chr(10) +
    "Which of these most likely made this sound here? {options}"
    + chr(10) +
    "If none of them is visible or suggested by the place, or you are not sure, answer exactly: unsure. "
    "Otherwise answer with one option exactly as written."
)
MAKER_LOG: list = []          # one entry per changed subject (clip-agnostic; the caller adds the clip)


def _maker_named(phrase: str, words) -> bool:
    got = set("".join(c if c.isalnum() else " " for c in (phrase or "").lower()).split())
    return any(w in got or w + "s" in got for w in words)


def with_maker(spec, phrase: str, frames, mdl, proc) -> str:
    """PICTURE_MAKER: the subject for an action sound names the object that makes it (see MAKERS)."""
    src = getattr(spec, "source", "") or spec.event_label
    makers = MAKERS.get(src) or MAKERS.get(spec.event_label)
    if not makers:
        return phrase
    named = [i for i, (words, _) in enumerate(makers) if _maker_named(phrase, words)]
    pick, how = None, ""
    if len(makers) > 1 and frames and mdl is not None:
        names = [m[0][0] for m in makers]
        ans = _clean_phrase(_ask(mdl, proc, MAKER_PROMPT.format(sound=src, options=", ".join(names)),
                                 images=frames, max_new=8), max_words=3).lower()
        if ans in names:
            pick, how = names.index(ans), "frames"
        print("       [stage5] maker from frames: " + src + " -> " + (ans or "-"), flush=True)
    if pick is None and named:
        return phrase                                    # unsure: keep the maker the subject already names
    if pick is not None and pick in named:
        return phrase                                    # the frames agree with the subject
    if pick is None:
        pick, how = 0, ("default" if len(makers) > 1 else "only maker")
    new = makers[pick][1]
    MAKER_LOG.append({"label": spec.event_label, "source": src, "old": phrase, "new": new, "how": how})
    print("       [stage5] maker (" + how + "): " + src + ": " + repr(phrase) + " -> " + repr(new), flush=True)
    return new


# GP-4 two-step prompt (five-panel, 2026-09-25): the frames choose the thing (RESOLVE, above); a TEXT-ONLY
# rewrite may then lengthen the prompt to the 40-80 words these generators were trained on, describing only
# HOW the thing looks at the moment it makes the sound. It never sees the frames and never chooses the noun;
# a mechanical word check refuses any other sound source, person or place, and on any refusal the short
# prompt is used (never a retry).
EXPAND_PROMPT = (
    "Rewrite this picture description as one detailed caption of 40 to 60 words for an image generator: "
    "{subject}."
    + chr(10) +
    "Describe only how this thing looks at the moment it makes its sound: the whole of it fully in frame, "
    "its pose or motion, the visible effect of the sound (for example what moves, splashes, breaks or "
    "flashes), and the framing: one large subject, centred, on a plain white background."
    + chr(10) +
    "Do not add any other object, animal, person or place. Answer with the caption only."
)
EFFECT_WORDS = {"pieces", "shards", "splash", "splashes", "droplets", "drops", "flash", "flashes", "smoke",
                "steam", "sparks", "dust", "debris", "ripples", "waves", "light", "lights", "glow", "motion",
                "lines", "blur", "air", "water", "fragments", "feathers", "beak", "mouth", "wings", "legs",
                "wheels", "tail", "head", "body", "background", "white", "frame", "centre", "center"}
PEOPLE = {"person", "man", "woman", "people", "someone", "boy", "girl", "child", "children", "hand", "hands",
          "driver", "rider", "player", "worker", "soldier", "farmer"}
PLACES = {"street", "road", "room", "kitchen", "forest", "city", "building", "house", "field", "park", "beach",
          "sky", "farm", "station", "shop", "office", "garden", "mountain", "river", "lake", "sea", "town",
          "village", "yard", "interior", "landscape", "scenery", "night"}


def expand_guard(text: str, subject: str, source: str, family: str) -> list:
    """Words of the long prompt that name a sound source, person or place the short subject does not."""
    from src.labels import _parents, label_names, ancestors
    allowed = set()
    for x in [source, family, *ancestors(source)]:
        for n in label_names(x):
            allowed |= set(n.split())
    words = set("".join(c if c.isalnum() else " " for c in text.lower()).split())
    subj = set("".join(c if c.isalnum() else " " for c in subject.lower()).split())
    allowed |= subj | EFFECT_WORDS
    vocab = set()
    for lab in _parents():
        for n in label_names(lab):
            if " " not in n:
                vocab.add(n)
    bad = sorted(w for w in words - allowed if w in vocab or w in PEOPLE or w in PLACES
                 or (w.endswith("s") and (w[:-1] in vocab or w[:-1] in PEOPLE)))
    if "person" in subj or "people" in subj:          # a human-sound subject keeps its person words
        bad = [w for w in bad if w not in PEOPLE]
    return bad


def expand_prompt(subject: str, source: str, family: str, mdl, proc) -> tuple:
    """-> (long prompt or "", refused words). Text only; one call; no retry."""
    raw = _ask(mdl, proc, EXPAND_PROMPT.format(subject=subject), max_new=120)
    text = " ".join(raw.replace(chr(10), " ").split()).strip().strip('"')
    if not text or len(text.split()) < 15:
        return "", ["(empty)"]
    bad = expand_guard(text, subject, source, family)
    return ("" if bad else text), bad


def _drop_place_phrase(phrase: str, place: str, label: str = "", detail: str = "") -> str:
    """DEPICT_V2 only: remove a trailing "in / on / at ... <place>" phrase.

    Under v2 the answer leads with the object, so the place is never needed to make the object
    drawable, and when the model adds it anyway ("Crowd cheering in palace", "Firecracker exploding
    on street") the generator paints the place -- two crowds in front of a palace on DEV. Scoped to
    the clip's OWN place answer, not a word list: the phrase goes only if every content word after
    the preposition is a word of that answer, so no object can ever be removed.
    """
    stop = {"a", "an", "the", "of"}
    words = phrase.split()
    placew = {w.lower().strip(",.") for w in place.replace(",", " ").split()} - stop
    for k in range(len(words) - 1, 0, -1):
        if words[k].lower() in _PREPS:
            tail = [w.lower().strip(",.") for w in words[k + 1:] if w.lower() not in stop]
            # never strip a word that names the sound or its source ("Knock on door" keeps the door,
            # even in a clip whose place answer happens to contain it) -- reviewer C, round 4
            own = {w.lower().strip(",.()") for w in (label + " " + detail).replace(",", " ").split()}
            if (tail and all(w in placew or w.rstrip("s") in placew for w in tail)
                    and not any(w in own or w.rstrip("s") in own for w in tail)
                    and k >= 2 and words[k - 1].lower() not in _PREPS):
                return " ".join(words[:k])
            break
    return phrase


def _without_place(phrase: str, place: str, keep: str = "") -> str:
    """Drop the place's own words from a depiction, if something is left.

    The prompt says not to name the place unless the sound cannot be drawn without it,
    and the model names it anyway: "Palace window cracks", "a palace window breaks".
    Adam: the palace is an assumption. A deterministic strip is the only thing a 7B
    model reliably obeys. Kept whole if stripping would leave fewer than two words --
    "a stream in a forest" for Water in a forest is a place the sound needs.
    """
    stop = {"a", "an", "the", "of", "in", "on", "at"}
    bad = {w for w in place.lower().replace(",", " ").split() if w not in stop and len(w) > 2}
    # the kind of the SOURCE, read from the frames, is not the place: "subway" in "subway train"
    # is what makes the generator draw a subway train and not a steam locomotive
    bad -= {w for w in keep.lower().replace(",", " ").split()}
    if not bad:
        return phrase
    kept = [w for w in phrase.split() if w.lower().strip(",.") not in bad
            and w.lower().strip(",.").rstrip("s") not in bad]
    if len(kept) < 2 or kept == phrase.split():
        return phrase
    # "a stream in a forest" minus "forest" is "a stream in a": the place was the object
    # of a preposition, i.e. the sound needed it. A dangling connective means keep whole.
    if kept[-1].lower().strip(",.") in _DANGLING:
        return phrase
    return " ".join(kept).strip()


# The place may veto a sound that does not belong in it -- with two bounds.
#
# This is an assumption, made at run time from the frames rather than from a list, and
# Adam asked for no assumptions. It went in anyway because of what the random demo set
# showed: a Horse at 0.84 in a quarry-blast clip, an Ice cream truck at 0.63 on a train
# platform, a Train at a construction site. Confident detector errors cannot be caught by
# any threshold, and the only information that separates them from a real surprising
# off-screen sound is whether the sound fits the place. Bounds: never a sound people are
# reacting to (speech beats the prior), never above PLAUSIBLE_MAX (a very confident
# surprising sound is exactly what a hearing viewer would react to). a/b in both
# orderings; a split is settled by an open question -- what sounds would you expect
# here -- and the sound stays unless that list plainly excludes it.
PLAUSIBLE_MAX = 0.90


def _fits_the_place(label: str, place: str, frames, mdl, proc):
    """True if the sound belongs here, False if it plainly does not, None if unsure."""
    verdict = _ab(mdl, proc,
                  "This place is " + place + ". A sound detector heard " + label
                  + " here. Judge from the frames and the kind of place.",
                  "a sound of " + label + " is plausible in this place",
                  "a sound of " + label + " is out of place here and the detector is "
                  "probably wrong",
                  frames=frames)
    if verdict is not False:
        return True if verdict is None else verdict
    # One a/b "out of place" is not enough to drop a real sound: it called a dog out of
    # place in a parking lot. A veto needs a second, independent signal -- the open
    # list of sounds expected here must ALSO leave it out. The horse at the quarry and
    # the ice-cream truck at the station fail both; a dog in a city passes the second.
    expect = _clean_phrase(_ask(mdl, proc,
                                "This place is " + place + ". List up to twelve sounds "
                                "you might hear here, comma separated.",
                                images=frames, max_new=64), max_words=60)
    return _about_the_sound(expect, label)


def _speech_near(segments, start: float, end: float) -> str:
    """Whatever was said within SPEECH_WINDOW seconds of the sound, in order."""
    lo, hi = start - SPEECH_WINDOW, end + SPEECH_WINDOW
    texts = [seg.text.strip() for seg in (segments or [])
             if seg.end >= lo and seg.start <= hi and seg.text.strip()]
    return " ".join(" ".join(texts).split())[:240]


def _talked_about(label: str, speech: str, mdl, proc, flip: bool = False) -> bool:
    """Asked in BOTH orderings; yes only if both say yes.

    Alternating the letter between calls is not enough. In the first run the model
    answered "b" to 22 of 23 questions whichever option (b) was, so the alternation
    turned a position bias into a coin flip, and the coin rescued an Alarm on the
    strength of "Ugh." and a Crowd on "Hm, hm, hm." Asking both orderings and requiring
    agreement is the standard repair: a bias towards one letter produces yes in one
    ordering and no in the other, and only an answer driven by the content survives. It
    costs one extra short call per sound that has speech near it.
    """
    votes = []
    for fl in (False, True):
        opt_a, opt_b = (SPEECH_NO, SPEECH_YES) if fl else (SPEECH_YES, SPEECH_NO)
        want = "b" if fl else "a"
        reply = _ask(mdl, proc, SPEECH_PROMPT.format(label=label, speech=speech,
                                                     opt_a=opt_a, opt_b=opt_b),
                     max_new=6).strip().lower().lstrip("(")
        votes.append(reply[:1] == want)
    verdict = all(votes)
    print("       [stage5] talked about? " + label + " <- \"" + speech[:60]
          + ("..." if len(speech) > 60 else "") + "\" -> "
          + ("yes" if verdict else "no") + " (votes "
          + "/".join("y" if v else "n" for v in votes) + ")", flush=True)
    return verdict


def decide_subjects(video_path, specs, transcript: str = "", segments=None,
                    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    device: str = "cuda", frames_per_sound: int = 4,
                    display_threshold: float = 0.12) -> None:
    """Fill in spec.subject for every augmented sound, and drop the visible ones.

    ``segments`` (timestamped Whisper output) feeds ONE question -- are people reacting
    to this sound? -- and nothing else. ``transcript`` is accepted for interface
    stability and ignored: fed to the depiction step, dialogue leaked straight into the
    pictures ("Hiccup on the phone, slow down, how many"). Speech is evidence for the
    gate, never material for the illustrator.
    """
    VOTE_LOG.clear()
    from src.stage2_video_understanding import _sample_frames, _sample_frames_at
    active = [s for s in specs if s.augment]
    # Marginal sounds the gate declined on confidence alone. Speech can rescue one of
    # these, but only from the upper half of the band below the threshold: a sound at a
    # tenth of the bar is noise whatever anyone says about it.
    marginal = [s for s in specs if not s.augment
                and s.reason.startswith("below display threshold")
                and s.confidence >= 0.5 * display_threshold]
    if not active and not (marginal and segments):
        return
    mdl, proc = _load(model, device)
    sim_device = device if device == "cuda" else "cpu"

    # 0. speech: is anyone reacting to this sound? (never touches the picture)
    if segments and getattr(config, "SPEECH_CONTEXT", True):
        for k, spec in enumerate(active + marginal):
            said = _speech_near(segments, spec.start, spec.end)
            if len(said.split()) < SPEECH_MIN_WORDS:
                continue
            # Never rescue what the whole-clip pass already saw on screen. The rescue
            # once let a Crowd through on "oh" in a clip whose scene sentence began "A
            # crowd of people with torches"; Stage 2 had marked it visible and the gate
            # had recorded only the confidence reason.
            if not spec.augment and "visible" in spec.reason:
                continue
            if _talked_about(spec.event_label, said, mdl, proc):
                spec.talked_about = True
                if not spec.augment:
                    spec.augment = True
                    spec.reason = ("marginal (" + format(spec.confidence, ".2f")
                                   + ") but people are reacting to it -> augment")
                    print("       [stage5] rescued by speech: " + spec.event_label,
                          flush=True)
                else:
                    spec.reason += " | people are reacting to it"
        active = [s for s in specs if s.augment]
        if not active:
            return

    frames = _sample_frames(Path(video_path), 4)
    scene = _ask(mdl, proc, SCENE_PROMPT, images=frames, max_new=40) if frames else ""
    scene = " ".join(scene.split())[:160] or "an unknown place"
    print("       [stage5] scene: " + scene, flush=True)
    place = _clean_phrase(_ask(mdl, proc, PLACE_PROMPT, images=frames, max_new=16),
                          max_words=4) if frames else ""
    place = place or scene
    print("       [stage5] place: " + place, flush=True)

    spec_frames = {}
    # 1. visibility, per sound and per STRETCH of it. A siren that runs the whole clip
    # is off screen while the police car approaches and on screen once it arrives; one
    # verdict over frames spread across nineteen seconds was a coin toss between runs.
    # Each burst longer than STRETCH is cut into stretches, each stretch is judged on
    # its own frames, and only the stretches where the source is not visible keep the
    # picture. Authoritative: a sound people are talking about is still not shown
    # where its source is on screen.
    if getattr(config, "VLM_VISIBILITY", True):
        STRETCH = float(getattr(config, "VISIBILITY_STRETCH", 5.0))
        for spec in active:
            bursts = list(getattr(spec, "spans", None) or [(spec.start, spec.end)])
            pieces = []
            for a, b in bursts:
                k = max(1, int(round((b - a) / STRETCH)))
                edges = [a + (b - a) * i / k for i in range(k + 1)]
                pieces += list(zip(edges, edges[1:]))
            kept, named_any = [], ""
            for a, b in pieces:
                # a second before to a second after each stretch, six frames: a short
                # sound is not one frame, and the cause is often legible from what
                # changed. Adam: "make sure he sees enough frames."
                n = max(frames_per_sound, 6)
                lo, hi = a - 1.0, b + 1.0
                times = [lo + (hi - lo) * t / (n - 1) for t in range(n)]
                win = _sample_frames_at(Path(video_path), times)
                if id(spec) not in spec_frames:
                    spec_frames[id(spec)] = win
                seen, named = _sound_is_visible(spec.event_label, win, mdl, proc, sim_device)
                # raw votes per sound per stretch, dumped by the pipeline to gate_votes.json so
                # that the silence rule (majority / unanimous) can be re-decided on CPU later
                VOTE_LOG.append({"label": spec.event_label, "start": spec.start, "end": spec.end,
                                 "confidence": spec.confidence, "stretch": [a, b], "seen": bool(seen),
                                 **{k: v for k, v in LAST_VOTES.items()}})
                if seen:
                    named_any = named
                else:
                    kept.append((a, b))
            if not kept:
                spec.augment = False
                spec.subject = ""
                spec.image_prompt = ""
                spec.reason = "source visible on screen (" + named_any + ") - stay silent"
                print("       [stage5] silent: " + spec.event_label + " is visible ("
                      + named_any + ")", flush=True)
            elif len(kept) < len(pieces):
                # Adam: one continuous sound, one continuous picture. Cutting the
                # picture for the stretches where the source is in frame produced a
                # siren that appeared, vanished and reappeared for a sound that never
                # stopped -- the flicker the panel exists to avoid, and a false claim
                # that the sound had stopped. The stretch verdicts decide only whether
                # the source was visible THROUGHOUT; if not, the picture stays for the
                # whole burst.
                print("       [stage5] " + spec.event_label + ": source visible for "
                      + str(len(pieces) - len(kept)) + " of " + str(len(pieces))
                      + " stretch(es); shown throughout", flush=True)
        # A kind of a visible source is the same source. Laughter was silenced because
        # the woman laughing is on screen, and Giggle -- her giggle -- was then shown,
        # because dedup only compares sounds that are still live.
        from src.labels import same_source
        gone = [s for s in specs if not s.augment and "visible" in s.reason
                and not s.reason.startswith("below display threshold")]   # a marginal sound never silences others
        for spec in [s for s in specs if s.augment]:
            for g in gone:
                # Amendment 3 (2026-09-21, bug fix): the kinship test alone was time-blind --
                # a Vehicle seen at 0-4 s silenced a horn at 20 s whose own check said "not
                # visible". Same source means same family AND the same moment (_dedup's test).
                # Direction (Adam, 28 Sept 2026; bug found in mc_bridge_scene): a visible label may silence only
                # the same label or a MORE GENERAL one (a visible "Chink, clink" silences "Glass"). A visible general
                # label never silences a specific one whose own check said "not visible" ("cars" seen -> Vehicle
                # visible must not silence Helicopter). Off with KINSHIP_DIRECTED = False (the scored runs).
                from src.labels import is_descendant as _desc
                directed = getattr(config, "KINSHIP_DIRECTED", False)
                if directed and not (g.event_label == spec.event_label or _desc(g.event_label, spec.event_label)):
                    continue
                if same_source(spec.event_label, g.event_label) and _overlap(spec, g):
                    spec.augment = False
                    spec.subject = ""
                    spec.image_prompt = ""
                    spec.reason = ("a kind of " + g.event_label
                                   + ", whose source is visible - stay silent")
                    print("       [stage5] silent: " + spec.event_label + " is a kind of "
                          + g.event_label + ", which is visible", flush=True)
                    break
        active = [s for s in specs if s.augment]
        if not active:
            print("       [stage5] every sound was already visible; nothing to add",
                  flush=True)
            return

    # 1a'. corroboration: a sound in the band just above the bar needs a second signal
    band = float(getattr(config, "CORROBORATE_BELOW", 0.0))
    if band > display_threshold:
        for spec in [s for s in specs if s.augment]:
            if spec.confidence >= band or spec.talked_about:
                continue
            backed = bool(spec.detail and spec.detail != spec.event_label)
            if not backed:
                kind = _kind_from_frames(spec.event_label, spec_frames.get(id(spec)),
                                         mdl, proc, seen_only=True)
                if kind:
                    spec.detail = kind
                    backed = True
            if not backed:
                spec.augment = False
                spec.subject = ""
                spec.image_prompt = ""
                spec.reason = ("faint (" + format(spec.confidence, ".2f")
                               + ") and nothing backs it - dropped")
                print("       [stage5] dropped " + spec.event_label + " ("
                      + format(spec.confidence, ".2f") + "): nothing corroborates it",
                      flush=True)
        active = [s for s in specs if s.augment]
        if not active:
            return

    # 1a. does the sound fit the place? (bounded; see _fits_the_place)
    if getattr(config, "PLAUSIBILITY_CHECK", True):
        for spec in [s for s in specs if s.augment]:
            if spec.talked_about or spec.confidence >= PLAUSIBLE_MAX:
                continue
            fits = _fits_the_place(spec.event_label, place, spec_frames.get(id(spec)),
                                   mdl, proc)
            print("       [stage5] fits the place? " + spec.event_label + " at "
                  + place + " -> " + ("yes" if fits else "no"), flush=True)
            if fits is False:
                spec.augment = False
                spec.subject = ""
                spec.image_prompt = ""
                spec.reason = ("out of place at " + place + " (conf "
                               + format(spec.confidence, ".2f") + ") - dropped")
        active = [s for s in specs if s.augment]
        if not active:
            return

    # 1b. one acoustic event with two names: let the frames choose between them
    if getattr(config, "DISAMBIGUATE", True):
        _disambiguate(specs, video_path, mdl, proc, frames_per_sound)
        active = [s for s in specs if s.augment]
        if not active:
            return

    # 2. depiction: the sound as an EVENT, the place as a modifier.
    labels = [s.event_label for s in active]
    v2 = bool(getattr(config, "DEPICT_V2", False))
    v3 = bool(getattr(config, "PICTURE_V3", False))
    for spec in active:
        if v3:
            if getattr(config, "PICTURE_SCENE", False):
                # V3.1: the scene may qualify the heard word; the raw firings are not on the spec, so
                # the guard runs with none allowed beyond the source and its ancestors (the strict side)
                phrase = _depict_v31(spec, place, spec_frames.get(id(spec)), [], mdl, proc) or ""
            else:
                phrase = ""
            phrase = phrase or _depict_v3(spec, place, mdl, proc) or (
                (getattr(spec, "source", "") or spec.event_label).split(",")[0].split("(")[0].strip()
                + " making its sound")
            if getattr(config, "PICTURE_MAKER", False):
                phrase = with_maker(spec, phrase, spec_frames.get(id(spec)), mdl, proc)
            spec.subject = phrase
            spec.reason += " | depiction (v3, source " + (getattr(spec, "source", "") or "-") + "): " + phrase
            spec.image_prompt = spec.subject
            print("       [stage5] " + spec.event_label + " [" + (getattr(spec, "source", "") or "-")
                  + "] -> " + spec.subject, flush=True)
            continue
        detail, kind = "", ""
        if spec.detail and spec.detail != spec.event_label:
            detail = " (specifically: " + spec.detail.split(",")[0].split("(")[0].strip() + ")"
            if getattr(config, "KIND_ALWAYS", False):
                # The detector's sub-label is usually generic ("Rail transport", "Thunderstorm")
                # and it used to switch the frames off entirely: a subway clip got a steam
                # locomotive. The frames now always get a say about which kind it is.
                kind = _kind_from_frames(spec.event_label, spec_frames.get(id(spec)), mdl, proc)
                if kind:
                    detail = detail[:-1] + "; the frames suggest: " + kind + ")"
        else:
            # no sub-label from the detector: let the frames say what kind, if they can
            kind = _kind_from_frames(spec.event_label, spec_frames.get(id(spec)), mdl, proc)
            if kind:
                detail = " (specifically: " + kind + ")"
        prompt = (DEPICT_PROMPT_V2 if v2 else DEPICT_PROMPT).format(
            label=spec.event_label, detail=detail, scene=place)
        phrase = _clean_phrase(_ask(mdl, proc, prompt, max_new=48))
        phrase = _without_place(phrase, place, keep=kind if v2 else "")
        if v2:
            phrase = _drop_place_phrase(phrase, place, spec.event_label, spec.detail or "")
        if phrase and not _still_the_sound(phrase, spec.event_label, labels, mdl, proc):
            print("       [stage5] rejected (not about " + spec.event_label + "): "
                  + phrase, flush=True)
            first = phrase
            phrase = _without_place(_clean_phrase(_ask(mdl, proc, RETRY_PROMPT.format(
                label=spec.event_label, detail=detail, scene=place), max_new=48)), place)
            same_again = phrase.lower().split() == first.lower().split()
            if same_again and not _still_the_sound(phrase, spec.event_label, labels,
                                                   mdl, proc):
                # Rejected twice AND the model had nothing else to say. A phantom "Sigh"
                # at a shooting range came back "gun recoil" both times; accepting the
                # retry put a gun beside the video for a sound that was never there.
                # The condition is deliberately narrow. Dropping on two rejections
                # alone also removed "people clapping hands" for Crowd and "gun fire"
                # for Explosion, both real -- the forced choice, asked twice, rejects
                # too many good depictions to be a filter on its own. An identical
                # retry is the signal that the model is describing the scene because
                # there is no sound to describe.
                print("       [stage5] dropped " + spec.event_label
                      + ": no depiction reads as it (" + phrase + ", twice)", flush=True)
                spec.augment = False
                spec.subject = ""
                spec.image_prompt = ""
                spec.reason = "no depiction reads as this sound - dropped"
                continue
        if not phrase:
            # Only if the model returned nothing at all. Still an event, never a noun.
            phrase = (spec.detail.split(",")[0] if spec.detail else spec.event_label) + " happening"
        spec.subject = phrase
        spec.reason += " | depiction: " + phrase
        spec.image_prompt = spec.subject
        print("       [stage5] " + spec.event_label + " -> " + spec.subject, flush=True)

    # 3. one picture per source
    active = [s for s in specs if s.augment]
    if active:
        _dedup(active, mdl, proc, sim_device)
