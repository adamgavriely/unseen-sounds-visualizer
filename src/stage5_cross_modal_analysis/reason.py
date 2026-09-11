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
    "Use the place only if it changes what the action looks like. Never describe the "
    "place instead of the action."
    + chr(10) +
    "Example: sound Glass, place a kitchen, gives: a drinking glass shattering on a tiled "
    "floor."
    + chr(10) +
    "Example: sound Crowd, place a talk show studio, gives: a studio audience applauding."
    + chr(10) +
    "Answer with a short phrase of at most 8 words. No punctuation, no explanation."
)

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
    "Answer with a short phrase of at most 8 words. No punctuation, no explanation."
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
    return any(any(g.startswith(w[:4]) for g in got) for w in want)


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
    options = [label] + [o for o in others if o != label][:4]
    # The right answer must not always be (a), or the question stops being a choice and
    # becomes the agreement task it was meant to replace. Its position is rotated by a
    # hash of the sound's own name: spread across sounds, stable for one sound, and
    # reproducible from run to run without a random seed.
    slot = sum(ord(c) for c in label) % len(options)
    options[0], options[slot] = options[slot], options[0]
    letters = "abcdef"
    body = chr(10).join("(" + letters[i] + ") " + o for i, o in enumerate(options))
    body += chr(10) + "(" + letters[len(options)] + ") none of them"
    reply = _ask(mdl, proc, CHOOSE_PROMPT.format(phrase=phrase, options=body),
                 max_new=6).strip().lower().lstrip("(")
    want = letters[slot]
    print("       [stage5] reads as? '" + phrase + "' -> " + reply.strip()
          + " (want " + want + " = " + label + ")", flush=True)
    return reply[:1] == want


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
            # the other, they are one source and no threshold has to be chosen.
            if same_source(spec.event_label, order[j].event_label):
                dup, score, why = j, 1.0, "same source in the AudioSet ontology"
                break
            if subs[i] == subs[j]:
                dup, score, why = j, 1.0, "identical depiction"
                break
            if mat is None:
                continue
            sim = mat[i][j]
            if sim >= sure:
                dup, score, why = j, sim, "similarity " + format(sim, ".2f")
                break
            if sim >= report:
                print("       [stage5] near-duplicate? " + spec.event_label + " / "
                      + order[j].event_label + " sim=" + format(sim, ".2f")
                      + " (bar " + format(sure, ".2f") + ")", flush=True)
        if dup is None:
            kept.append(i)
            continue
        spec.augment = False
        spec.subject = ""
        spec.image_prompt = ""
        spec.reason = ("same picture as " + order[dup].event_label + " (" + why + ")")
        print("       [stage5] merged " + spec.event_label + " into "
              + order[dup].event_label + " (" + why + ")", flush=True)


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
            if not said:
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

    # 1. visibility, per sound, on the frames spanning that sound. Authoritative: a
    # sound people are talking about is still not shown if its source is on screen.
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

    # 2. depiction: the sound as an EVENT, the place as a modifier.
    labels = [s.event_label for s in active]
    for spec in active:
        detail = ""
        if spec.detail and spec.detail != spec.event_label:
            detail = " (specifically: " + spec.detail.split(",")[0].split("(")[0].strip() + ")"
        prompt = DEPICT_PROMPT.format(label=spec.event_label, detail=detail, scene=place)
        phrase = _clean_phrase(_ask(mdl, proc, prompt, max_new=48))
        if phrase and not _still_the_sound(phrase, spec.event_label, labels, mdl, proc):
            print("       [stage5] rejected (not about " + spec.event_label + "): "
                  + phrase, flush=True)
            phrase = _clean_phrase(_ask(mdl, proc, RETRY_PROMPT.format(
                label=spec.event_label, detail=detail, scene=place), max_new=48))
        if not phrase:
            # Only if the model returned nothing at all. Still an event, never a noun.
            phrase = (spec.detail.split(",")[0] if spec.detail else spec.event_label) + " happening"
        spec.subject = phrase
        spec.reason += " | depiction: " + phrase
        spec.image_prompt = spec.subject
        print("       [stage5] " + spec.event_label + " -> " + spec.subject, flush=True)

    # 3. one picture per source
    _dedup(active, mdl, proc, sim_device)
