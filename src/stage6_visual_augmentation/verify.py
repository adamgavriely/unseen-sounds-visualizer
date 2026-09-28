"""PICTURE_VERIFY (2026-09-28): look at a drawn picture before it is shown, and redraw it if it shows the wrong thing.

About 6 of the 82 shipped inspector pictures tell the viewer something false: "crowd" came out as a flock of crows,
"alarm bell" as a desk bell, a car horn as a car with a megaphone, "steam" as a kettle, and one fire-alarm box
carries garbled letters. A deaf viewer cannot tell a wrong picture from a right one, so the system has to.

Two checks, both on the finished picture:

  * a SHUFFLED MULTIPLE-CHOICE question to the pipeline's VLM (config.VLM_MODEL, Qwen3.8-27B under use_shipped):
    "What is the main thing in this picture?" The options are the intended thing, its likely confusions and
    "something else". The picture passes only if the model picks the intended option. A free question ("is this a
    crowd?") is answered yes far too easily; a choice between the intended thing and its known look-alikes is not.
      - intended thing and confusions for known ambiguous words come from a small fixed table (AMBIGUOUS), matched
        on the SOUND (label / source) first and the subject second -- the V3.1 subject itself can carry the error
        ("Steam rising from a kettle"), so the subject cannot be the judge of itself;
      - otherwise the intended thing is the subject that was drawn, and the confusions are generic ones (a person,
        an animal, a vehicle, a building or landscape, text or a sign) minus any category the sound belongs to
        ("a person" is never offered against laughter).
  * a TEXT check: EasyOCR; a picture with a readable word (>= 3 letters, confidence >= 0.5, letters at least 2% of
    the picture tall) is refused. Digits alone do not count (a locomotive's running number is not a message).

The redraw loop lives in stage6._final_picture: up to 5 tries (PICTURE_VERIFY_TRIES), each with a new seed and the wrong thing the VLM saw added to the negative; from try 3 a clearer fixed
rewrite for ambiguous words and "no text" in the prompt; if every try fails, a word card with the sound's name.
"""
from __future__ import annotations

import random
import re
import zlib
from typing import Dict, List, Optional

import config

# ---------------------------------------------------------------------------------------------------------------
# Known ambiguous words. Each entry: when it applies (words in the sound's label/source, or phrases in the subject),
# the intended option, the look-alikes it was seen (or is likely) to be drawn as, and the clearer rewrite used from
# try 3 on (with extra negative-prompt words: a diffusion model draws what a prompt names, "no megaphone" included).
# Order matters: the first entry that applies wins.
AMBIGUOUS = [
    {"name": "car_alarm", "sound": {"car alarm"}, "subject": [],
     "intended": "a parked car with its lights flashing",
     "confusions": ["a police car with a light bar", "an alarm bell"],
     "rewrite": None, "neg": ""},
    {"name": "smoke_alarm", "sound": {"smoke detector", "smoke alarm"}, "subject": ["smoke detector", "smoke alarm"],
     "intended": "a smoke detector", "maker": "smoke detector", "noise": "alarm",
     # 28 Sept (Adam): + dome / CCTV camera -- round 3 passed a dome security camera wreathed in smoke
     "confusions": ["a security camera", "a ceiling lamp", "a dome security camera", "a CCTV camera"],
     "rewrite": "a round white smoke detector on a ceiling, sounding its alarm", "neg": "camera, lens",
     # 28 Sept (Adam): the rewrite from try 1 -- the checker passes a dome camera as a smoke detector, and only the
     # rewrite drew a real one (round 2, try 3)
     "rewrite_first": True},
    {"name": "alarm_bell", "sound": {"alarm", "fire alarm", "alarm bell"}, "subject": ["alarm bell", "fire alarm"],
     "intended": "a fire alarm (a bell or alarm box on a wall)", "maker": "fire alarm bell", "noise": "alarm",
     "confusions": ["a desk bell (service bell)", "a church bell", "a telephone"],
     "rewrite": "a red fire-alarm bell mounted on a wall, ringing", "neg": "desk bell, service bell, counter"},
    {"name": "horn", "sound": {"toot", "honk", "vehicle horn", "car horn", "honking", "air horn"},
     "subject": ["honk", "horn"],
     # AudioSet "Honk" is a goose (Goose > Fowl): a goose subject is not a horn (PICTURE_MAKER, 28 Sept)
     "unless_subject": ["goose", "geese"],
     "intended": "a {veh} sounding its horn", "maker": "{veh}", "noise": "horn",
     "confusions": ["a megaphone or loudspeaker", "a vehicle with a megaphone or loudspeaker on it",
                    "a trumpet or musical horn",
                    # 28 Sept: the sliceB Honk redraw passed with a trumpet-shaped horn stuck in a car's grille
                    "a vehicle with a large trumpet-shaped horn stuck on it"],
     "rewrite": "a {veh} on a street with its horn sounding, seen from the front",
     "neg": "megaphone, loudspeaker, bullhorn, trumpet, horn-shaped object, speaker cone"},
    {"name": "steam", "sound": {"steam"}, "subject": ["steam"],
     "intended": "steam hissing out of a pipe or valve", "maker": "jet of steam", "noise": "hissing",
     "confusions": ["a kettle", "smoke from a fire", "a cloud"],
     "rewrite": "white steam hissing out of a metal pipe valve", "neg": "kettle, teapot, cup, pot"},
    {"name": "crowd", "sound": {"crowd", "cheering", "hubbub, speech noise, speech babble"},
     "subject": ["crowd"],
     "intended": "a crowd of people", "maker": "crowd of people", "noise": "cheering",
     "confusions": ["a flock of birds", "a single person", "a building or landscape"],
     "rewrite": "a crowd of people cheering with raised arms", "neg": "birds, crows, ravens, animals"},
    {"name": "typing", "sound": {"typing", "computer keyboard", "typewriter"}, "subject": ["typing", "keyboard"],
     "intended": "hands typing on a keyboard", "maker": "computer keyboard", "noise": "typing",
     "confusions": ["a person's face", "a computer screen"],
     "rewrite": "two hands typing on a computer keyboard, seen from above", "neg": "face, head, mouth"},
    # 28 Sept (Adam): the sliceB rattle passed on try 3 as a ball of yarn. The instrument only: AudioSet's plain
    # "Rattle" is a rattling noise (a loose part, a vehicle), not a thing to draw as a toy.
    {"name": "rattle", "sound": {"rattle (instrument)", "maraca", "maracas"},
     "subject": ["rattle instrument", "baby rattle", "maraca"],
     "intended": "a rattle or maraca being shaken", "maker": "maraca", "noise": "rattle",
     "confusions": ["a ball of yarn", "a spinning top", "a toy ball"],
     "rewrite": "a hand shaking a wooden maraca, a rattle instrument with a handle",
     "neg": "yarn, wool, thread, spinning top, ball, swirl"},
    {"name": "bell", "sound": {"bell", "ding", "chime", "church bell", "jingle bell"}, "subject": ["bell"],
     "intended": "a bell",
     "confusions": ["a desk bell (service bell)", "a lamp"],
     "rewrite": None, "neg": ""},
]
# the vehicle a horn belongs to, read from the subject / label
_VEHICLES = ["bus", "truck", "train", "locomotive", "car", "motorcycle", "ship", "boat"]

# generic confusions and the words that put a sound in their category (then that option is not offered)
GENERIC = [
    ("a person", {"person", "people", "man", "woman", "men", "women", "child", "children", "baby", "infant",
                  "crowd", "audience", "hands", "hand", "fingers", "face", "laughter", "laughing", "laugh",
                  "giggle", "gasp", "gasping", "cry", "crying", "applause", "clapping", "cheering", "shout",
                  "shouting", "scream", "screaming", "yell", "run", "running", "footsteps", "walk", "walking",
                  "shaver", "razor", "speech", "singing", "cough", "coughing", "sneeze", "breathing", "whistling",
                  "someone", "runner", "snoring", "sigh", "chewing", "burping", "kiss"}),
    ("an animal", {"bird", "birds", "dog", "cat", "cricket", "insect", "pigeon", "dove", "rooster", "fowl",
                   "chicken", "hen", "crow", "owl", "mosquito", "bee", "fly", "horse", "cow", "sheep", "goat",
                   "frog", "animal", "chirp", "chirping", "tweet", "bark", "barking", "crowing", "cooing", "wolf",
                   "howl", "duck", "goose", "pig", "insects", "cicada", "gull", "seagull", "parrot", "bow-wow",
                   "purr", "meow", "moo", "neigh", "buzzing", "roar", "lion"}),
    ("a vehicle", {"car", "cars", "bus", "truck", "train", "railroad", "locomotive", "wagon", "vehicle", "ambulance",
                   "engine", "van", "motorcycle", "tank", "helicopter", "aircraft", "airplane", "plane", "boat",
                   "ship", "tram", "subway", "taxi", "police", "tractor", "jet", "motorboat", "bicycle", "scooter"}),
    ("a building or landscape", {"thunder", "thunderstorm", "lightning", "storm", "rain", "raining", "cloud",
                                 "clouds", "wind", "waterfall", "water", "river", "sea", "ocean", "waves", "stream",
                                 "explosion", "fireworks", "firecracker", "church", "tower", "window", "glass",
                                 "house", "door", "building", "fire", "avalanche", "earthquake", "artillery",
                                 "boom", "burst", "cupboard", "drip", "dripping", "faucet", "trickle"}),
    ("text or a sign", {"sign", "text", "letters", "writing"}),
]
SOMETHING_ELSE = "something else"

MC_PROMPT = ("Look at this picture. What is the main thing shown in it? Choose exactly one option.\n{options}\n"
             "Answer with the letter only.")
SEE_PROMPT = "In at most eight words, what is the main thing shown in this picture?"
TEXT_PROMPT = ("Does this picture contain any written letters or words (ignore digits)? Answer yes or no.")

# the OCR rule, fixed before the validation run (never tuned on it)
OCR_MIN_LETTERS = 3
OCR_MIN_CONF = 0.5
OCR_MIN_HEIGHT = 0.02


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z\-]+", (s or "").lower()) if w}


_FILLER = {"a", "an", "the", "or", "and", "of", "on", "in", "with", "it", "its", "large", "small", "stuck", "something",
           "else", "shaped", "trumpet-shaped", "picture", "some", "person", "people", "thing"}


def feedback_negative(picked: str, intended: str, subject: str, spec) -> str:
    """Refinement for the next try (Adam, 28 Sept): the wrong thing the VLM saw goes into the negative prompt. Words that
    name the source itself (subject, intended option, the sound's names) are never negated."""
    if not picked or picked == SOMETHING_ELSE or picked == intended:
        return ""
    keep = _words(" ".join([subject or "", intended or ""] + _sound_names(spec)))
    # the source's own category word is never negated either: "a vehicle with a megaphone on it" for a car horn must
    # not put "vehicle" in the negative (28 Sept fix)
    for opt, words in GENERIC:
        if keep & words:
            keep |= _words(opt)
    return ", ".join(w for w in sorted(_words(picked)) if w not in keep and w not in _FILLER and len(w) > 2)


def _sound_names(spec) -> List[str]:
    """The sound's own names, lowercase: label, source and each comma part of them."""
    out = []
    for s in (getattr(spec, "source", "") or "", spec.event_label or ""):
        s = s.lower().strip()
        if s:
            out.append(s)
            out += [p.split("(")[0].strip() for p in s.split(",") if p.strip()]
    return out


def _vehicle(spec, subject: str) -> str:
    text = " ".join([subject or "", getattr(spec, "source", "") or "", spec.event_label or ""]).lower()
    for v in _VEHICLES:
        if re.search(r"\b" + v + r"\b", text):
            return "train" if v == "locomotive" else v
    return "car"


def ambiguous_entry(spec, subject: str) -> Optional[dict]:
    """The AMBIGUOUS entry that applies to this sound, with {veh} filled in; None for an ordinary sound."""
    names = set(_sound_names(spec))
    subj = (subject or "").lower()
    for e in AMBIGUOUS:
        if names & e["sound"] or any(re.search(r"\b" + re.escape(p), subj) for p in e["subject"]):
            if getattr(config, "PICTURE_MAKER", False) and any(re.search(r"\b" + re.escape(p), subj)
                                                               for p in e.get("unless_subject", [])):
                continue
            veh = _vehicle(spec, subject)
            f = lambda x: x.format(veh=veh) if isinstance(x, str) else x
            return {**e, "intended": f(e["intended"]), "rewrite": f(e["rewrite"]), "maker": f(e.get("maker")),
                    "confusions": [f(c) for c in e["confusions"]]}
    return None


def _phrase(subject: str) -> str:
    s = " ".join((subject or "").split()).strip(" .")
    return s[0].lower() + s[1:] if s else s


def options_for(spec, subject: str, max_conf: Optional[int] = None) -> Dict:
    """intended option + confusions (table first, then generics not in the sound's category)."""
    entry = ambiguous_entry(spec, subject)
    intended = entry["intended"] if entry else _phrase(subject or spec.source or spec.event_label)
    confusions = list(entry["confusions"]) if entry else []
    if entry and entry.get("maker") and getattr(config, "PICTURE_LOOKALIKE_VLM", False):
        confusions = lookalikes(entry["maker"], entry["intended"]) or confusions      # the hand-written ones if the answer is unusable
    cat_words = _words(" ".join([subject or "", intended, getattr(spec, "source", "") or "", spec.event_label]))
    for opt, words in GENERIC:
        if max_conf is not None and len(confusions) >= max_conf:
            break
        if cat_words & words:
            continue
        if any(opt.split()[-1] in c for c in confusions):      # "a building or landscape" already in the table's
            continue
        confusions.append(opt)
    # 28 Sept (round 2): every allowed generic is offered, not the first three -- with the cap, "a building or
    # landscape" was dropped for a siren and a drawn house passed as the siren (validation w18)
    return {"intended": intended, "confusions": confusions if max_conf is None else confusions[:max_conf],
            "entry": entry["name"] if entry else None}


def _order_seed(spec, salt: int, tag: str = "") -> int:
    return zlib.crc32(f"{tag}|{spec.event_label}|{float(spec.start):.2f}|{salt}".encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------------------------------------------
_OCR = None


def _vlm():
    """The pipeline's VLM, loaded next to the generator. Shares reason._VLM (so stage 5 reuses it) but never goes
    through reason._load, which frees the generator first. With two cards, the VLM takes the last one."""
    from src.stage5_cross_modal_analysis import reason as R
    if R._VLM is None:
        import torch
        from transformers import AutoProcessor
        model = config.VLM_MODEL
        proc = AutoProcessor.from_pretrained(model)
        n = torch.cuda.device_count()
        dmap = {"": n - 1} if n > 1 else ("auto" if n == 1 else None)
        dtype = torch.bfloat16 if n else torch.float32
        if "Qwen2.5-VL" in model:
            from transformers import Qwen2_5_VLForConditionalGeneration
            mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, torch_dtype=dtype, device_map=dmap).eval()
        else:
            from transformers import AutoModelForImageTextToText
            mdl = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype, device_map=dmap).eval()
        R._VLM = (mdl, proc)
    return R._VLM


def _ocr():
    global _OCR
    if _OCR is None:
        import easyocr
        import torch
        _OCR = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _OCR


def text_found(img) -> Dict:
    """EasyOCR on the picture: the readable words (>= 3 letters, conf >= 0.5, >= 2% of the picture tall)."""
    import numpy as np
    arr = np.asarray(img.convert("RGB"))
    h = arr.shape[0]
    try:
        res = _ocr().readtext(arr)
    except Exception as e:                                  # an OCR failure must not block the picture
        return {"ok": True, "words": [], "all": [], "error": f"{type(e).__name__}: {e}"}
    all_, words = [], []
    for box, txt, conf in res:
        ys = [p[1] for p in box]
        tall = (max(ys) - min(ys)) / float(h)
        letters = sum(c.isalpha() for c in txt)
        all_.append([txt, round(float(conf), 2), round(tall, 3)])
        if letters >= OCR_MIN_LETTERS and conf >= OCR_MIN_CONF and tall >= OCR_MIN_HEIGHT:
            words.append(txt)
    return {"ok": not words, "words": words, "all": all_}


def ask_mc(img, spec, subject: str, salt: int = 0, tag: str = "") -> Dict:
    from src.stage5_cross_modal_analysis import reason as R
    mdl, proc = _vlm()
    o = options_for(spec, subject)
    opts = [o["intended"]] + o["confusions"]
    random.Random(_order_seed(spec, salt, tag)).shuffle(opts)
    opts.append(SOMETHING_ELSE)
    letters = "ABCDEFGHIJKL"[: len(opts)]
    assert len(letters) == len(opts), "more options than letters"
    q = MC_PROMPT.format(options="\n".join(f"{l}. {t}" for l, t in zip(letters, opts)))
    ans = R._ask(mdl, proc, q, images=[img], max_new=8)
    m = re.search(r"\b([A-L])\b", ans.upper()) or re.search(r"([A-L])", ans.upper())
    pick = opts[letters.index(m.group(1))] if m and m.group(1) in letters else None
    return {"options": opts, "answer": ans, "picked": pick, "intended": o["intended"], "entry": o["entry"],
            "ok": pick == o["intended"]}


def what_it_sees(img) -> str:
    from src.stage5_cross_modal_analysis import reason as R
    mdl, proc = _vlm()
    return " ".join(R._ask(mdl, proc, SEE_PROMPT, images=[img], max_new=24).split())


def vlm_text(img) -> str:
    from src.stage5_cross_modal_analysis import reason as R
    mdl, proc = _vlm()
    return R._ask(mdl, proc, TEXT_PROMPT, images=[img], max_new=4).strip().lower()


def check(path, spec, subject: str, salt: int = 0, tag: str = "", see: bool = True) -> Dict:
    """The full check on one picture: the multiple-choice question, the text check and (for the log) a short
    free description. ok = both checks pass."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    small = img.resize((768, 768)) if img.size[0] > 768 else img
    mc = ask_mc(small, spec, subject, salt, tag)
    txt = text_found(img)
    out = {"ok": bool(mc["ok"] and txt["ok"]), "mc": mc, "text": txt}
    if see:
        out["saw"] = what_it_sees(small)
    return out


def rewrite_first(spec, subject: str) -> bool:
    """True when the entry says its rewrite is used from try 1 (a look-alike the checker cannot catch)."""
    e = ambiguous_entry(spec, subject)
    return bool(e and e.get("rewrite_first") and e.get("rewrite"))


def rewrite_for(spec, subject: str) -> Optional[dict]:
    """The clearer subject for an ambiguous word (try 3 on), with its extra negative words. PICTURE_LOOK_VLM: written
    by the VLM from the entry's maker object (describe); None when its answer fails the guards (plain subject)."""
    e = ambiguous_entry(spec, subject)
    if e and e.get("rewrite") and getattr(config, "PICTURE_LOOK_VLM", False):
        d = describe(e["maker"], e["noise"], spec) if e.get("maker") else ""
        return {"subject": d, "neg": ""} if d else None
    if e and e.get("rewrite"):
        return {"subject": e["rewrite"], "neg": e.get("neg", "")}
    return None


# ---------------------------------------------------------------------------------------------------------------
# PICTURE_LOOK_VLM / PICTURE_LOOKALIKE_VLM (Adam, 28 Sept 2026: the hand-written rewrite wording is "too specific").
# The pipeline VLM, text only, greedy, one answer per (maker, sound) cached for the run.
DESCRIBE_PROMPT = ("In one short sentence for an image generator, describe what a typical {maker} looks like while it "
                   "makes its {noise} sound. Name only that object and its visible parts; no other objects, {people}"
                   "no place, no text. Begin the sentence with: A {maker}")
LOOKALIKE_PROMPT = ("A picture is meant to show {intended}. What could such a picture be mistaken for? List 3 common "
                    "objects, comma separated, each with 'a' or 'an'. Answer with the list only.")
_DESC: Dict = {}
_LOOK: Dict = {}
DESCRIBE_LOG: list = []


def describe(maker: str, noise: str, spec) -> str:
    """The VLM's one-sentence look of the maker object while it makes the sound, or "" when a guard refuses it:
    it must name the maker noun (list), and pass the same word guards as V3.1 -- reason.expand_guard (no other sound
    source, person or place than the maker and the sound's own names), labels.names_forbidden and, for the words
    before the maker noun, labels.other_branch_makers."""
    from src.stage5_cross_modal_analysis import reason as R
    from src.labels import names_forbidden, other_branch_makers, ancestors
    src = getattr(spec, "source", "") or spec.event_label
    key = (maker, noise, src)
    if key in _DESC:
        return _DESC[key]
    human = "Human sounds" in ancestors(src) or src == "Human sounds"
    mdl, proc = _vlm()
    prompt = DESCRIBE_PROMPT.format(maker=maker, noise=noise,
                                    people="" if human else "no people or hands, ")
    raw = R._ask(mdl, proc, prompt, max_new=80)
    text = " ".join("".join(c if ord(c) < 0x250 else " " for c in raw).split()).strip().strip('"').rstrip(".")
    noun = maker.split()[-1].lower()
    words = _words(text)
    why = []
    if not text or len(text.split()) < 4:
        why.append("empty")
    if not ({noun, noun + "s", noun + "es"} & words):
        why.append("no maker noun " + noun)
    own = _words(maker + " " + noise)            # the maker was chosen (entry / MAKERS): its own words never refused
    bad = [w for w in R.expand_guard(text, maker + " " + noise, src, spec.event_label)
           if w not in own and w.rstrip("s") not in own and w[:-2] not in own]         # plural of an allowed word
    if bad:
        why.append("guard " + ",".join(bad))
    nf = [n for n in names_forbidden(text, src, []) if not _words(n) <= own]
    if nf:
        why.append("forbidden " + ",".join(nf))
    alien = other_branch_makers([w for w in re.split(r"[^a-z]+", text.lower()) if w], src)
    if alien:
        why.append("other-branch " + ",".join(alien))
    out = "" if why else text
    _DESC[key] = out
    DESCRIBE_LOG.append({"maker": maker, "noise": noise, "source": src, "answer": text, "refused": why, "used": out})
    print(f"       [verify] describe {maker!r} ({noise}): {text!r} -> {'OK' if not why else 'REFUSED ' + '; '.join(why)}",
          flush=True)
    return out


def lookalikes(maker: str, intended: str) -> List[str]:
    """Three look-alikes of the intended picture written by the VLM (text only), for the checker's options; [] if
    unusable. An answer that is the maker itself (all its words) is dropped."""
    from src.stage5_cross_modal_analysis import reason as R
    if intended in _LOOK:
        return _LOOK[intended]
    mdl, proc = _vlm()
    raw = R._ask(mdl, proc, LOOKALIKE_PROMPT.format(intended=intended), max_new=48)
    raw = " ".join(raw.replace(chr(10), ", ").split())
    out = []
    mw = _words(maker) - _FILLER
    for part in raw.split(","):
        t = part.strip().strip(".").strip()
        if not t or mw <= _words(t):
            continue
        if not re.match(r"^(a|an)\s", t, re.I):
            t = ("an " if t[:1].lower() in "aeiou" else "a ") + t
        out.append(t[0].lower() + t[1:])
    out = out[:3] if len(out) >= 2 else []
    _LOOK[intended] = out
    print(f"       [verify] look-alikes of {intended!r}: {raw!r} -> {out}", flush=True)
    return out


def card_word(spec) -> str:
    """The sound's name for the word card: the drawn source's first name, else the label's."""
    s = (getattr(spec, "source", "") or spec.event_label or "SOUND")
    return s.split(",")[0].split("(")[0].strip().upper()
