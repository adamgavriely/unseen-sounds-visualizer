"""PICTURE_SENSE (28 Sept 2026): a GENERIC replacement for the hand-written per-word table (verify.AMBIGUOUS).

Two parts, both per label, both cached per (label, picture model):

  (a) SLOT FORM. The pipeline VLM (config.VLM_MODEL), text only, greedy, gets the label, its AudioSet ontology path and
      the label's OFFICIAL AudioSet description (src/audioset_ontology.json, the unchanged ontology.json of
      github.com/audioset/ontology) and fills four fixed slots: OBJECT / WHERE / ACTION / CUE. The drawing sentence is
      built from them with one fixed template. Guards (list checks, no model): the object's head noun is a word of the
      label's names, its ancestors' names or its description; no other sound source, person or place (reason.expand_guard,
      labels.names_forbidden, labels.other_branch_makers); no text words; no people unless a human sound. Any failure
      -> "" (the caller draws the plain subject).
  (b) MISTAKE MINING. 4 pictures of the plain subject (fixed seeds), the VLM asked openly "What is the main thing in
      this picture?"; each answer judged by the VLM (text only, one fixed closed question) against the target; the
      wrong answers become negative-prompt words and checker look-alike options for that label.

Hooks: stage6._final_picture (PICTURE_SENSE: plan() before the tries, check() instead of verify.check, no AMBIGUOUS
rewrite). Test: scripts/picture_sense_test.py (release v1.2.0), plan docs/history/analyses/picture_sense_test_2026-09-28.md (release v1.2.0).
"""
from __future__ import annotations

import json
import random
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import config

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "work" / "sense_cache"
ONTOLOGY = _ROOT / "src" / "audioset_ontology.json"

SLOT_PROMPT = (
    "Sound class: {name}" + chr(10)
    + "Where it sits in the AudioSet ontology: {path}" + chr(10)
    + "Official description: {desc}" + chr(10) + chr(10)
    + "Fill four slots for ONE picture that shows this sound being made, for a deaf viewer." + chr(10)
    + "OBJECT: the one thing that makes this sound, 1 to 4 words, a noun phrase; use a word from the lines above if "
      "you can." + chr(10)
    + "WHERE: where that object is, 2 to 5 words, close up (for example: on a ceiling, in the water); no landscape, "
      "room or street." + chr(10)
    + "ACTION: what the object is doing while it makes this sound, 2 to 6 words." + chr(10)
    + "CUE: one visible sign of this sound on or right next to the object, 2 to 6 words." + chr(10)
    + "No other objects, no text or letters{people}. If a word in the name could mean something else, use the meaning "
      "given by the ontology path and the description." + chr(10)
    + "Answer in exactly four lines:" + chr(10) + "OBJECT: ..." + chr(10) + "WHERE: ..." + chr(10) + "ACTION: ..."
    + chr(10) + "CUE: ...")
SLOTS = ("object", "where", "action", "cue")
MINE_PROMPT = "What is the main thing in this picture? Answer in a few words."
JUDGE_PROMPT = (
    "A picture was meant to show: {target}. (It stands for the sound '{name}': {desc})" + chr(10)
    + "A viewer says the main thing in the picture is: '{answer}'." + chr(10)
    + "Is that the intended thing, or a kind of it? Answer no if the viewer names a different main object, even a "
      "related one. Answer yes or no.")
MC_PROMPT = ("Look at this picture. What is the main thing shown in it? Choose exactly one option.\n{options}\n"
             "Answer with the letter only.")
LETTERS = "ABCDEFGHIJKLMNOPQRST"
N_MINE = 4
MAX_LOOK = 4
TEXT_WORDS = {"text", "letter", "letters", "word", "words", "sign", "signs", "label", "labels", "logo", "writing",
              "written", "caption", "banner", "number", "numbers"}
PEOPLE_X = {"hand", "hands", "finger", "fingers", "face", "player", "someone", "person", "people", "man", "woman"}
# negative-prompt stoplist (declared in the plan): colours, materials, picture words, empty nouns
NEG_STOP = {"red", "white", "black", "blue", "green", "yellow", "orange", "brown", "grey", "gray", "silver", "gold",
            "golden", "pink", "purple", "dark", "light", "bright", "metal", "metallic", "wooden", "wood", "plastic",
            "glass", "brass", "steel", "view", "image", "picture", "drawing", "illustration", "cartoon", "icon",
            "photo", "close-up", "closeup", "background", "object", "device", "thing", "shape", "large", "small", "big",
            "round", "old", "new", "vintage", "modern", "stylized", "abstract", "simple", "top", "front", "side"}
FILLER = {"a", "an", "the", "or", "and", "of", "on", "in", "with", "it", "its", "at", "to", "from", "by", "for", "is",
          "are", "being", "some", "that", "this", "into", "out", "up", "down", "over", "under", "near", "next"}

_ONT: Optional[Dict] = None
_CACHE: Dict[str, Dict] = {}
LOG: list = []


# ---------------------------------------------------------------------------------------------------------------
def ontology() -> Dict[str, Dict]:
    """name -> ontology entry (id, description, ...), from the vendored official ontology.json."""
    global _ONT
    if _ONT is None:
        _ONT = {x["name"]: x for x in json.loads(ONTOLOGY.read_text(encoding="utf-8"))}
    return _ONT


def description(label: str) -> str:
    return " ".join((ontology().get(label, {}).get("description") or "").split())


def path_of(label: str) -> str:
    from src.labels import ancestors
    return " > ".join(list(reversed(ancestors(label))) + [label])


def _toks(s: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9\-']+", (s or "").lower().replace("'s", "")) if w]


def _sing(w: str) -> str:
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _wset(s: str) -> set:
    out = set()
    for w in _toks(s):
        out |= {w, _sing(w)}
        if "-" in w:
            out |= {p for p in w.split("-") if p} | {_sing(p) for p in w.split("-") if p}
    return out


def vocab(label: str) -> set:
    """The words the OBJECT's head noun may come from: the label's names, its ancestors' names, its description."""
    from src.labels import ancestors, label_names
    v = set()
    for x in [label] + ancestors(label):
        for n in label_names(x) or [x.lower()]:
            v |= _wset(n)
    return v | _wset(description(label))


# ontology branches whose names are acoustics or sound shapes, not things ("Inside, small room", "Beep, bleep")
NON_SOURCE = ("Channel, environment and background", "Source-ambiguous sounds")
_SRC_WORDS: Optional[Dict[str, bool]] = None


def _names_a_source(w: str) -> bool:
    """False for a word that is only the name of acoustics / sound-shape labels (inside, beep, ding); True otherwise
    (people, places and every other label name stay refused)."""
    global _SRC_WORDS
    from src.labels import _parents, label_names, ancestors
    if _SRC_WORDS is None:
        _SRC_WORDS = {}
        for lab in _parents():
            other = not any(b == lab or b in ancestors(lab) for b in NON_SOURCE)
            for n in label_names(lab):
                if " " not in n:
                    _SRC_WORDS[n] = _SRC_WORDS.get(n, False) or other
    for x in (w, _sing(w)):
        if x in _SRC_WORDS:
            return _SRC_WORDS[x]
    return True


def _human(label: str) -> bool:
    from src.labels import ancestors
    return label == "Human sounds" or "Human sounds" in ancestors(label)


def _article(p: str) -> str:
    p = p.strip()
    if re.match(r"^(a|an|the|two|some|one)\s", p, re.I):
        return p[0].lower() + p[1:]
    last = (p.split() or [""])[-1].lower()
    if last.endswith("s") and not last.endswith(("ss", "us", "is")):     # a plural object takes no article
        return p[0].lower() + p[1:]
    return ("an " if p[:1].lower() in "aeiou" else "a ") + p


def _latin(s: str) -> str:
    return " ".join("".join(c if ord(c) < 0x250 else " " for c in (s or "")).split())


# ---------------------------------------------------------------------------------------------------------------
def cache_path(model: Optional[str] = None) -> Path:
    model = model or config.GEN_MODEL
    return CACHE_DIR / (model.split("/")[-1] + ".json")


def _cache(model: Optional[str] = None) -> Dict:
    p = cache_path(model)
    if str(p) not in _CACHE:
        d = {}
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                d = {}
        d.setdefault("picture_model", model or config.GEN_MODEL)
        d.setdefault("labels", {})
        _CACHE[str(p)] = d
    return _CACHE[str(p)]


def _save(model: Optional[str] = None) -> None:
    p = cache_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache(model), indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _ask(prompt: str, images=None, max_new: int = 48) -> str:
    from src.stage5_cross_modal_analysis import reason as R
    from src.stage6_visual_augmentation.verify import _vlm
    mdl, proc = _vlm()
    return R._ask(mdl, proc, prompt, images=images, max_new=max_new)


# ---------------------------------------------------------------------------------------------------------------
# (a) slot form
def parse_slots(raw: str) -> Optional[Dict[str, str]]:
    out = {}
    for line in (raw or "").replace("\r", "").split("\n"):
        m = re.match(r"^\W*(object|where|action|cue)\W*[:\-]\s*(.+)$", line.strip(), re.I)
        if m and m.group(1).lower() not in out:
            v = _latin(m.group(2)).strip().strip('"').strip("*").strip().rstrip(".").strip()
            out[m.group(1).lower()] = v
    return out if all(out.get(k) for k in SLOTS) else None


def sentence_of(slots: Dict[str, str]) -> str:
    """The fixed template: '{a/an} {object} {where}, {action}, {cue}'."""
    return f"{_article(slots['object'])} {slots['where']}, {slots['action']}, {slots['cue']}"


def guard_slots(slots: Optional[Dict[str, str]], label: str, family: str) -> List[str]:
    """Why the slots are refused ([] = accepted). Mechanical list checks only."""
    from src.stage5_cross_modal_analysis import reason as R
    from src.labels import names_forbidden, other_branch_makers
    if not slots:
        return ["unparsed"]
    why = []
    for k in SLOTS:
        n = len(slots[k].split())
        if n < 1 or n > 8:
            why.append(f"{k} length {n}")
    obj_words = [w for w in _toks(slots["object"]) if w not in FILLER]
    v = vocab(label)
    head = obj_words[-1] if obj_words else ""
    if not head or not ({head, _sing(head)} & v):
        why.append("object head '" + head + "' not in label/ancestors/description")
    text = sentence_of(slots)
    desc_words = _wset(description(label))
    # the object's own words, the description's words and "-ing" words (the sound's action: ringing, cheering,
    # hissing) are not extra sources
    own = _wset(slots["object"]) | desc_words | {w for w in _toks(text) if w.endswith("ing") and len(w) > 5
                                                          and w not in R.PLACES}
    bad = [w for w in R.expand_guard(text, slots["object"], label, family)
           if w not in own and _sing(w) not in own]
    bad = [w for w in bad if _names_a_source(w)]
    human = _human(label)
    if human:
        bad = [w for w in bad if w not in R.PEOPLE and _sing(w) not in R.PEOPLE and w not in PEOPLE_X]
    if bad:
        why.append("extra object/person/place " + ",".join(bad))
    nf = [n for n in names_forbidden(text, label, []) if not (_wset(n) <= own)]
    if nf:
        why.append("forbidden " + ",".join(nf))
    alien = [w for w in other_branch_makers(_toks(text), label) if w not in own]
    if alien:
        why.append("other-branch " + ",".join(alien))
    tw = _wset(text) & TEXT_WORDS
    if tw:
        why.append("text " + ",".join(sorted(tw)))
    if not human:
        pw = _wset(text) & (set(R.PEOPLE) | PEOPLE_X)
        if pw:
            why.append("people " + ",".join(sorted(pw)))
    return why


def slot_form(spec) -> Dict:
    """The slot form for the sound's label (cached): {"slots", "sentence" ("" = refused), "object", "why", "raw"}."""
    label = getattr(spec, "source", "") or spec.event_label
    c = _cache()
    ent = c["labels"].setdefault(label, {})
    if ent.get("slots") and ent["slots"].get("llm") == config.VLM_MODEL:
        return ent["slots"]
    prompt = SLOT_PROMPT.format(name=label, path=path_of(label), desc=description(label) or "(none)",
                                people="" if _human(label) else ", no people or hands")
    raw = _ask(prompt, max_new=96)
    slots = parse_slots(raw)
    why = guard_slots(slots, label, spec.event_label or label)
    sent = sentence_of(slots) if slots and not why else ""
    out = {"llm": config.VLM_MODEL, "raw": raw, "slots": slots, "why": why, "sentence": sent,
           "object": _article(slots["object"]) if slots and not why else ""}
    ent["slots"] = out
    _save()
    print(f"       [sense] slots {label!r}: {slots} -> {'OK ' + repr(sent) if sent else 'REFUSED ' + '; '.join(why)}",
          flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------
# (b) mistake mining
def mine_seed(label: str, k: int) -> int:
    return zlib.crc32(f"sense-mine|{label}|{k}".encode()) & 0x7FFFFFFF


def _norm_answer(ans: str) -> str:
    a = _latin(ans).strip().strip('"').strip().rstrip(".").strip()
    a = re.sub(r"^(the main thing (in this picture )?is|it is|this is|it's)\s+", "", a, flags=re.I)
    a = re.sub(r"^(a|an|the)\s+", "", a, flags=re.I)
    return a[:1].lower() + a[1:] if a else a


def _keep_words(label: str, drawn: str, target: str) -> set:
    from src.labels import ancestors, label_names
    k = _wset(drawn) | _wset(target)
    for x in [label] + ancestors(label):
        for n in label_names(x) or [x.lower()]:
            k |= _wset(n)
    return k


def neg_words(answer: str, keep: set) -> List[str]:
    # "-ing" words are what the thing does ("steaming kettle" -> kettle), not a thing to keep out of the picture
    return [w for w in _toks(answer) if w not in keep and _sing(w) not in keep and w not in FILLER
            and w not in NEG_STOP and len(w) > 2 and not w.isdigit() and not w.endswith("ing")]


def mine(spec, subject: str, drawn: str, target: str, model: str, device: str, size) -> Dict:
    """4 pictures of the plain subject, the open question, the closed judge. Cached per (label, picture model)."""
    label = getattr(spec, "source", "") or spec.event_label
    c = _cache(model)
    ent = c["labels"].setdefault(label, {})
    if ent.get("mined") and ent["mined"].get("llm") == config.VLM_MODEL:
        return ent["mined"]
    from PIL import Image
    from benchmark.gold.picture_templates import RULES_TAIL, negative_for as screen_negative
    from src.stage6_visual_augmentation import _diffusion_image
    d = CACHE_DIR / "mine" / model.split("/")[-1]
    d.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    keep = _keep_words(label, drawn, target)
    rows, looks, negs = [], [], []
    for k in range(N_MINE):
        p = d / f"{slug}_{k}.png"
        seed = mine_seed(label, k)
        if not p.exists():
            ok = _diffusion_image(p, subject + RULES_TAIL, size, model=model, device=device, seed=seed,
                                  negative=screen_negative(subject) or " ")
            if not ok:
                rows.append({"k": k, "seed": seed, "error": "draw failed"})
                continue
        img = Image.open(p).convert("RGB")
        small = img.resize((768, 768)) if img.size[0] > 768 else img
        ans = _norm_answer(_ask(MINE_PROMPT, images=[small], max_new=24))
        if img.size[0] > 512:
            img.resize((512, 512), Image.LANCZOS).save(p)
        verdict = _ask(JUDGE_PROMPT.format(target=target, name=label, desc=description(label) or label, answer=ans),
                       max_new=4).strip().lower()
        match = verdict.startswith("yes")
        row = {"k": k, "seed": seed, "answer": ans, "judge": verdict, "match": match}
        if not match and ans:
            nw = neg_words(ans, keep)
            row["neg"] = nw
            for w in nw:
                if w not in negs:
                    negs.append(w)
            content = {w for w in _wset(ans) if w not in FILLER and w not in NEG_STOP}
            opt = _article(ans)
            if content and not content <= keep and opt not in looks and len(looks) < MAX_LOOK:
                looks.append(opt)
        rows.append(row)
        print(f"       [sense] mine {label!r} k{k}: {ans!r} -> judge {verdict!r}", flush=True)
    out = {"llm": config.VLM_MODEL, "subject": subject, "drawn": drawn, "target": target, "rows": rows,
           "lookalikes": looks, "neg": negs}
    ent["mined"] = out
    _save(model)
    return out


# ---------------------------------------------------------------------------------------------------------------
def plan(spec, subject: str, model: str, device: str, size) -> Dict:
    """What arm C draws: the slot sentence (or the plain subject), the mined negatives, and its checker's intended /
    look-alikes. Called once per picture by stage6._final_picture under PICTURE_SENSE."""
    label = getattr(spec, "source", "") or spec.event_label
    sf = slot_form(spec)
    drawn = sf["sentence"] or subject
    target = sf["object"] or (label.split(",")[0].split("(")[0].strip().lower())
    mn = mine(spec, subject, drawn, target, model, device, size)
    out = {"subject": drawn, "neg": ", ".join(mn["neg"]), "intended": sf["object"] or _phrase(subject),
           "lookalikes": list(mn["lookalikes"]), "slots_ok": bool(sf["sentence"])}
    LOG.append({"label": label, **out})
    return out


def _phrase(subject: str) -> str:
    from src.stage6_visual_augmentation.verify import _phrase as vp
    return vp(subject)


def own_options(spec, subject: str) -> Dict:
    """C's own table-free checker options: intended "a <object>" (or the plain subject), mined look-alikes, generics
    outside the sound's category (verify.GENERIC)."""
    from src.stage6_visual_augmentation.verify import GENERIC, _words
    label = getattr(spec, "source", "") or spec.event_label
    ent = _cache()["labels"].get(label, {})
    sf, mn = ent.get("slots") or {}, ent.get("mined") or {}
    intended = sf.get("object") or _phrase(subject or label)
    conf = list(mn.get("lookalikes") or [])
    cat = _words(" ".join([subject or "", intended, label, spec.event_label or ""]))
    for opt, words in GENERIC:
        if not (cat & words):
            conf.append(opt)
    return {"intended": intended, "confusions": conf}


# fixed per-sound options for the test (the union of the arms' look-alikes), keyed "tag|label"; set by the runner
FIXED: Dict[str, Dict] = {}


def _ask_mc(img, spec, o: Dict, salt: int, tag: str) -> Dict:
    from src.stage6_visual_augmentation.verify import _order_seed, SOMETHING_ELSE
    opts = [o["intended"]] + [c for c in o["confusions"] if c != o["intended"]]
    random.Random(_order_seed(spec, salt, tag)).shuffle(opts)
    opts.append(SOMETHING_ELSE)
    assert len(opts) <= len(LETTERS), "more options than letters"
    letters = LETTERS[: len(opts)]
    q = MC_PROMPT.format(options="\n".join(f"{l}. {t}" for l, t in zip(letters, opts)))
    ans = _ask(q, images=[img], max_new=8)
    m = re.search(r"\b([A-T])\b", ans.upper()) or re.search(r"([A-T])", ans.upper())
    pick = opts[letters.index(m.group(1))] if m and m.group(1) in letters else None
    return {"options": opts, "answer": ans, "picked": pick, "intended": o["intended"], "entry": "sense",
            "ok": pick == o["intended"]}


def _check_with(o: Dict, path, spec, salt: int, tag: str, see: bool) -> Dict:
    from PIL import Image
    from src.stage6_visual_augmentation.verify import text_found, what_it_sees
    img = Image.open(path).convert("RGB")
    small = img.resize((768, 768)) if img.size[0] > 768 else img
    mc = _ask_mc(small, spec, o, salt, tag)
    txt = text_found(img)
    out = {"ok": bool(mc["ok"] and txt["ok"]), "mc": mc, "text": txt}
    if see:
        out["saw"] = what_it_sees(small)
    return out


def check(path, spec, subject: str, salt: int = 0, tag: str = "", see: bool = True) -> Dict:
    """PICTURE_SENSE's checker (same shape as verify.check): C's own options, the same OCR text check."""
    return _check_with(own_options(spec, subject), path, spec, salt, tag, see)


def check_fixed(path, spec, subject: str, salt: int = 0, tag: str = "", see: bool = True) -> Dict:
    """Test only: the fixed union question for this sound (FIXED), the same for every arm."""
    label = getattr(spec, "source", "") or spec.event_label
    return _check_with(FIXED[f"{tag}|{label}"], path, spec, salt, tag, see)
