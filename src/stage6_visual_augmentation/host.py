"""PICTURE_HOST (4 Oct 2026; Adam: "a siren was a red thingy instead of an ambulance with its siren lights"; three-Fable
panel, three blind rounds, blind rounds on DEV only).

For every sound under the AudioSet "Alarm" node (siren, fire alarm, smoke detector, alarm clock, doorbell, horns,
whistles, telephone ...; by ANY parent, the ontology is a DAG) the shipped subject's head noun is a signalling DEVICE,
and the generator draws the head noun large: a lone beacon, a bell. One general rule replaces that subject:

  1. HOST step (the pipeline VLM, text only): is the device a PART of a larger thing (answer that thing, general enough
     to cover every kind of it in the ontology, never one of those kinds) or a whole object (answer MOUNT: the surface
     it is seen on); plus what a person SEES when it goes off.
  2. Mechanical guards as in stage 5 (names_forbidden, people, other_branch_makers, no category word as host), and on
     the sign: a clause that only says it makes a sound, or names another maker (a bell), is dropped.
  3. subject = PART / MOUNT / ALONE / FALLBACK below; the white-background RULES_TAIL is unchanged.

Every sound outside the family returns None and keeps its shipped prompt, negative and seed byte for byte.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

HOST_PROMPT = (
    "A sound detector heard: {thing}. The current picture description is: {subject}. Kinds of {thing} in the sound "
    "list: {children}. The sound list describes it as: {description}"
    + chr(10) +
    "{thing} is a warning or signal device. If the current picture description already names the larger thing it is "
    "fixed to, or the sound list says it is made by, fitted to or attached to a larger thing, it is a PART. "
    "If it is a PART of a larger thing, answer that whole larger thing in one "
    "to three words, something a viewer knows by its shape, general enough to cover every kind listed, never one of "
    "the kinds themselves, never a person, never a place. If it is a whole object on its own, answer MOUNT: and the "
    "surface a person usually sees it on, at the height they see it (wall at eye height, ceiling, pole, desk, "
    "bedside table, roof); if the description says where it is mounted, use that. Never answer a general "
    "word such as device, object, machine, thing, equipment or system."
    + chr(10) +
    "Then write a semicolon and, in 3 to 8 words, the visible change on that object itself when it goes off (its own "
    "light, moving part or shaking); no light unless it really has one. Answer in the form: answer; what you see"
)
PART = "a whole {host} seen from the front with no writing on it, its {thing} going off{sign}"
MOUNT = "one large {thing} going off, fixed on a {mount}, filling the picture, no writing on it{sign}"
ALONE = "one large {thing} going off, filling the picture, no writing on it{sign}"
FALLBACK = "a wall-mounted alarm box going off, red strobe flashing, no writing on it"
# added to the negative, minus any word the subject itself uses; the comic sound marks of round 2 read as explosions
FAMILY_NEG = ["close-up of a single device", "lone beacon", "bell", "church bell", "hand bell", "wifi symbol",
              "signal icon", "vibration lines", "lettering", "logo", "label plate", "signage",
              "explosion", "starburst", "comic burst", "sparks", "fire", "flames"]
AUDIO = {"sound", "sounds", "noise", "loud", "waves", "emits", "rings", "ringing", "ring", "beeps", "beeping",
         "blares", "horn", "claps", "chimes", "tone", "tones"}
MOTION = ", shaking with motion blur"         # the cue Adam picked (round-2 alarm clock) when no visible sign is left
GENERIC = {"device", "object", "machine", "thing", "equipment", "system", "appliance", "unit", "item"}
VERBS = {"ringing", "rings", "ring", "sounding", "sounds", "blaring", "beeping", "buzzing", "going", "off", "goes",
         "wailing", "spinning", "honking", "a", "an", "the"}
PEOPLE = {"person", "man", "woman", "people", "someone", "boy", "girl", "child", "hand", "hands"}


def toks(s: str) -> set:
    return set("".join(c if c.isalnum() else " " for c in s.lower()).split())


@lru_cache(maxsize=1)
def _onto():
    onto = json.loads((Path(__file__).resolve().parents[1] / "audioset_ontology.json").read_text(encoding="utf-8"))
    by_name = {x["name"]: x for x in onto}
    by_id = {x["id"]: x for x in onto}
    family, todo = set(), ["Alarm"]
    while todo:
        n = todo.pop()
        if n not in family:
            family.add(n)
            todo += [by_id[c]["name"] for c in by_name[n]["child_ids"]]
    return by_name, by_id, frozenset(family)


def in_family(source: str) -> bool:
    return source in _onto()[2]


def family_negative(subject: str) -> str:
    st = toks(subject)
    return ", ".join(n for n in FAMILY_NEG if not (toks(n) & st))


def host_subject(source: str, subject: str, ask) -> Optional[dict]:
    """The new subject for a family sound, or None outside the family. `ask(prompt) -> str` is the pipeline VLM
    (text only). Returns {"subject", "neg", "log"}."""
    from src.labels import names_forbidden, label_names, other_branch_makers, ancestors
    by_name, by_id, family = _onto()
    if source not in family:
        return None
    thing = " ".join(w for w in subject.lower().split() if w not in VERBS) or label_names(source)[0]
    named = sorted(n for n in family if thing in label_names(n))     # the subject names a kind ("fire alarm")
    node = named[0] if named else source
    kids = [label_names(by_id[c]["name"])[0] for c in by_name[node]["child_ids"]]
    raw = ask(HOST_PROMPT.format(thing=thing, subject=subject, children=", ".join(kids) or "none",
                                 description=by_name[node]["description"]))
    ans, _, sign = raw.partition(";")
    mount = ans.strip().upper().startswith("MOUNT")
    ans = ans.split(":", 1)[1] if mount and ":" in ans else ans

    def clean(x, keep=" -"):
        return " ".join("".join(c if (c.isalnum() or c in keep) else " " for c in x).split()).lower()
    host, sign = clean(ans), clean(sign, " -,")
    for art in ("a ", "an ", "the "):
        host = host[len(art):] if host.startswith(art) else host
    had = toks(subject)                      # words the shipped guards already accepted
    keep = []
    for cl in sign.replace(" and ", ",").split(","):
        w = toks(cl)
        if (not w or w & AUDIO or w & {"no", "none", "nothing"}
                or (w & {"bell", "bells", "clapper", "clappers"} and not toks(thing) & {"bell", "bells"})):
            continue
        if other_branch_makers(cl.split(), source) or [b for b in names_forbidden(cl, source, [])
                                                       if not toks(b) <= had]:
            continue
        keep.append(cl.strip())
    sign = ", ".join(keep)
    why = []
    if not host or host == "none":
        why.append("NONE")
    elif toks(host) <= GENERIC:
        why.append("generic")
    elif not mount:
        cats = [n for x in ancestors("Alarm") + ["Alarm"] for n in label_names(x)]
        if host == thing or host in cats:
            why.append("host is the thing or its kind")
        bad = [b for b in names_forbidden(host, source, []) if not toks(b) <= had]
        if bad:
            why.append("forbidden " + ",".join(bad))
        if PEOPLE & toks(host):
            why.append("person")
        alien = [w for w in other_branch_makers(host.split(), source) if w not in had]
        if alien:
            why.append("other-branch " + ",".join(alien))
    if thing == "alarm":
        why.append("family root")            # "an alarm" alone draws an alarm clock
    sg = (", " + sign) if sign else MOTION
    if not why:
        subj = MOUNT.format(thing=thing, mount=host, sign=sg) if mount else PART.format(host=host, thing=thing, sign=sg)
    elif len(thing.split()) > len(label_names(source)[0].split()):
        subj = ALONE.format(thing=thing, sign=sg)
    else:
        subj = FALLBACK
    return {"subject": subj, "neg": family_negative(subj),
            "log": {"thing": thing, "node": node, "raw": raw, "host": host, "mount": mount, "sign": sign,
                    "refused": why, "subject": subj}}
