"""Shared AudioSet-label helpers: which sounds we visualize, and how we merge
the label families PANNs emits for a single real-world source.

Used by the pipeline (Stage 5) and by benchmark/curate.py so the two stay in
sync (the "one entry per real source" + "music non-salient for now" conventions).
"""
from __future__ import annotations

from typing import List

from src.types import AudioEvent

# Speech / silence: not the non-speech ambient sound we target.
SPEECH_LABELS = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Narration, monologue", "Conversation", "Speech synthesizer",
    "Child speech, kid speaking", "Hubbub, speech noise, speech babble", "Chatter",
    "Whispering",
}
# Non-word vocal events: no words for a caption to carry, real scene events (a scream off
# screen). Drawn like Laughter. Moved out of SPEECH_LABELS on 2026-09-21 (Adam + Fable panel:
# SDH always tags [screaming], [shouting]).
VOCAL_EVENTS = {"Shout", "Yell", "Children shouting", "Screaming"}
# Steady textures with no event and no drawable source: never drawn, whatever family they map
# to (Rumble -> Thunder is kept for merging real thunder, not for drawing a bare rumble).
TEXTURE_LABELS = {"Breathing", "Rumble", "Hum", "Whir", "Rustle", "Rustling"}
# Musical instruments / score elements — treat as music (non-salient for now, A6),
# so film/trailer soundtracks aren't mistaken for real ambient sound.
INSTRUMENTS = {
    "Piano", "Electric piano", "Organ", "Keyboard (musical)", "Synthesizer", "Sampler",
    "Orchestra", "Violin, fiddle", "Cello", "Bowed string instrument", "Plucked string instrument",
    "Guitar", "Acoustic guitar", "Electric guitar", "Bass guitar", "Banjo", "Harp",
    "Drum", "Drum kit", "Percussion", "Cymbal", "Snare drum", "Timpani",
    "Trumpet", "Trombone", "French horn", "Flute", "Choir", "Singing", "Mantra", "Theremin",
    # brass/woodwind: were slipping through as a "salient ambient sound" and got
    # suggested as off-screen sources on film-score soundtracks (2026-08-22).
    "Saxophone", "Brass instrument", "Wind instrument, woodwind instrument",
    "Bagpipes", "Accordion", "Harmonica", "Gong", "Singing bowl", "Bell cymbal",
    "Male singing", "Female singing", "Child singing", "Humming", "Whistling",
    # found polluting benchmark suggestions as "off-screen ambient sound" (2026-08-22)
    "Strum", "String section", "Double bass", "Marimba, xylophone", "Glockenspiel",
    "Electronic organ", "Hammond organ", "Tuning fork", "Cowbell", "Steelpan",
    "Zither", "Ukulele", "Mandolin", "Sitar", "Tabla", "Vibraphone", "Rimshot",
}
# Acoustic-environment / ambience descriptors: real, but not a discrete source to depict.
SCENE_LABELS = {
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural", "Reverberation", "Echo",
    "Noise", "Environmental noise", "Background noise", "White noise", "Pink noise",
    "Silence", "Field recording", "Sound effect", "Wind noise (microphone)",
    # Bare "Wind" is almost always mic/airflow artifact in our clips (riding, cycling,
    # handheld outdoors) rather than a depictable event -- and it was a repeat source of
    # false augmentations. Treated as ambience. See notes sec:annotation.
    "Wind", "Rustling leaves", "Howl (wind)",
}
# Music: flagged; non-salient for now (decision: handle diegetic music later, task A6).
MUSIC_LABELS = {
    "Music", "Scary music", "Background music", "Musical instrument", "Soundtrack music",
    "Theme music", "Sad music", "Happy music", "Exciting music",
    # genre labels that don't end in "music" (leaked into a benchmark clip as "Dubstep")
    "Dubstep", "Techno", "Disco", "Reggae", "Jazz", "Blues", "Funk", "Opera",
    "Swing music", "Beatboxing", "Rapping", "Drum and bass", "House music",
    "Punk rock", "Heavy metal", "Rock and roll", "Bluegrass", "Flamenco", "Salsa",
    "Soul music", "Gospel music", "Ska", "Grunge", "Progressive rock", "Psychedelic rock",
}


# Generic superclasses: too vague to visualize well, and usually redundant with a
# specific sibling (PANNs fires "Animal" + "Domestic animals" alongside "Dog").
GENERIC_LABELS = {
    "Animal", "Domestic animals, pets", "Wild animals",
    "Livestock, farm animals, working animals", "Sounds of things", "Mechanisms",
    "Human sounds", "Domestic sounds, home sounds", "Human group actions",
    "Generic impact sounds", "Surface contact", "Onomatopoeia",
}


_MUSIC_RE = None


def is_music(label: str) -> bool:
    """Music/instrument check. Also catches AudioSet's open-ended genre and
    regional-music labels ("Music of Asia", "Music of Bollywood", "Bluegrass"),
    which no fixed list can enumerate."""
    global _MUSIC_RE
    if label in MUSIC_LABELS or label in INSTRUMENTS or label.endswith("music"):
        return True
    if _MUSIC_RE is None:
        import re
        _MUSIC_RE = re.compile(r"music|singing|soundtrack", re.I)
    return bool(_MUSIC_RE.search(label))


ENV_BRANCH = "Channel, environment and background"   # AudioSet ontology branch: room tone, noise, radio/TV, hum


def is_salient_nonspeech(label: str) -> bool:
    """True for a sound worth (potentially) visualizing.

    Two rules, chosen by config.LABEL_FILTER:
      "lists" (v1-v4ab): hand lists SPEECH / SCENE / GENERIC written around BEATs' 527 names
              -- a detector with other names (PSED's AudioSet-Strong set) slips past them
              (docs/prereg_v4.md, v4ab diagnosis);
      "branch" (v4ab2 on): only speech, music and the ontology branch "Channel, environment
              and background" are kept off screen (Adam, 20 Sept 2026); everything else may
              be drawn, whatever the detector calls it. Detector-neutral by construction.
    """
    import config
    mode = getattr(config, "LABEL_FILTER", "lists")
    if mode == "depictable":
        # v4ab3 / v4b3 (docs/prereg_v4.md) with amendment 2 (2026-09-21, Adam + Fable panel,
        # declared before the re-run): draw a label only if a captioner would write it as a
        # bracket tag AND it names a source one can picture. Order: (1) steady textures are out
        # on the raw name (Wind subtree, Breathing, Rumble, Hum ...); (2) the label is mapped to
        # its family first (Bang -> Explosion, Beep -> Alarm, Smash -> Glass, Ding -> Bell), so
        # a gunshot the detector calls "Bang" is drawn as an explosion; (3) on the mapped name:
        # speech with words and music are never drawn, nor the recording/environment branch,
        # bare category names, "no source" names (Generic impact sounds, Onomatopoeia, Sound
        # effect, Video game sound, Silence) and any Source-ambiguous label left unmapped.
        # Non-word vocal events (Shout, Scream, Laughter, Cough, Snoring ...) are drawn.
        # Same rule for every detector.
        if label in TEXTURE_LABELS or label == "Wind" or is_descendant(label, "Wind"):
            return False
        lab = canonical(label)
        if lab in SPEECH_LABELS or any(is_descendant(lab, sp) for sp in SPEECH_LABELS):
            return False
        if is_music(lab) or is_descendant(lab, "Singing"):
            return False
        if lab in (ENV_BRANCH, "Silence", "Sound effect", "Video game sound", "Human voice", "Respiratory sounds")                 or is_descendant(lab, ENV_BRANCH) or lab in GENERIC_LABELS or lab in TEXTURE_LABELS:
            return False
        if lab == "Source-ambiguous sounds" or is_descendant(lab, "Source-ambiguous sounds"):
            return False          # unmapped ambiguous label: nothing to picture
        return True
    if mode == "branch":
        return not (label in SPEECH_LABELS or is_music(label) or label in (ENV_BRANCH, "Silence", "Sound effect")
                    or is_descendant(label, ENV_BRANCH))
    return not (label in SPEECH_LABELS or label in SCENE_LABELS
                or label in GENERIC_LABELS or is_music(label))


# PANNs fires a whole family of labels for one real source. Map the common
# sub-labels to a canonical parent so we visualize one image per source, not five.
# (Heuristic, conservative: only clear sub-types; generic "Animal" is left alone.)
FAMILY = {
    # vehicles / traffic
    "Car": "Vehicle", "Truck": "Vehicle", "Bus": "Vehicle", "Motor vehicle (road)": "Vehicle",
    "Vehicle horn, car horn, honking": "Vehicle", "Toot": "Vehicle",
    "Traffic noise, roadway noise": "Vehicle", "Motorcycle": "Vehicle",
    # dog
    "Bark": "Dog", "Bow-wow": "Dog", "Yip": "Dog", "Whimper (dog)": "Dog",
    "Canidae, dogs, wolves": "Dog", "Growling": "Dog", "Howl": "Dog",
    # cat
    "Meow": "Cat", "Purr": "Cat", "Hiss": "Cat", "Cat communication": "Cat",
    # water
    "Waves, surf": "Water", "Ocean": "Water", "Waterfall": "Water", "Stream": "Water",
    "Gurgling": "Water", "Slosh": "Water", "Trickle, dribble": "Water", "Drip": "Water",
    # train / rail
    "Railroad car, train wagon": "Train", "Rail transport": "Train",
    "Subway, metro, underground": "Train", "Train wheels squealing": "Train",
    "Train horn": "Train", "Train whistle": "Train", "Clickety-clack": "Train",
    # engines: generic engine sounds are almost always road traffic in our clips
    "Engine": "Vehicle", "Engine starting": "Vehicle", "Idling": "Vehicle",
    "Accelerating, revving, vroom": "Vehicle",
    "Light engine (high frequency)": "Vehicle", "Medium engine (mid frequency)": "Vehicle",
    "Heavy engine (low frequency)": "Vehicle",
    # aircraft / boat
    "Jet engine": "Aircraft", "Fixed-wing aircraft, airplane": "Aircraft",
    "Aircraft engine": "Aircraft", "Propeller, airscrew": "Aircraft",
    "Motorboat, speedboat": "Boat", "Boat, Water vehicle": "Boat", "Ship": "Boat",
    "Sailboat, sailing ship": "Boat", "Motor vehicle (road)": "Vehicle",
    # thunder / storm
    "Thunderstorm": "Thunder", "Rumble": "Thunder",
    # rain
    "Raindrop": "Rain", "Rain on surface": "Rain",
    # bird / poultry
    "Bird vocalization, bird call, bird song": "Bird", "Chirp, tweet": "Bird", "Pigeon, dove": "Bird",
    "Chicken, rooster": "Bird", "Crowing, cock-a-doodle-doo": "Bird", "Fowl": "Bird",
    "Cluck": "Bird", "Coo": "Bird", "Caw": "Bird",
    # crowd
    "Cheering": "Crowd", "Applause": "Crowd", "Children shouting": "Crowd", "Hubbub": "Crowd",
    # bell
    "Church bell": "Bell", "Chime": "Bell", "Chimes": "Bell", "Wind chime": "Bell",
    "Change ringing (campanology)": "Bell", "Bell": "Bell",
    # emergency siren -> canonical Siren (has its own visible concept: an emergency vehicle)
    "Ambulance (siren)": "Siren", "Police car (siren)": "Siren", "Emergency vehicle": "Siren",
    "Fire engine, fire truck (siren)": "Siren", "Civil defense siren": "Siren",
    # fireworks
    "Firecracker": "Fireworks", "Fireworks": "Fireworks",
    # guns / blasts -> one canonical source
    "Machine gun": "Gunshot", "Gunshot, gunfire": "Gunshot", "Fusillade": "Gunshot",
    "Artillery fire": "Gunshot", "Cap gun": "Gunshot",
    "Boom": "Explosion", "Eruption": "Explosion", "Bang": "Explosion",
    # breaking glass
    "Shatter": "Glass", "Breaking": "Glass", "Smash, crash": "Glass",
    # horse
    "Clip-clop": "Horse", "Neigh, whinny": "Horse",
    # livestock (kept distinct from Horse: different depiction)
    "Bleat": "Sheep", "Sheep": "Sheep", "Goat": "Sheep",
    "Moo": "Cattle", "Cattle, bovinae": "Cattle", "Cowbell (livestock)": "Cattle",
    "Oink": "Pig", "Pig": "Pig",
    # racing = road vehicles
    "Race car, auto racing": "Vehicle", "Skidding": "Vehicle", "Tire squeal": "Vehicle",
    "Reversing beeps": "Vehicle",
    # alarms
    "Car alarm": "Alarm", "Fire alarm": "Alarm", "Alarm clock": "Alarm",
    "Smoke detector, smoke alarm": "Alarm", "Buzzer": "Alarm", "Beep, bleep": "Alarm",
    # resonator sounds a captioner writes as [bell] / [ding] (amendment 2026-09-21)
    "Ding": "Bell", "Ringing (of resonator)": "Bell",
    # telephone
    "Telephone bell ringing": "Telephone", "Ringtone": "Telephone",
    "Telephone dialing, DTMF": "Telephone",
    # kitchen / household
    "Dishes, pots, and pans": "Dishes", "Cutlery, silverware": "Dishes",
    "Frying (food)": "Cooking", "Chopping (food)": "Cooking", "Sizzle": "Cooking",
    "Door": "Door", "Slam": "Door", "Cupboard open or close": "Door",
    "Creak": "Door", "Squeak": "Door",
    # people moving
    "Walk, footsteps": "Footsteps", "Run": "Footsteps", "Shuffle": "Footsteps",
    # liquids -> Water
    "Liquid": "Water", "Pour": "Water", "Fill (with liquid)": "Water",
    "Water tap, faucet": "Water", "Sink (filling or washing)": "Water",
    # instruments (a visible musician) -> canonical Saxophone concept
    "Brass instrument": "Saxophone", "Wind instrument, woodwind instrument": "Saxophone",
    "Saxophone": "Saxophone",
}


# Wide-band, low-structure textures that PANNs routinely reports on noisy recordings
# (mic hiss read as rain, room rumble read as traffic). They need more evidence than a
# structured sound like a bark or a siren before we act on them. See notes sec:annotation.
NOISE_LIKE_MIN_CONF = {"Rain": 0.30, "Vehicle": 0.20, "Water": 0.25,
                       "Thunder": 0.25, "Boat": 0.30, "Wind": 0.35}


def min_confidence(label: str, default: float) -> float:
    return max(default, NOISE_LIKE_MIN_CONF.get(label, 0.0))


def canonical(label: str) -> str:
    return FAMILY.get(label, label)


# ----------------------------------------------------------------------
# AudioSet ontology: the detector's own taxonomy, used to recognise when two detected
# labels are one source.
#
# FAMILY above is 134 hand-written entries, and it covered none of the pairs that
# actually caused trouble: Laughter/Giggle/Belly laugh/Snicker/Chuckle/Baby laughter all
# absent, Owl/Hoot absent. Extending it by hand is the preset trap again -- the entries
# that are missing are the ones nobody thought of, and there are 527 classes.
#
# So the parent edges come from the ontology published with AudioSet itself
# (github.com/audioset/ontology, CC BY 4.0), reduced here to child -> parent and
# vendored as audioset_parents.json. It is not a list of categories chosen for this
# project; it is the label space PANNs was trained on, read correctly. Every sound the
# detector can emit is in it, so it generalises exactly as far as the detector does.
# ----------------------------------------------------------------------
_PARENTS = None


def _parents() -> dict:
    global _PARENTS
    if _PARENTS is None:
        import json
        from pathlib import Path
        try:
            _PARENTS = json.loads(
                (Path(__file__).with_name("audioset_parents.json")).read_text("utf-8"))
        except Exception:
            _PARENTS = {}
    return _PARENTS


def ancestors(label: str) -> list:
    """Every ancestor of `label`, nearest first. Empty if the label is unknown."""
    out, seen, cur = [], {label}, label
    par = _parents()
    while cur in par:
        cur = par[cur]
        if cur in seen:                     # the ontology is a DAG; do not loop
            break
        seen.add(cur)
        out.append(cur)
    return out


def is_descendant(child: str, ancestor: str) -> bool:
    """Is `child` a more specific kind of `ancestor` in the AudioSet ontology?"""
    return bool(child != ancestor and ancestor in ancestors(child))


def same_source(a: str, b: str) -> bool:
    """Do these two DETECTED labels describe one source?

    Only an ancestor/descendant relation counts, and that restriction is doing real
    work. Merging on a shared parent instead would put Air horn together with Siren
    (both are children of Alarm... in fact of different branches, but the shared-parent
    rule fails on plenty of other pairs) and Dog together with Sheep, since Animal is
    not marked abstract in the ontology and would happily swallow both. Requiring one
    label to BE the other's ancestor keeps exactly the cases where the detector reported
    a family and one of its members: Giggle under Laughter, Hoot under Owl.
    """
    return is_descendant(a, b) or is_descendant(b, a)


# Better image-search phrases than the bare AudioSet label (e.g. "Vehicle" alone
# returns toy photos). Used by Stage 6 retrieval. v2 (diffusion) won't need these.
QUERY_HINTS = {
    "Vehicle": "traffic cars street", "Dog": "dog", "Cat": "cat", "Water": "waterfall",
    "Thunder": "lightning", "Rain": "rain", "Bird": "bird", "Crowd": "crowd",
    "Siren": "ambulance", "Wind": "storm wind", "Fire": "fire flames",
    "Explosion": "explosion", "Aircraft": "airplane", "Helicopter": "helicopter",
    "Train": "train railway", "Bell": "church bell", "Gunshot, gunfire": "gun",
    "Applause": "applause audience", "Insect": "insect", "Horse": "horse",
}


def search_query(label: str) -> str:
    return QUERY_HINTS.get(label, label)


def _start_rule() -> str:
    import config
    return str(getattr(config, "MERGE_START", "earliest"))


BREAK_TOL = 0.05     # s: frame-grid slack when testing whether a gap between two bursts spans a recorded break


def crosses_break(breaks, end: float, start: float) -> bool:
    """R13-6: does the gap between a burst ending at `end` and one starting at `start` contain a recorded break?"""
    return any(g0 >= end - BREAK_TOL and g1 <= start + BREAK_TOL for g0, g1 in (breaks or ()))


def merge_by_label(events: List[AudioEvent], gap: float = 1.0, raw_labels: dict = None) -> List[AudioEvent]:
    """One event per label, carrying every separate BURST of that sound.

    This used to take min-start and max-end over every firing of a label, which made
    the display span a function of the weakest detection in the clip. In the rodeo demo
    a 0.23 Vehicle at 1.6-4.2 s was stretched to 1.6-17.3 s by three later firings at
    0.08, 0.09 and 0.12 -- noise -- and the viewer got a tractor beside the video for
    sixteen seconds while the one real sound in the clip, a siren, was correctly silent.

    Firings closer together than `gap` are one burst; anything further apart is a
    separate burst. ``start``/``end`` are the strongest burst, which is where the
    visibility check samples its frames; ``spans`` lists them all, which is what the
    panel shows. Sorted by confidence descending.
    """
    by_label = {}
    for e in sorted(events, key=lambda e: e.start):
        by_label.setdefault(e.label, []).append(e)
    out = []
    for label, firings in by_label.items():
        bursts = []                      # [start, end, conf, raw labels in the burst]
        # R13-6: breaks recorded by stage 4 (the family's evidence absences) are never bridged; with raw_labels given
        # (config.RETRIGGER_RAW), a firing that starts after the burst has ended and names a different sound (not the
        # same label, not an ancestor or descendant) starts a new burst, and the boundary is recorded as a break too.
        brk = sorted({tuple(b) for f in firings for b in (getattr(f, "breaks", None) or [])})
        extra = []
        for e in firings:
            join = bool(bursts) and e.start - bursts[-1][1] <= gap
            if join and brk and crosses_break(brk, bursts[-1][1], e.start):
                join = False
            if join and raw_labels is not None and e.start >= bursts[-1][1]:
                r = raw_labels.get(id(e))
                if r is not None and not any(r == q or same_source(r, q) for q in bursts[-1][3]):
                    join = False
                    extra.append((round(float(bursts[-1][1]), 3), round(float(e.start), 3)))
            if join:
                if raw_labels is not None and raw_labels.get(id(e)) is not None:
                    bursts[-1][3].add(raw_labels[id(e)])
                bursts[-1][1] = max(bursts[-1][1], e.end)
                # a burst starts where its STRONGEST firing starts, not where its earliest one
                # does: chaining firings a second apart made the Gunshot burst start at 8.25 for a
                # shot the annotator marked at 10.10 (docs/onset_timing.md). Off by default.
                if _start_rule() == "strongest" and e.confidence > bursts[-1][2]:
                    bursts[-1][0] = e.start
                bursts[-1][2] = max(bursts[-1][2], e.confidence)
            else:
                bursts.append([e.start, e.end, e.confidence,
                               {raw_labels[id(e)]} if raw_labels is not None and id(e) in raw_labels else set()])
        best = max(bursts, key=lambda b: b[2])
        out.append(AudioEvent(label, best[0], best[1], best[2],
                              spans=[(b[0], b[1]) for b in bursts], breaks=sorted(set(brk) | set(extra))))
    return sorted(out, key=lambda e: -e.confidence)


def _depth(label: str) -> int:
    return len(ancestors(label))


def _common_parent(a: str, b: str) -> str:
    """The deepest label both descend from (itself included), or "" if they share none."""
    chain_a = [a] + ancestors(a)
    chain_b = set([b] + ancestors(b))
    for x in chain_a:
        if x in chain_b:
            return x
    return ""


def choose_source(raw: List[AudioEvent], fam: str, burst: Tuple[float, float], explain: bool = False):
    """The most specific sound the detector really heard in one burst of a family (PICTURE_V3).

    Adam, 2026-09-24: pictures were drawn from the family tag -- a bus drawn as a car, a horn drawn
    as a train. The detector usually DID hear the specific sound; it was lost because only children
    above the display bar could become ``detail``, and the most confident member was often an
    ontology ancestor ("Rail transport") rather than a child. Panel of five, two rounds:

      * candidates: every raw firing of this family at or above the detector's own bar
        (AED_THRESHOLD) that overlaps the burst, and at least SOURCE_REL_FLOOR x the family's peak
        in it -- so a 0.05 "Reversing beeps" can still never turn a car into a reversing tractor;
      * a candidate that is an ancestor of another candidate is dropped (it says less);
      * ranked by ontology depth, then confidence;
      * two siblings within SOURCE_TIE_MARGIN of each other are not guessed between: their common
        parent is drawn (Ambulance / Police car / Fire engine -> Emergency vehicle). A wrong specific
        source is the error DHH users rank worst (Jain et al., CHI 2019);
      * nothing qualifies -> the family itself.
    Both numbers are fixed a priori (not tuned on the clips the rules were written from).
    """
    out = _choose_source(raw, fam, burst)
    if not explain:
        return out
    a, b = burst
    top = sorted(((e.confidence, e.label) for e in raw if canonical(e.label) == fam
                  and e.start <= b and e.end >= a), reverse=True)
    seen, cands = set(), []
    for c, lab in top:
        if lab not in seen:
            seen.add(lab)
            cands.append([lab, round(float(c), 3)])
    return out, cands[:4]


def _choose_source(raw: List[AudioEvent], fam: str, burst: Tuple[float, float]) -> str:
    import config
    floor = float(getattr(config, "SOURCE_REL_FLOOR", 0.5))
    margin = float(getattr(config, "SOURCE_TIE_MARGIN", 0.8))
    bar = float(getattr(config, "AED_THRESHOLD", 0.175))
    a, b = burst
    mine = [e for e in raw if canonical(e.label) == fam and e.start <= b and e.end >= a]
    if not mine:
        return fam
    peak = max(e.confidence for e in mine)
    cand = {}
    for e in mine:
        if e.confidence >= max(bar, floor * peak):
            cand[e.label] = max(cand.get(e.label, 0.0), e.confidence)
    cand.pop(fam, None)
    # never broader than the family itself: "Rail transport" is mapped onto Train and "Thunderstorm"
    # onto Thunder, but both say LESS than the family does
    cand = {x: c for x, c in cand.items() if not is_descendant(fam, x)}
    labels = [x for x in cand if not any(x != y and is_descendant(y, x) for y in cand)]
    if not labels:
        return fam
    labels.sort(key=lambda x: (-_depth(x), -cand[x]))
    top = labels[0]
    # The tie test sees EVERY sibling at the detector's bar, not only those that cleared the floor
    # (P1, round 3): with a loud family firing, the floor alone hid a Fire engine at 0.34 behind an
    # Ambulance at 0.40, and a 0.06 gap between two identical sirens decided the picture.
    allfirings = {}
    for e in mine:
        if e.confidence >= bar and e.label != fam and not is_descendant(fam, e.label):
            allfirings[e.label] = max(allfirings.get(e.label, 0.0), e.confidence)
    for other in sorted(allfirings, key=lambda x: -allfirings[x]):
        if other == top or _depth(other) != _depth(top):
            continue
        if allfirings[other] >= margin * cand[top] and not same_source(top, other):
            parent = _common_parent(top, other)
            # the fallback may never be broader than the family (Subway vs Railroad car share
            # "Rail transport", which says less than "Train")
            if not parent or parent == fam or is_descendant(fam, parent):
                return fam
            return parent
    return top


def label_names(label: str) -> list:
    """The plain words a label goes by: its comma parts, without AudioSet's bracketed disambiguation
    ("Ambulance (siren)" -> ambulance). Lower case."""
    return [p.strip().lower() for p in label.split("(")[0].split(",") if p.strip()]


def forbidden_names(source: str, fired) -> set:
    """PICTURE_SCENE guard: words a picture of `source` may not use, because the audio never
    established them -- the source's ontology siblings (and everything under them), and any kind
    under the source that the detector did not fire. A bare Siren may not become a police car; a tie
    between Police car and Ambulance may not be settled by a guess."""
    par = _parents()
    kids = {}
    for c, p in par.items():
        kids.setdefault(p, []).append(c)

    def below(x):
        out, todo, seen = [], list(kids.get(x, [])), {x}
        while todo:
            y = todo.pop()
            if y not in seen:
                seen.add(y)
                out.append(y)
                todo.extend(kids.get(y, []))
        return out
    fired = set(fired or [])
    bad = set()
    p = par.get(source)
    for s in (kids.get(p, []) if p else []):
        if s != source and s not in fired and not is_descendant(source, s):
            bad |= {s, *below(s)}
    bad |= {d for d in below(source) if d not in fired}
    # the single-parent map loses AudioSet's second parents ("Police car (siren)" sits under Emergency
    # vehicle only), so a label is also a kind of the source when its full name carries the source's
    # head word: "Police car (siren)", "Civil defense siren" are kinds of Siren
    head = (label_names(source) or [""])[0].split()[-1:] or [""]
    for x in par:
        words = set("".join(c if c.isalnum() else " " for c in x.lower()).split())
        if head[0] and head[0] in words and x != source and x not in fired and x not in ancestors(source):
            bad.add(x)
    allowed = {w for x in [source, *ancestors(source), *fired] for w in label_names(x)}
    return {w for x in bad for w in label_names(x)} - allowed


_STOP = {"a", "an", "the", "of", "or", "and", "in", "on", "with", "sound", "sounds", "noise"}


def _top(label: str) -> str:
    chain = [label] + ancestors(label)
    return chain[-1]


def other_branch_makers(words, source: str) -> list:
    """PICTURE_SCENE_GUARD2: qualifier words that name a sound class ONLY in other top-level branches of
    the ontology than the source's. "car" before "door" is in the source's own branch (Sounds of things)
    and passes; "cat" before "screaming" names an Animal where the source is a Human sound."""
    top = _top(source)
    by_word = {}
    for lab in _parents():
        for n in label_names(lab):
            if " " not in n:
                by_word.setdefault(n, set()).add(_top(lab))
    out = []
    for w in words:
        w = w.lower().rstrip("s") if w.lower() not in by_word else w.lower()
        tops = by_word.get(w)
        # only LIVING makers are refused (an animal or a person brought in from another branch -- the
        # invention risk); a material or natural qualifier is a kind ("steam train", "farm vehicle")
        if tops and top not in tops and tops & {"Animal", "Human sounds"}:
            out.append(w)
    return out


def names_forbidden(phrase: str, source: str, fired) -> list:
    """The forbidden words (forbidden_names, reduced to the words that tell them apart from what the
    audio did establish) that `phrase` uses. Mechanical: a list, not a question to a model."""
    def toks(s):
        return set("".join(c if c.isalnum() else " " for c in s.lower()).split())
    allowed = set()
    for x in [source, *ancestors(source), *(fired or [])]:
        allowed |= toks(" ".join(label_names(x)))
    got = toks(phrase)
    hits = []
    for n in forbidden_names(source, fired):
        distinct = toks(n) - allowed - _STOP
        if distinct and distinct <= got:
            hits.append(n)
    return sorted(hits)


def consolidate_families(events: List[AudioEvent],
                         threshold: float = 0.0) -> List[AudioEvent]:
    """Relabel sub-types to their canonical parent, then merge. One entry/source.

    The family is what the GATE reasons about -- visibility concepts are keyed on it,
    and one entry per source is what keeps the panel from filling with near-duplicates.
    But the family is far too coarse to DEPICT: an audit of 100 clips found 85 whose raw
    detections contained a more specific label than the one being drawn. "Fire engine,
    fire truck (siren)" collapsed to "Siren", whose query hint is "ambulance", so every
    fire truck in the benchmark was rendered as an ambulance; "Truck" and "Car" both
    became a generic "Vehicle"; "Waves, surf" became "Water".

    So the most confident specific child is preserved in ``detail`` and used for the
    image only. Gating behaviour is unchanged -- this affects what is drawn, never
    whether it is drawn.
    """
    # A firing below the display threshold can neither extend a span nor supply the
    # detail. It was never going to be shown on its own, so it must not be able to
    # change what IS shown -- "Reversing beeps" at 0.05 turned a rodeo Vehicle into a
    # reversing tractor.
    raw = list(events)            # PICTURE_V3 reads the unfiltered list for the drawn source only
    events = [e for e in events if e.confidence >= threshold]
    relabelled = [AudioEvent(canonical(e.label), e.start, e.end, e.confidence,
                             breaks=list(getattr(e, "breaks", None) or [])) for e in events]
    import config as _cfg
    rawlab = ({id(r): e.label for r, e in zip(relabelled, events)}
              if getattr(_cfg, "RETRIGGER_RAW", False) else None)
    merged = merge_by_label(relabelled, raw_labels=rawlab)
    best = {}
    for e in events:
        fam = canonical(e.label)
        if e.label == fam:
            continue                      # not more specific than its own family
        if fam not in best or e.confidence > best[fam].confidence:
            best[fam] = e
    for m in merged:
        child = best.get(m.label)
        if child is not None:
            m.detail = child.label
    import config as _c
    if getattr(_c, "PICTURE_V3", False):
        for m in merged:
            m.source = choose_source(raw, m.label, (m.start, m.end))
    return merged


# Specific labels that name the SOUND rather than its source, and so make a worse
# image than the family does: searching "Chirp" returns waveforms and logos, while
# "Bird" returns a bird. Kept as a short explicit list because no rule distinguishes
# these reliably -- an AudioSet name is a sound name, and only some happen to be objects.
DEPICTION_SKIP = {
    "Chirp", "Tweet", "Squawk", "Coo", "Caw", "Bird vocalization",
    "Rain on surface", "Vocalization", "Rustling", "Rumble", "Hum", "Hiss",
    "Whoosh", "Clatter", "Roar", "Buzz",
}


# The setting/disambiguation/phrasing tables that lived here are gone. They were an
# attempt to encode cross-modal reasoning as lookups, and they broke where fixed
# categories always break: a motorcycle POV shot scored 'vehicle interior' at 0.98,
# true of the camera and false of the scene, and every depiction for that clip came
# out 'indoors' over an outdoor street. The proposal specifies an LLM for this, and
# stage5/reason.py now does it from frames spanning each sound.

def depiction_query(label: str, detail: str = "") -> str:
    """What to search for or draw: the specific sound if one was detected, else the family.

    AudioSet names are written for annotators, not image search: they carry synonym
    lists ("Boat, Water vehicle") and disambiguating parentheticals ("Police car
    (siren)"). Both wreck a search query, so keep the leading name and drop the rest --
    which turns "Fire engine, fire truck (siren)" into "Fire engine" and "Ambulance
    (siren)" into "Ambulance", the two things a viewer would actually recognise.
    """
    chosen = detail or label
    if chosen in QUERY_HINTS:
        return QUERY_HINTS[chosen]
    name = chosen.split(",")[0]
    if "(" in name:
        name = name.split("(")[0]
    name = name.strip(" -")
    if not name or name in DEPICTION_SKIP:
        return QUERY_HINTS.get(label, label)      # the family depicts better
    return name


# ---------------------------------------------------------------------------------------------------------------
# SENSE ANCHOR (Adam, 28 Sept 2026): a label word is read in its ONTOLOGY sense, never its plain-English one. "Honk"
# is the goose's call (Honk > Goose > Fowl), not a car horn; "Tap" is a knock on a door (Tap > Door), not a water tap;
# "Bark" is a dog (Bark > Dog), not tree bark. For an ACTION label (a sound word that names no thing) the maker is the
# nearest ancestor that is a THING. Two fixed lists, read off the 346 drawable labels (benchmark/gold/
# depictable_vocab.json, 215 families and their children):
#   ACTION_LABELS    -- labels whose name is the sound or the action, not the thing that makes it;
#   NON_MAKERS       -- grouping labels that are no thing to draw (branches, "Onomatopoeia", "Domestic sounds"...),
#                       and "Arrow", the single-parent map's parent for Whoosh / Thump / Wobble (AudioSet lists
#                       Onomatopoeia as their other parent; an arrow is not what makes a thud).
ACTION_LABELS = {
    "Accelerating, revving, vroom", "Air horn, truck horn", "Applause", "Bang", "Bark", "Bay", "Beep, bleep",
    "Bellow", "Belly laugh", "Biting", "Bleat", "Boiling", "Booing", "Boom", "Bow-wow", "Breaking",
    "Burping, eructation", "Burst, pop", "Busy signal", "Buzz", "Caterwaul", "Caw", "Change ringing (campanology)",
    "Cheering", "Chewing, mastication", "Chink, clink", "Chirp, tweet", "Chop", "Chopping (food)", "Chorus effect",
    "Chuckle, chortle", "Clapping", "Clickety-clack", "Clip-clop", "Cluck", "Clunk", "Coo", "Cough", "Crack",
    "Crackle", "Creak", "Croak", "Crowing, cock-a-doodle-doo", "Crying, sobbing", "Dial tone", "Ding", "Ding-dong",
    "Drip", "Eruption", "Fart", "Fill (with liquid)", "Footsteps", "Frying (food)", "Fusillade", "Gargling", "Gasp",
    "Giggle", "Gobble", "Groan", "Growling", "Grunt", "Gurgling", "Gush", "Hiccup", "Hiss", "Honk", "Hoot", "Howl",
    "Howl (wind)", "Idling", "Knock", "Laughter", "Meow", "Moo", "Neigh, whinny", "Nicker", "Oink", "Pant", "Patter",
    "Pour", "Purr", "Quack", "Rattle", "Reversing beeps", "Ringing (of resonator)", "Ringtone", "Roar", "Run",
    "Sanding", "Sawing", "Screaming", "Shatter", "Shout", "Shuffle", "Sigh", "Sizzle", "Skidding", "Slam", "Slosh",
    "Smash, crash", "Snap", "Sneeze", "Snicker", "Sniff", "Snoring", "Snort", "Snort (horse)", "Sonic boom",
    "Splash, splatter", "Splinter", "Spray", "Squawk", "Squeak", "Squish", "Stir", "Stomach rumble", "Tap",
    "Throat clearing", "Thump, thud", "Thunk", "Tick", "Tick-tock", "Toot", "Train horn", "Train whistle",
    "Trickle, dribble", "Typing", "Vehicle horn, car horn, honking", "Wail, moan", "Walk, footsteps", "Wheeze",
    "Whimper", "Whoop", "Whoosh, swoosh, swish", "Wobble", "Wolf-whistling", "Writing", "Yawn", "Yell", "Yip",
    "Bird vocalization, bird call, bird song", "Whistling", "Heart murmur", "Otoacoustic emission",
    "Tinnitus, ringing in the ears", "Baby laughter", "Baby cry, infant cry", "Battle cry", "Children shouting",
}
NON_MAKERS = {
    "Animal", "Wild animals", "Domestic animals, pets", "Livestock, farm animals, working animals", "Human sounds",
    "Human voice", "Human group actions", "Human locomotion", "Respiratory sounds", "Digestive", "Breathing",
    "Sounds of things", "Natural sounds", "Source-ambiguous sounds", "Generic impact sounds", "Onomatopoeia",
    "Brief tone", "Clicking", "Other sourceless", "Miscellaneous sources", "Specific impact sounds",
    "Domestic sounds, home sounds", "Mechanisms", "Tools", "Liquid", "Music", "Musical instrument", "Explosion",
    "Sound equipment", "Heart sounds, heartbeat", "Arrow", "Channel, environment and background",
}
# how a maker label is drawn when its own name is a category-like plural or a material
MAKER_NAME = {"Roaring cats (lions, tigers)": "lion", "Rodents, rats, mice": "mouse", "Cattle, bovinae": "cow",
              "Wood": "piece of wood", "Fly, housefly": "housefly", "Canidae, dogs, wolves": "dog",
              "Chicken, rooster": "chicken", "Pigeon, dove": "pigeon", "Bee, wasp, etc.": "bee",
              "Boat, Water vehicle": "boat", "Motor vehicle (road)": "car", "Snake": "snake",
              "Steam": "jet of steam", "Fire": "fire"}


def is_action(label: str) -> bool:
    return label in ACTION_LABELS


# The official ontology keeps EVERY parent of a label (38 labels have several: Hiss < Cat, Snake, Steam, Onomatopoeia).
# ancestors() / is_descendant() above read the single-parent map and stay as they are (the scoring and the family rule
# use them); the maker decision reads all parents from the vendored src/audioset_ontology.json.
_ONTO_PARENTS = None
_ONTO_ABSTRACT = None
# a parent that is a group of sounds, not a thing that makes one (in addition to NON_MAKERS and the ontology's own
# "abstract" flag): an alarm is what a horn or a doorbell is FOR, not what makes it
GROUP_PARENTS = {"Alarm", "Whistle", "Engine", "Siren"}


def ontology_parents(label: str) -> list:
    """All parents of a label in the official AudioSet ontology (src/audioset_ontology.json); falls back to the
    single-parent map when the file is missing."""
    global _ONTO_PARENTS, _ONTO_ABSTRACT
    if _ONTO_PARENTS is None:
        import json
        from pathlib import Path
        f = Path(__file__).resolve().parent / "audioset_ontology.json"
        _ONTO_PARENTS, _ONTO_ABSTRACT = {}, set()
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            byid = {x["id"]: x["name"] for x in d}
            for x in d:
                if "abstract" in (x.get("restrictions") or []):
                    _ONTO_ABSTRACT.add(x["name"])
                for c in x.get("child_ids") or []:
                    _ONTO_PARENTS.setdefault(byid[c], []).append(x["name"])
    if label in _ONTO_PARENTS:
        return list(_ONTO_PARENTS[label])
    p = _parents().get(label)
    return [p] if p else []


def _not_a_maker(x: str) -> bool:
    ontology_parents(x)
    return x in NON_MAKERS or x in ACTION_LABELS or x in (_ONTO_ABSTRACT or set())


def ontology_senses(label: str) -> list:
    """The label's senses as makers, one per parent branch, in ontology order: the nearest THING up each parent
    (several parents -> several makers: Hiss -> Cat, Snake, Steam), and None for a branch that reaches no thing (an
    abstract group such as Onomatopoeia, Brief tone, Generic impact sounds, Clicking: the label also has a generic,
    source-less sense). A label that names a thing itself is its own single sense. Duplicates removed."""
    if not _not_a_maker(label):
        return [label]
    out = []

    def up(x, seen):
        for p in ontology_parents(x):
            if p in seen:
                continue
            if p in GROUP_PARENTS:
                continue                  # what the sound is FOR (a horn is an alarm), neither a maker nor a sense
            elif _not_a_maker(p):
                before = len(out)
                up(p, seen | {p})
                if len(out) == before:
                    out.append(None)
            else:
                out.append(p)
    up(label, {label})
    res = []
    for m in out:
        if m not in res:
            res.append(m)
    return res


def ontology_maker(label: str):
    """The sense anchor: the ONE thing that makes this sound in every sense the ontology gives it, else None (no
    thing at all -- Bang, Thump; several things -- Hiss, Growling; or a thing plus a source-less sense -- Rattle,
    Crack, Tap). A human sound has no thing maker (the maker is a person)."""
    s = ontology_senses(label)
    return s[0] if len(s) == 1 and s[0] is not None else None


def maker_name(label: str) -> str:
    return MAKER_NAME.get(label) or (label_names(label) or [label.lower()])[0]


def maker_words(label: str) -> set:
    """Words that name this maker in a subject (every name of the label, its drawing name, singular and plural)."""
    out = set()
    for n in label_names(label) + [maker_name(label)] + [MAKER_NAME.get(label, "")]:
        for w in n.split():
            if len(w) > 2:
                out |= {w, w.rstrip("s"), w + "s"}
    return out - {"motor", "piece", "wood", "water"} | ({"wood", "wooden", "plank", "branch", "log"}
                                                        if label == "Wood" else set())


def _thing_chain(label: str) -> set:
    return {x for x in [label] + ancestors(label) if x not in NON_MAKERS and x not in ACTION_LABELS}


def sense_consistent(maker_label, sound_label: str) -> bool:
    """A maker is consistent with the sound's ontology sense when they share a THING in their chains (a bus and a
    Toot share Motor vehicle; a car and a Honk share nothing). A person (None) is consistent with a human sound."""
    if maker_label is None:
        return "Human sounds" in ancestors(sound_label) or sound_label == "Human sounds"
    if maker_label == sound_label or is_descendant(maker_label, sound_label):      # a kind of it (Typing > keyboard)
        return True
    if maker_label in ontology_senses(sound_label):                                 # one of its parents' senses
        return True
    return bool(_thing_chain(maker_label) & _thing_chain(sound_label))
