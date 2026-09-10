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
    "Whispering", "Shout", "Yell", "Children shouting", "Screaming",
}
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


def is_salient_nonspeech(label: str) -> bool:
    """True for a discrete non-speech sound worth (potentially) visualizing."""
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


def merge_by_label(events: List[AudioEvent]) -> List[AudioEvent]:
    """Collapse repeated firings of the same label into one span (min start,
    max end, max confidence). Sorted by confidence descending."""
    by_label = {}
    for e in events:
        if e.label in by_label:
            m = by_label[e.label]
            m.start = min(m.start, e.start)
            m.end = max(m.end, e.end)
            m.confidence = max(m.confidence, e.confidence)
        else:
            by_label[e.label] = AudioEvent(e.label, e.start, e.end, e.confidence)
    return sorted(by_label.values(), key=lambda e: -e.confidence)


def consolidate_families(events: List[AudioEvent]) -> List[AudioEvent]:
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
    relabelled = [AudioEvent(canonical(e.label), e.start, e.end, e.confidence) for e in events]
    merged = merge_by_label(relabelled)
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


# ----------------------------------------------------------------------
# Cross-modal depiction: what the SETTING says an unseen sound is
# ----------------------------------------------------------------------
# The gate asks whether a sound's source is on screen. For the sounds this system
# exists to depict the answer is always no -- and the video still constrains what the
# sound plausibly IS. A crackle in a living room is a fire; the same crackle beside a
# forest stream is water, and PANNs confuses the two routinely because they are
# spectrally similar. Depicting from the audio label alone throws that away.

# Audio labels that are acoustically ambiguous, and which label each setting group
# favours. Deliberately short: every entry is a confusion that actually occurs and that
# the setting genuinely resolves, not a guess about what might.
# Keyed on the SPECIFIC setting, not the coarse group. Grouping was wrong and
# dangerously so: "home" covers kitchens and bathrooms, where running water is far more
# likely than a hearth, and the group-level rule would have turned every indoor tap into
# a fireplace. A confusion is only resolvable where the scene genuinely resolves it.
AMBIGUOUS = {
    ("Water", "living room"): "Fire",     # a crackle in a lounge is a hearth, not a tap
    ("Fire", "river"): "Water",           # a rush beside a stream is water, not flames
    ("Fire", "forest"): "Water",
    ("Fire", "sea"): "Water",
    ("Rain", "kitchen"): "Frying",        # a patter on a stove is a frying pan
    ("Applause", "forest"): "Rain",       # a patter among trees is rain on leaves
    ("Applause", "field"): "Rain",
}


def min_confidence(label: str, default: float) -> float:
    return max(default, NOISE_LIKE_MIN_CONF.get(label, 0.0))


def canonical(label: str) -> str:
    return FAMILY.get(label, label)


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


def merge_by_label(events: List[AudioEvent]) -> List[AudioEvent]:
    """Collapse repeated firings of the same label into one span (min start,
    max end, max confidence). Sorted by confidence descending."""
    by_label = {}
    for e in events:
        if e.label in by_label:
            m = by_label[e.label]
            m.start = min(m.start, e.start)
            m.end = max(m.end, e.end)
            m.confidence = max(m.confidence, e.confidence)
        else:
            by_label[e.label] = AudioEvent(e.label, e.start, e.end, e.confidence)
    return sorted(by_label.values(), key=lambda e: -e.confidence)


def consolidate_families(events: List[AudioEvent]) -> List[AudioEvent]:
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
    relabelled = [AudioEvent(canonical(e.label), e.start, e.end, e.confidence) for e in events]
    merged = merge_by_label(relabelled)
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

# How to SAY a sound once the setting is known. Keys are (family, setting group).
# The value replaces the bare label in the generation prompt, which is the whole point:
# "a stream flowing through a forest" is depictable, "Water" is not.
IN_SETTING = {
    ("Water", "home"): "water running from a kitchen tap",
    ("Water", "nature"): "a stream flowing through a forest",
    ("Water", "urban"): "water running in a gutter on a street",
    ("Fire", "home"): "a fire crackling in a fireplace",
    ("Fire", "nature"): "a campfire burning outdoors",
    ("Bird", "nature"): "a bird singing in a tree",
    ("Bird", "urban"): "a pigeon on a city street",
    ("Vehicle", "urban"): "a car driving along a city street",
    ("Vehicle", "nature"): "a car on a country road",
    ("Siren", "urban"): "an emergency vehicle with its siren on in a street",
    ("Rain", "urban"): "rain falling on a city street",
    ("Rain", "nature"): "rain falling in a forest",
    ("Wind", "nature"): "wind blowing through trees",
    ("Footsteps", "home"): "footsteps on a wooden floor indoors",
    ("Footsteps", "urban"): "footsteps on a pavement",
    ("Footsteps", "nature"): "footsteps on a forest path",
    ("Crowd", "urban"): "a crowd of people in a busy street",
    ("Crowd", "indoor"): "a crowd of people inside a hall",
    ("Dishes", "home"): "dishes being washed in a kitchen sink",
    ("Door", "home"): "a door closing inside a house",
    ("Frying", "home"): "food frying in a pan on a stove",
}

# Fallback phrasing when the pair is not in the table: place the subject in the scene.
SETTING_PHRASE = {
    "home": "inside a home", "indoor": "indoors", "urban": "on a city street",
    "nature": "outdoors in nature", "work": "at a work site",
}


# ----------------------------------------------------------------------
# What the VIEWER needs to know, which is not the same as the audio taxonomy
# ----------------------------------------------------------------------
# AudioSet separates Laughter, Snicker, Chuckle, Giggle and Belly laugh. A deaf viewer
# needs one fact -- people are laughing -- and a panel that spends three of its three
# slots on three photographs of laughing people has told them nothing extra while
# crowding out the siren they could not hear. The families are correct and the panel
# was still wrong, because the taxonomy is not the goal.
#
# So families are collapsed a second time, by what a viewer would take away. Anything
# absent from this table keeps its own meaning and its own slot.
MEANING_OF = {
    "Laughter": "people laughing", "Snicker": "people laughing",
    "Chuckle, chortle": "people laughing", "Belly laugh": "people laughing",
    "Giggle": "people laughing", "Baby laughter": "a baby laughing",
    "Crowd": "a crowd of people", "Applause": "an audience clapping",
    "Cheering": "an audience cheering",
    "Water": "water", "Stream": "water", "Waves, surf": "water", "Ocean": "water",
    "Rain": "rain", "Raindrop": "rain", "Thunderstorm": "thunder and lightning",
    "Thunder": "thunder and lightning",
    "Vehicle": "a vehicle", "Traffic noise, roadway nois": "traffic",
    "Siren": "an emergency vehicle siren",
    "Footsteps": "someone walking", "Walk, footsteps": "someone walking",
    "Typing": "someone typing", "Computer keyboard": "someone typing",
    "Sawing": "a power tool", "Power tool": "a power tool", "Tools": "a power tool",
    "Chainsaw": "a chainsaw",
    "Hum": "a machine humming", "Mains hum": "a machine humming",
    "Burst, pop": "a sudden bang", "Explosion": "an explosion", "Gunshot": "a gunshot",
    "Bird": "a bird", "Dog": "a dog", "Cat": "a cat",
}


def viewer_meaning(label: str) -> str:
    """The one thing a deaf viewer should take from this sound."""
    return MEANING_OF.get(label, "")


def dedupe_by_meaning(items):
    """Keep one entry per distinct meaning, highest confidence first.

    ``items`` is a list of (label, confidence, payload); returns the surviving payloads
    in the original confidence order. Two sounds that would produce the same picture
    are the same sound as far as the panel is concerned.
    """
    seen, out = set(), []
    for label, _conf, payload in sorted(items, key=lambda t: -t[1]):
        key = viewer_meaning(label) or label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def disambiguate(label: str, setting: str) -> str:
    """Correct an acoustically ambiguous label using the setting, or leave it alone.

    Conservative by construction: only the seven (label, setting) pairs above can change
    anything, and every other detection passes through untouched. A confident audio
    detection is never overridden by a guess about the scene -- the rule fires only
    where the two sounds are genuinely hard to tell apart AND the setting decides it.
    """
    if not setting:
        return label
    return AMBIGUOUS.get((label, setting), label)


def contextual_subject(label: str, detail: str = "", setting: str = "",
                       setting_group: str = "") -> str:
    """What to draw, given both the sound and the scene it happens in."""
    base = viewer_meaning(label) or depiction_query(label, detail)
    if not setting_group:
        return base
    phrase = IN_SETTING.get((label, setting_group))
    if phrase:
        return phrase
    where = SETTING_PHRASE.get(setting_group)
    return f"{base} {where}" if where else base


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
