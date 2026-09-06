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
    # instruments (a visible musician) -> canonical Saxophone concept
    "Brass instrument": "Saxophone", "Wind instrument, woodwind instrument": "Saxophone",
    "Saxophone": "Saxophone",
}


# Wide-band, low-structure textures that PANNs routinely reports on noisy recordings
# (mic hiss read as rain, room rumble read as traffic). They need more evidence than a
# structured sound like a bark or a siren before we act on them. See notes sec:annotation.
NOISE_LIKE_MIN_CONF = {"Rain": 0.30, "Vehicle": 0.20, "Water": 0.25, "Thunder": 0.25}


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
    """Relabel sub-types to their canonical parent, then merge. One entry/source."""
    relabelled = [AudioEvent(canonical(e.label), e.start, e.end, e.confidence) for e in events]
    return merge_by_label(relabelled)
