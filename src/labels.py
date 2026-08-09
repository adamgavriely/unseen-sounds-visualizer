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
}
# Acoustic-environment / ambience descriptors: real, but not a discrete source to depict.
SCENE_LABELS = {
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural", "Reverberation", "Echo",
    "Noise", "Environmental noise", "Background noise", "White noise", "Pink noise",
    "Silence", "Field recording", "Sound effect", "Wind noise (microphone)",
}
# Music: flagged; non-salient for now (decision: handle diegetic music later, task A6).
MUSIC_LABELS = {
    "Music", "Scary music", "Background music", "Musical instrument", "Soundtrack music",
    "Theme music", "Sad music", "Happy music", "Exciting music",
}


# Generic superclasses: too vague to visualize well, and usually redundant with a
# specific sibling (PANNs fires "Animal" + "Domestic animals" alongside "Dog").
GENERIC_LABELS = {
    "Animal", "Domestic animals, pets", "Wild animals",
    "Livestock, farm animals, working animals", "Sounds of things", "Mechanisms",
    "Human sounds", "Domestic sounds, home sounds", "Human group actions",
    "Generic impact sounds", "Surface contact", "Onomatopoeia",
}


def is_music(label: str) -> bool:
    return label in MUSIC_LABELS or label.endswith("music")


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
    # water
    "Waves, surf": "Water", "Ocean": "Water", "Waterfall": "Water", "Stream": "Water",
    "Gurgling": "Water", "Slosh": "Water", "Trickle, dribble": "Water", "Drip": "Water",
    # thunder / storm
    "Thunderstorm": "Thunder", "Rumble": "Thunder",
    # rain
    "Raindrop": "Rain", "Rain on surface": "Rain",
    # bird
    "Bird vocalization, bird call, bird song": "Bird", "Chirp, tweet": "Bird", "Pigeon, dove": "Bird",
    # crowd
    "Cheering": "Crowd", "Applause": "Crowd", "Children shouting": "Crowd", "Hubbub": "Crowd",
}


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
