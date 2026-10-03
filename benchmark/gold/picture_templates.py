"""Picture templates of the final system: the fixed prompt parts, subjects and seeds the picture step uses.

RULES_TAIL (the prompt rules), TEMPLATES (a fixed subject for a sound with a canonical maker), TEMPLATE_NEG (words
added to the negative prompt), CARDS (a comic burst card with a word, never generated), seed_of (one seed per sound),
negative_for, burst_card and ink (the blank-picture guard). Imported by src/stage6_visual_augmentation/ and
comfyui_nodes/run_frozen.py.

(design record: release v1.2.0)
"""
from __future__ import annotations

import zlib

import numpy as np

NEGATIVE_V3 = ["scenery", "landscape", "background scene", "room interior", "street", "buildings",
               "sky background", "text", "letters", "watermark"]
# GP-4 3(b) image-prompt rules (G2): the whole thing, mid-act with its visible effect, one large subject
RULES_TAIL = (", the whole thing fully in frame, caught at the moment it makes the sound with its visible "
              "effect, one large subject filling the picture, plain white background")
# group (b): sounds with a canonical maker get a fixed subject per label (G2, G1; design review GP-4)
TEMPLATES = {
    "Thunder": "one large lightning bolt striking down from a dark storm cloud",
    "Thunderstorm": "one large lightning bolt striking down from a dark storm cloud",
    # reworded 28 Sept 2026 (picture check): "heavy rain drops splashing on a window pane" drew a single water-crown
    # splash, which the checker (and a viewer) reads as a splash, not rain -- both good pictures it rejected
    "Rain on surface": "heavy rain pouring down in long falling streaks onto an open umbrella",
    "Rain": "heavy rain pouring down in long falling streaks onto an open umbrella",
    # reworded after the by-eye check (GP-4, before the freeze): "hazard lights flashing" drew a police light bar
    "Car alarm": "an ordinary parked car seen from the front, its headlights and indicator lights flashing",
    "Train horn": "the front of a whole locomotive blowing its horn",
    "Church bell": "a large church bell swinging in its tower",
    "Shatter": "a glass window shattering with sharp pieces flying",
    "Smash, crash": "a glass window shattering with sharp pieces flying",
}
# words a template must never draw, added to the negative prompt (generators ignore "no ..." in a prompt):
# the car alarm drew a police light bar, then roof beacons, at the by-eye check (GP-4, before the freeze)
TEMPLATE_NEG = {"Car alarm": "roof light, light bar, beacon, police car, emergency vehicle, siren",
                "Rain": "splash crown, water crown, single droplet, splash close-up",
                "Rain on surface": "splash crown, water crown, single droplet, splash close-up"}
# group (c): no maker the audio established -> a fixed comic burst card with the word, never generated
CARDS = {"Whoosh, swoosh, swish": "WHOOSH", "Thunk": "THUD", "Thump, thud": "THUD", "Bang": "BANG",
         "Slap, smack": "SMACK", "Whack, thwack": "WHACK"}


def seed_of(item) -> int:                                   # benchmark/gold/picture_bench.py (release v1.2.0), unchanged
    return zlib.crc32(f"{item['clip']}|{item['label']}|{item['start']:.2f}".encode()) & 0x7FFFFFFF


def negative_for(subject: str) -> str:                      # stage6.negative_for, unchanged
    words = {w.strip(",.").lower() for w in subject.split()}
    weather = words & {"cloud", "clouds", "lightning", "rain", "raining", "storm", "thunder",
                       "thunderstorm", "wind", "snow", "hail"}
    keep = [n for n in NEGATIVE_V3 if not (set(n.split()) & words)
            and not (weather and n in ("sky background", "landscape"))]
    return ", ".join(keep)


def ink(img) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return float((a.min(axis=2) < 238).mean())


def burst_card(word, fit=False):
    """A fixed comic burst with the word inside it (drawn once per word, identical every time).
    fit=True (PICTURE_VERIFY word card, a sound's name of any length): the font shrinks and the name wraps onto up
    to two lines so it stays inside the burst; the default path is the frozen card, unchanged."""
    if fit:
        return _burst_card_fit(word)
    import math
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1024, 1024), "white")
    d = ImageDraw.Draw(img)
    pts = []
    for k in range(24):
        r = 470 if k % 2 == 0 else 330
        a = 2 * math.pi * k / 24
        pts.append((512 + r * math.cos(a), 512 + r * math.sin(a)))
    d.polygon(pts, fill=(255, 214, 64), outline=(30, 30, 30), width=10)
    size = 170 if len(word) <= 5 else 140
    font = None
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            font = ImageFont.truetype(f, size)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default(size=size)
    x0, y0, x1, y1 = d.textbbox((0, 0), word, font=font)
    d.text((512 - (x1 - x0) / 2 - x0, 512 - (y1 - y0) / 2 - y0), word, fill=(20, 20, 20), font=font)
    return img


def _font(size):
    from PIL import ImageFont
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _burst_card_fit(word):
    """burst_card's burst with a name of any length: one line if it fits at >= 90 px, else two balanced lines."""
    from PIL import ImageDraw
    img = burst_card("")
    d = ImageDraw.Draw(img)
    words = word.split()
    splits = [[word]] + ([[" ".join(words[:k]), " ".join(words[k:])] for k in range(1, len(words))]
                         if len(words) > 1 else [])
    best = None
    for lines in splits:
        for size in range(170, 49, -10):
            font = _font(size)
            boxes = [d.textbbox((0, 0), ln, font=font) for ln in lines]
            w = max(b[2] - b[0] for b in boxes)
            h = sum(b[3] - b[1] for b in boxes) + 20 * (len(lines) - 1)
            if w <= 600 and h <= 420:
                if best is None or size > best[0] + (10 if len(lines) > len(best[1]) else 0):
                    best = (size, lines)
                break
    size, lines = best or (50, [word])
    font = _font(size)
    boxes = [d.textbbox((0, 0), ln, font=font) for ln in lines]
    total = sum(b[3] - b[1] for b in boxes) + 20 * (len(lines) - 1)
    y = 512 - total / 2
    for ln, (x0, y0, x1, y1) in zip(lines, boxes):
        d.text((512 - (x1 - x0) / 2 - x0, y - y0), ln, fill=(20, 20, 20), font=font)
        y += (y1 - y0) + 20
    return img
