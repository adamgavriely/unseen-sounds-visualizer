"""GP-4 screening: draw the picture-DEV sounds with each candidate generator, same text, same seed rule
(docs/history/panels/panel_2026-09-25_scene_prompt.md and panel_2026-09-25_generator.md, release v1.2.0). Runs in ~/venvs/gen
(torch 2.7, diffusers from main), not in msproj, so it imports nothing from the pipeline.

    gate:   python benchmark/gold/gen_screen.py --model q21 --gate
            (five fixed sounds; seconds per picture, peak VRAM, negative prompt used or not -> gate json)
    draw:   python benchmark/gold/gen_screen.py --model q21 --bench data/work/picture_bench_fresh \
                --subjects V31 --arm G_q21
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
PLAIN_TAIL = ", plain white background, clearly visible"          # stage6.plain_prompt, unchanged
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
RGBA_WRAP = ("This is an RGBA image with transparency. {subject}. The image has alpha channel and the "
             "background is transparent.")                           # the Qwen-Image-2.1 card's own format

MODELS = {
    # key: (repo, pipeline class, call kwargs, uses negative prompt)
    "q2512": ("Qwen/Qwen-Image-2512", "QwenImagePipeline",
              {"num_inference_steps": 50, "true_cfg_scale": 4.0}, True),
    "q21": ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline", {"num_inference_steps": 40}, False),
    "q21rgba": ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline", {"num_inference_steps": 40}, False),
}


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


def load(model_key):
    import torch
    import diffusers
    repo, cls, _, _ = MODELS[model_key]
    pipe = getattr(diffusers, cls).from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


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


def draw(pipe, model_key, subject, seed, tail=PLAIN_TAIL, long_prompt="", extra_neg=""):
    import torch
    from PIL import Image
    _, _, kw, uses_neg = MODELS[model_key]
    kw = dict(kw, width=1024, height=1024, generator=torch.Generator("cuda").manual_seed(seed))
    if model_key == "q21rgba":
        prompt = RGBA_WRAP.format(subject=subject)
    elif long_prompt:
        prompt = long_prompt
    else:
        prompt = subject + tail
    if uses_neg:
        kw["negative_prompt"] = ", ".join(x for x in (negative_for(subject), extra_neg) if x) or " "
    img = pipe(prompt=prompt, **kw).images[0]
    if img.mode == "RGBA":                     # the viewer sees it on white
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("RGB"), prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--bench", default="data/work/picture_bench_fresh")
    ap.add_argument("--subjects", default="V31")
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed-offset", type=int, default=1)      # GP-4: seed_of(item) + 1 for every arm
    ap.add_argument("--tail", choices=["plain", "rules"], default="plain")
    ap.add_argument("--long", action="store_true", help="use the subjects file's guarded 'long' prompt")
    ap.add_argument("--templates", action="store_true", help="group (b) templates and group (c) cards")
    ap.add_argument("--only", default="", help="comma list of item ids (e.g. the template sounds)")
    a = ap.parse_args()
    import torch
    bench = _ROOT / a.bench
    specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
    subj = json.loads((bench / f"subjects_{a.subjects}.json").read_text(encoding="utf-8"))
    if a.gate:                                 # five fixed sounds: the first five by seed
        specs = sorted(specs, key=seed_of)[:5]
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        specs = [s for s in specs if s["i"] in keep]
    tail = RULES_TAIL if a.tail == "rules" else PLAIN_TAIL
    out = bench / (a.arm or f"gate_{a.model}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = load(a.model)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    manifest, times = [], []
    for it in specs:
        s = subj.get(str(it["i"]), {})
        subject = s.get("subject") or it["subject"]
        source = s.get("source") or it["label"]
        long_prompt = s.get("long", "") if a.long else ""
        seed = seed_of(it) + a.seed_offset
        t = time.time()
        if a.templates and (source in CARDS or it["label"] in CARDS):
            word = CARDS.get(source) or CARDS.get(it["label"])
            img, prompt = burst_card(word), "CARD:" + word
        else:
            extra_neg = ""
            if a.templates and (source in TEMPLATES or it["label"] in TEMPLATES):
                key = source if source in TEMPLATES else it["label"]
                subject = TEMPLATES[key]
                extra_neg = TEMPLATE_NEG.get(key, "")
                long_prompt = ""
            img, prompt = draw(pipe, a.model, subject, seed, tail, long_prompt, extra_neg)
            if ink(img) < 0.05:                # the blank guard, as in the pipeline: redraw once
                img, prompt = draw(pipe, a.model, subject, seed + 1, tail, long_prompt, extra_neg)
        times.append(time.time() - t)
        img.save(out / f"{it['i']:02d}.png")
        manifest.append({"i": it["i"], "subject": subject, "prompt": prompt, "seed": seed,
                         "ink": round(ink(img), 4), "seconds": round(times[-1], 1)})
        print(f"  {a.model} {it['i']:2d} {times[-1]:5.1f}s {subject}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    gate = {"model": a.model, "repo": MODELS[a.model][0], "load_seconds": round(load_s, 1),
            "seconds_per_picture_median": round(float(np.median(times)), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            "negative_prompt_used": MODELS[a.model][3], "gpu": torch.cuda.get_device_name(0),
            "pass_20s": float(np.median(times)) <= 20.0}
    print(json.dumps(gate))
    if a.gate:
        (_ROOT / "benchmark/gold/pictures" / f"gate_{a.model}.json").write_text(json.dumps(gate, indent=1),
                                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
