"""Try prompt variants side by side, with the model and full prompt printed under every
image, and a contact sheet per sound.

Rounds so far, each settled only by opening the PNGs:
  1  "minimal line art" -> literal black stripes behind the subject
  2  four styles: the 3d-icon look was the only one that stayed on a plain white ground
     for both an animal and a vehicle; the realistic ones put birds on leafy branches
  3  realistic + "no branches, no leaves" -> still branches (SDXL-Turbo runs at guidance
     0 and therefore ignores the negative prompt AND handles negation poorly), and a
     generic "with sound waves" cue produced a fire engine engulfed in FLAMES
  4  this one: the 3d-icon base with muted colour, and a cue written SEPARATELY FOR EACH
     SOUND -- lines from a dog's mouth, notes at a bird's beak, a siren on a truck roof.
     A generic cue asks the model to invent an idiom; a specific one names the picture.

Usage (on the cluster):
    python scripts/icon_grid.py
"""
from __future__ import annotations

import argparse
import gc
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw

MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxlturbo": "stabilityai/sdxl-turbo",
}

# (subject, cue) -- the cue names the picture wanted rather than asking for "sound"
SUBJECTS = [
    ("a dog barking",
     "with small curved sound lines coming out of its open mouth"),
    ("a bird chirping",
     "with a few small music notes next to its open beak"),
    ("a fire engine with its siren on",
     "with a siren light on the roof and small curved sound lines around the siren"),
    ("rain falling",
     "with falling raindrops and small splash marks below"),
]

# Muted, not candy-bright: Adam's note on round 2 was that the colours were too loud.
BASE3D = ("cute simple 3d icon of {s} {cue}, soft rounded shapes, soft muted natural "
          "colours, plain white background, centered, friendly and clear, no text")

VARIANTS = {
    "V1_3d_cue": BASE3D,
    "V2_3d_cue_strong": ("cute simple 3d icon of {s}, {cue}, the sound lines clearly "
                         "visible, soft rounded shapes, soft muted natural colours, "
                         "plain white background, centered, no text"),
    "V3_semi_cue": ("a simple semi-realistic illustration of {s} {cue}, soft muted "
                    "natural colours, plain white background, centered, clean and "
                    "uncluttered, no text"),
    "V4_3d_nocue": ("cute simple 3d icon of {s}, soft rounded shapes, soft muted natural "
                    "colours, plain white background, centered, friendly and clear, "
                    "no text"),
}

NEGATIVE = ("photograph, photorealistic, text, letters, words, watermark, logo, stripes, "
            "grid, frame, border, pattern, scenery, background objects, branches, leaves, "
            "grass, smoke, fire, clutter, multiple objects, blurry")


def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label_below(img: Image.Image, title: str, prompt: str, width: int = 512):
    img = img.resize((width, width), Image.LANCZOS)
    lines = textwrap.wrap(prompt, width=62)[:6]
    strip = 26 + 15 * (len(lines) + 1)
    out = Image.new("RGB", (width, width + strip), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, width, width, width + strip], fill=(28, 30, 38))
    d.text((8, width + 6), title, font=_font(14), fill=(255, 214, 102))
    for i, ln in enumerate(lines):
        d.text((8, width + 24 + 15 * i), ln, font=_font(12), fill=(226, 228, 235))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["sdxlturbo"])
    ap.add_argument("--out", default="data/output/icon_grid")
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image
    out = _ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cells = {name: [] for name, _ in SUBJECTS}

    for mkey in args.models:
        repo = MODELS[mkey]
        print(f"[{mkey}] loading {repo}", flush=True)
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                repo, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
            pipe.set_progress_bar_config(disable=True)
        except Exception as e:
            print(f"  ! {mkey} failed to load: {type(e).__name__}: {e}", flush=True)
            continue
        turbo = "turbo" in mkey
        for vkey, tmpl in VARIANTS.items():
            for subj, cue in SUBJECTS:
                prompt = tmpl.format(s=subj, cue=cue)
                try:
                    kw = dict(prompt=prompt, width=512, height=512)
                    if turbo:
                        kw.update(num_inference_steps=4, guidance_scale=0.0)
                    else:
                        kw.update(num_inference_steps=28, guidance_scale=7.5,
                                  negative_prompt=NEGATIVE)
                    img = pipe(**kw).images[0]
                except Exception as e:
                    print(f"  ! {mkey}/{vkey}/{subj}: {type(e).__name__}: {e}", flush=True)
                    continue
                cell = label_below(img, f"{mkey} | {vkey} | {subj}", prompt)
                cell.save(out / f"{mkey}_{vkey}_{subj.replace(' ', '_')[:28]}.png")
                cells[subj].append(cell)
                print(f"  {mkey:10} {vkey:18} {subj}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    for subj, imgs in cells.items():
        if not imgs:
            continue
        cols = len(VARIANTS)
        rows = (len(imgs) + cols - 1) // cols
        cw, ch = imgs[0].size
        sheet = Image.new("RGB", (cols * cw, rows * ch), (255, 255, 255))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))
        name = f"SHEET_{subj.replace(' ', '_')[:28]}.png"
        sheet.save(out / name)
        print(f"sheet -> {name} ({len(imgs)} cells)", flush=True)


if __name__ == "__main__":
    main()
