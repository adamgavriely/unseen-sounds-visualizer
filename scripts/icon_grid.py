"""Try several generators and several prompts side by side, with the prompt printed
under every image.

Four prompt rewrites were spent one at a time, each needing a human to open the PNG to
find out what went wrong -- stripes, then a badge on grey, then papercut folk art. That
is the wrong loop. This runs the whole cross-product in one job and labels every result
with the model and the exact prompt that produced it, so the choice is made by looking
at a contact sheet instead of by guessing the next adjective.

Simpler models are included deliberately: SDXL is the strongest photographic generator
here and that is precisely its problem -- asked for a pictogram it produces an ornate
illustration. A smaller, older model may be worse at photographs and better at flat
shapes, which is what the panel needs.

Usage (on the cluster):
    python scripts/icon_grid.py --subjects Bird "Fire engine" Dog
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

# Round 2. Round 1 chased minimalism -- silhouettes, line art, two-colour icons -- and
# Adam's reaction was that he wants a simple PICTURE, clipart or realistic clipart, not
# a drawing to colour in. So every candidate here is in full colour and depicts the
# thing recognisably; they differ in how stylised that colour is.
# Round 3. Style settled on the 3d-icon look (round 2): it was the only candidate that
# stayed on a plain white ground for an animal AND a vehicle -- the others put birds on
# leafy branches, which is background the viewer has to look past.
#
# The open question is the sound cue. A code-drawn arc symbol is identical every time
# and free; asking the generator for the cue puts it AT the mouth and looks like part of
# the picture, but only if the model complies. N4 is the control with no cue asked for.
BASE = ("cute simple 3d icon of {s}{cue}, soft rounded shapes, bright colours, plain "
        "white background, centered, friendly and clear, no text")

# Round 3. Style settled on the 3d-icon look from round 2: the only candidate that
# stayed on a plain white ground for an animal AND a vehicle -- the others put birds on
# leafy branches, which is background the viewer must look past.
#
# The open question is the sound cue. Code-drawn arcs are identical every time and
# free; asking the generator for the cue puts it at the mouth and looks like part of
# the picture, but only if the model complies. N4 is the control, no cue requested.
BASE = ("cute simple 3d icon of {s}CUE, soft rounded shapes, bright colours, plain "
        "white background, centered, friendly and clear, no text")

PROMPTS = {
    "N1_notes":   BASE.replace("CUE", " with small music notes nearby"),
    "N2_waves":   BASE.replace("CUE", " with sound waves coming from its mouth"),
    "N3_singing": BASE.replace("CUE", ", mouth open, with musical notes floating beside it"),
    "N4_nocue":   BASE.replace("CUE", ""),
}

NEGATIVE = ("photograph, photorealistic, 3d render, text, letters, words, watermark, "
            "logo, caption, stripes, lines, bars, grid, frame, border, pattern, "
            "scenery, clutter, multiple objects, blurry, badge, sticker outline, "
            "coloured background, grey background, gradient, shadow, vignette, ornate, "
            "decorative, folk art, papercut, floral, intricate, engraving, mandala")


def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label_below(img: Image.Image, title: str, prompt: str, width: int = 512):
    """The image with its model and full prompt printed underneath."""
    img = img.resize((width, width), Image.LANCZOS)
    lines = textwrap.wrap(prompt, width=62)[:5]
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
    ap.add_argument("--subjects", nargs="*",
                    default=["a bird chirping", "a fire engine with siren",
                             "a dog barking", "rain falling"])
    ap.add_argument("--models", nargs="*", default=["sdxlturbo"])
    ap.add_argument("--out", default="data/output/icon_grid")
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image
    out = _ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cells = {s: [] for s in args.subjects}

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
        for pkey, tmpl in PROMPTS.items():
            for subj in args.subjects:
                prompt = tmpl.format(s=subj)
                try:
                    kw = dict(prompt=prompt, width=512, height=512)
                    if turbo:      # turbo is distilled: no CFG, very few steps
                        kw.update(num_inference_steps=4, guidance_scale=0.0)
                    else:
                        kw.update(num_inference_steps=28, guidance_scale=7.5,
                                  negative_prompt=NEGATIVE)
                    img = pipe(**kw).images[0]
                except Exception as e:
                    print(f"  ! {mkey}/{pkey}/{subj}: {type(e).__name__}: {e}", flush=True)
                    continue
                cell = label_below(img, f"{mkey} | {pkey} | {subj}", prompt)
                cell.save(out / f"{mkey}_{pkey}_{subj.replace(' ', '_')}.png")
                cells[subj].append(cell)
                print(f"  {mkey:10} {pkey:13} {subj}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # one contact sheet per subject: every model x prompt for that sound, side by side
    for subj, imgs in cells.items():
        if not imgs:
            continue
        cols = len(PROMPTS)
        rows = (len(imgs) + cols - 1) // cols
        cw, ch = imgs[0].size
        sheet = Image.new("RGB", (cols * cw, rows * ch), (255, 255, 255))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))
        sheet.save(out / f"SHEET_{subj.replace(' ', '_')}.png")
        print(f"sheet -> SHEET_{subj.replace(' ', '_')}.png  ({len(imgs)} cells)", flush=True)


if __name__ == "__main__":
    main()
