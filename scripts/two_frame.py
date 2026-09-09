"""Two-frame cycle: does alternating two generated frames read as the action happening?

The question is whether motion is worth its cost. Full video generation is ~1-3 min per
clip against ~1 s for a still, breaks the Stage-7 evaluation (which describes an image),
and fights the divided-attention finding in the literature review -- a moving panel
competes for the attention it is meant to supplement. A two-frame cycle is the cheapest
thing that still reads as motion: two stills and a timer.

The risk is consistency. Two independent generations of "a dog" give two different dogs,
and a cycle between them reads as a glitch rather than a bark. Both frames therefore use
the SAME SEED and prompts differing only in the moving part, which keeps composition,
colour and breed fixed and moves the mouth.

Writes, per subject: both frames, an animated GIF, and a labelled side-by-side PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw

BASE = ("cute simple 3d icon of {s}, soft rounded shapes, soft muted natural colours, "
        "plain white background, centered, friendly and clear, no text")

# (name, frame A = at rest, frame B = making the sound, seed)
PAIRS = [
    ("dog_barking", "a dog with its mouth closed sitting",
     "the same dog barking with its mouth wide open and a few small music notes beside it", 11),
    ("bird_chirping", "a small bird with its beak closed",
     "the same small bird chirping with its beak open and a few small music notes beside it", 23),
    ("glass_shatter", "a drinking glass standing whole",
     "the same drinking glass shattering into sharp flying pieces", 7),
    ("footsteps", "a pair of shoes standing still",
     "the same pair of shoes walking with small motion marks under them", 31),
]


def _font(size):
    from PIL import ImageFont
    for n in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label(img, text, width=420):
    img = img.resize((width, width), Image.LANCZOS)
    out = Image.new("RGB", (width, width + 30), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, width, width, width + 30], fill=(28, 30, 38))
    d.text((8, width + 8), text[:70], font=_font(13), fill=(226, 228, 235))
    return out


def main():
    import torch
    from diffusers import AutoPipelineForText2Image
    import config
    out = _ROOT / "data" / "output" / "two_frame"
    out.mkdir(parents=True, exist_ok=True)
    pipe = AutoPipelineForText2Image.from_pretrained(
        config.GEN_MODEL, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for name, rest, sound, seed in PAIRS:
        frames = []
        for tag, subj in (("A_rest", rest), ("B_sound", sound)):
            g = torch.Generator("cuda").manual_seed(seed)   # same seed = same subject
            img = pipe(prompt=BASE.format(s=subj), width=512, height=512,
                       num_inference_steps=4, guidance_scale=0.0, generator=g).images[0]
            img.save(out / f"{name}_{tag}.png")
            frames.append(img)
            print(f"  {name} {tag}", flush=True)
        # the cycle itself, at roughly the rate a mouth opens
        frames[0].save(out / f"{name}_cycle.gif", save_all=True,
                       append_images=[frames[1]], duration=380, loop=0)
        sheet = Image.new("RGB", (420 * 2, 450), (255, 255, 255))
        sheet.paste(label(frames[0], "A - at rest"), (0, 0))
        sheet.paste(label(frames[1], "B - making the sound"), (420, 0))
        sheet.save(out / f"{name}_frames.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
