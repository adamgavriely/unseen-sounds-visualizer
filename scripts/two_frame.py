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

# Adam's correction, and it is the right one: these generations were optimising how the
# picture LOOKS rather than how fast it is UNDERSTOOD. "cute 3d icon" buys rendering
# quality that nobody needs and costs clarity. What matters is choosing the depiction
# that most directly encodes the sound event, then drawing it plainly.
#
# So each frame is now specified by its MEANING, in his terms: a beak that closes and
# opens with a note when open; a dog's mouth closed then open with lines coming out; a
# whole glass then a broken one -- it does not have to be a cup; footprints alternating.
# The style prefix is deliberately plain and is the only thing varied across columns.

STYLES = {
    "P1_flat": ("simple flat illustration of {s}, clear simple shapes, few flat colours, "
                "plain white background, easy to understand at a glance, no text"),
    "P2_icon": ("simple clear picture of {s}, bold simple shapes, plain white "
                "background, instantly understandable, no text"),
}

# (name, frame A = at rest, frame B = the sound happening, seed)
PAIRS = [
    ("bird_chirping",
     "a small bird seen from the side with its beak closed",
     "the same small bird with its beak open and one music note next to its beak", 23),
    ("dog_barking",
     "a dog seen from the side with its mouth closed",
     "the same dog with its mouth open and three short curved lines coming out of it", 11),
    ("glass_shatter",
     "a whole unbroken glass object",
     "the same glass object broken into several large sharp pieces", 7),
    ("footsteps",
     "a single footprint on the ground",
     "two footprints on the ground, one in front of the other", 31),
]


def _font(size):
    from PIL import ImageFont
    for n in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label(img, text, width=380):
    img = img.resize((width, width), Image.LANCZOS)
    out = Image.new("RGB", (width, width + 34), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, width, width, width + 34], fill=(28, 30, 38))
    d.text((8, width + 9), text[:64], font=_font(13), fill=(226, 228, 235))
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
        rows = []
        for skey, style in STYLES.items():
            frames = []
            for tag, subj in (("A", rest), ("B", sound)):
                g = torch.Generator("cuda").manual_seed(seed)   # same seed = same subject
                img = pipe(prompt=style.format(s=subj), width=512, height=512,
                           num_inference_steps=4, guidance_scale=0.0,
                           generator=g).images[0]
                img.save(out / f"{name}_{skey}_{tag}.png")
                frames.append(img)
                print(f"  {name} {skey} {tag}", flush=True)
            frames[0].save(out / f"{name}_{skey}_cycle.gif", save_all=True,
                           append_images=[frames[1]], duration=380, loop=0)
            rows.append((skey, frames))
        sheet = Image.new("RGB", (380 * 2, 414 * len(rows)), (255, 255, 255))
        for i, (skey, fr) in enumerate(rows):
            sheet.paste(label(fr[0], f"{skey} | A rest"), (0, i * 414))
            sheet.paste(label(fr[1], f"{skey} | B sound"), (380, i * 414))
        sheet.save(out / f"SHEET_{name}.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
