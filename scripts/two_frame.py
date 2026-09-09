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

# Two independent generations give two different birds, because a changed prompt moves
# the image with it -- a fixed seed holds the composition only while the prompt is
# nearly identical. Adam's requirement is the same picture with a slight action change,
# which is what img2img is for: frame B is generated FROM frame A, so the bird, its
# colours and its pose are inherited and only the beak opens.
#
# strength is the whole game. Too low and nothing moves; too high and it is a fresh
# image again, which is the failure this is meant to fix. Three values are tried so the
# sweet spot is chosen by looking rather than guessed.

STRENGTHS = (0.25, 0.40, 0.55)

STYLE = ("simple flat illustration of {s}, clear simple shapes, few flat colours, "
         "plain white background, easy to understand at a glance, no text")

# (name, frame A = at rest, frame B = the sound happening, seed)
PAIRS = [
    ("bird_chirping",
     "a small bird seen from the side with its beak closed",
     "a small bird seen from the side with its beak wide open, one music note beside it", 23),
    ("dog_barking",
     "a dog seen from the side with its mouth closed",
     "a dog seen from the side with its mouth wide open, three short lines coming out", 11),
    ("glass_shatter",
     "a drinking glass standing whole and unbroken",
     "a drinking glass broken into several large sharp pieces", 7),
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


def label(img, text, width=340):
    img = img.resize((width, width), Image.LANCZOS)
    out = Image.new("RGB", (width, width + 32), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, width, width, width + 32], fill=(28, 30, 38))
    d.text((8, width + 8), text[:58], font=_font(13), fill=(226, 228, 235))
    return out


def main():
    import torch
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
    import config
    out = _ROOT / "data" / "output" / "two_frame"
    out.mkdir(parents=True, exist_ok=True)
    t2i = AutoPipelineForText2Image.from_pretrained(
        config.GEN_MODEL, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    t2i.set_progress_bar_config(disable=True)
    # shares the same weights, so this costs no extra memory
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    i2i.set_progress_bar_config(disable=True)

    for name, rest, sound, seed in PAIRS:
        g = torch.Generator("cuda").manual_seed(seed)
        frame_a = t2i(prompt=STYLE.format(s=rest), width=512, height=512,
                      num_inference_steps=4, guidance_scale=0.0, generator=g).images[0]
        frame_a.save(out / f"{name}_A.png")
        print(f"  {name} A", flush=True)
        cells = [label(frame_a, "A - at rest (text2img)")]
        for st in STRENGTHS:
            g = torch.Generator("cuda").manual_seed(seed + 1)
            # turbo needs enough steps that strength*steps >= 1, or nothing changes
            steps = max(2, int(round(4 / max(st, 0.1))))
            b = i2i(prompt=STYLE.format(s=sound), image=frame_a, strength=st,
                    num_inference_steps=steps, guidance_scale=0.0,
                    generator=g).images[0]
            b.save(out / f"{name}_B_str{int(st * 100)}.png")
            frame_a.save(out / f"{name}_cycle_str{int(st * 100)}.gif", save_all=True,
                         append_images=[b], duration=380, loop=0)
            cells.append(label(b, f"B - strength {st}"))
            print(f"  {name} B strength={st}", flush=True)
        cw, ch = cells[0].size
        sheet = Image.new("RGB", (cw * len(cells), ch), (255, 255, 255))
        for i, c in enumerate(cells):
            sheet.paste(c, (i * cw, 0))
        sheet.save(out / f"SHEET_{name}.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
