"""FLUX.1-schnell with plain event prompts -- the proposal's own candidate model.

Two corrections at once, both found by re-reading the proposal.

MODEL. The proposal names FLUX.1 first and Stable Diffusion XL second. Six rounds were
spent on SDXL base, SDXL-Turbo and SD1.5+AnimateDiff without trying FLUX, which is the
one modern open model specifically known for following the prompt rather than imposing a
house style.

PROMPT. The proposal asks for "static storyboard-style images" -- a plain depiction of
the event. Rounds 1-6 invented a style instead ("minimal line art", "3d icon", "sticker",
"cartoon drawing") and every one of those is a strong artistic prior that pulls the model
away from simply showing the thing, which is exactly how the outputs kept coming out
stylised and hard to read. Simplicity has to come from the SUBJECT being isolated, not
from art-direction adjectives.

So: two prompt forms only, both plain.
  bare      the event and nothing else
  isolated  the event plus a plain background and a single subject
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw

MODEL = "black-forest-labs/FLUX.1-schnell"

SUBJECTS = ["a dog barking", "a bird chirping", "a fire engine with its siren on",
            "a glass shattering", "rain falling", "footsteps on a wooden floor"]

VARIANTS = {
    "bare": "{s}",
    "isolated": "{s}, plain white background, single subject, clearly visible",
}


def _font(size):
    from PIL import ImageFont
    for n in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label_below(img, title, prompt, width=380):
    img = img.resize((width, width), Image.LANCZOS)
    lines = textwrap.wrap(prompt, width=52)[:3]
    strip = 24 + 15 * (len(lines) + 1)
    out = Image.new("RGB", (width, width + strip), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, width, width, width + strip], fill=(28, 30, 38))
    d.text((7, width + 5), title, font=_font(13), fill=(255, 214, 102))
    for i, ln in enumerate(lines):
        d.text((7, width + 22 + 15 * i), ln, font=_font(12), fill=(226, 228, 235))
    return out


def main():
    import torch
    from diffusers import FluxPipeline

    out = _ROOT / "data" / "output" / "flux_test"
    out.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    # FLUX bf16 is ~24 GB and the L4 has 23; offloading keeps it resident on CPU and
    # moves modules to the GPU as needed. Slower per image, but it fits.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    cells = {s: [] for s in SUBJECTS}
    for vkey, tmpl in VARIANTS.items():
        for subj in SUBJECTS:
            prompt = tmpl.format(s=subj)
            try:
                img = pipe(prompt=prompt, width=512, height=512,
                           num_inference_steps=4,      # schnell is distilled for 4
                           guidance_scale=0.0,
                           max_sequence_length=256,
                           generator=torch.Generator("cpu").manual_seed(7)).images[0]
            except Exception as e:
                print(f"  ! {vkey}/{subj}: {type(e).__name__}: {e}", flush=True)
                continue
            stem = f"{vkey}_{subj.replace(' ', '_')[:26]}"
            img.save(out / f"{stem}.png")
            cells[subj].append(label_below(img, f"flux-schnell | {vkey}", prompt))
            print(f"  {vkey:9} {subj}", flush=True)

    rows = [c for s in SUBJECTS for c in cells[s]]
    if rows:
        cw, ch = rows[0].size
        cols = 4
        n = (len(rows) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, n * ch), (255, 255, 255))
        for i, c in enumerate(rows):
            sheet.paste(c, ((i % cols) * cw, (i // cols) * ch))
        sheet.save(out / "SHEET_flux.png")
        print(f"sheet -> SHEET_flux.png ({len(rows)} cells)", flush=True)


if __name__ == "__main__":
    main()
