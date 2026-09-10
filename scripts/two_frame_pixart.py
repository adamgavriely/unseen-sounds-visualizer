"""Two-frame GIF, generated simply: same picture, one small change that shows the sound.

Earlier attempts failed for a reason that is now understood. Same-seed text2img on
SDXL-Turbo changed the whole subject between frames; img2img on the same model held the
subject but never opened the mouth. Both are the same root cause: Turbo is distilled and
runs at guidance 0, so the prompt barely steers it once an init image is involved.

PixArt-Sigma is not distilled and uses real classifier-free guidance, so the prompt has
force. Two ways of keeping frame B the same picture as frame A are tried:

  seed   both frames from PixArt with the SAME SEED, prompts differing only in the
         moving part -- the cheapest thing that could work
  img2img  frame B denoised FROM frame A with SDXL at a low strength, which inherits the
         composition by construction. SDXL is used for this step because it has a
         well-supported img2img pipeline and, unlike Turbo, honours guidance.

Prompts stay plain: the whole lesson of the last several rounds is that style directives
("flat", "3d icon", "cartoon") are what produced unusable images.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw

PIXART = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
TAIL = ", plain white background, single subject, clearly visible"

# (name, frame A = at rest, frame B = the sound happening, seed)
PAIRS = [
    ("dog_barking", "a dog with its mouth closed",
     "a dog barking with its mouth wide open", 11),
    ("bird_chirping", "a bird with its beak closed",
     "a bird chirping with its beak wide open", 23),
    ("glass_shatter", "a whole drinking glass",
     "a drinking glass shattering into sharp pieces", 7),
]
STRENGTHS = (0.35, 0.50)


def _font(size):
    from PIL import ImageFont
    for n in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label(img, text, width=320):
    img = img.convert("RGB").resize((width, width), Image.LANCZOS)
    out = Image.new("RGB", (width, width + 30), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, width, width, width + 30], fill=(28, 30, 38))
    d.text((7, width + 8), text[:52], font=_font(13), fill=(226, 228, 235))
    return out


def main():
    import torch
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

    out = _ROOT / "data" / "output" / "two_frame_pixart"
    out.mkdir(parents=True, exist_ok=True)

    t2i = AutoPipelineForText2Image.from_pretrained(
        PIXART, torch_dtype=torch.float16).to("cuda")
    t2i.set_progress_bar_config(disable=True)

    frames_a, frames_seed = {}, {}
    for name, rest, sound, seed in PAIRS:
        for tag, subj in (("A", rest), ("B_seed", sound)):
            g = torch.Generator("cpu").manual_seed(seed)
            img = t2i(prompt=subj + TAIL, width=1024, height=1024,
                      num_inference_steps=25, guidance_scale=4.5,
                      generator=g).images[0]
            img.save(out / f"{name}_{tag}.png")
            (frames_a if tag == "A" else frames_seed)[name] = img
            print(f"  pixart {name} {tag}", flush=True)
    del t2i
    gc.collect()
    torch.cuda.empty_cache()

    i2i = AutoPipelineForImage2Image.from_pretrained(
        SDXL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    i2i.set_progress_bar_config(disable=True)

    for name, rest, sound, seed in PAIRS:
        base = frames_a[name]
        cells = [label(base, "A - at rest (PixArt)"),
                 label(frames_seed[name], "B - same seed")]
        base.save(out / f"{name}_cycle_seed.gif", save_all=True,
                  append_images=[frames_seed[name].resize(base.size)],
                  duration=420, loop=0)
        for st in STRENGTHS:
            g = torch.Generator("cpu").manual_seed(seed + 1)
            b = i2i(prompt=sound + TAIL, image=base.resize((1024, 1024)),
                    strength=st, num_inference_steps=30, guidance_scale=6.5,
                    generator=g).images[0]
            b.save(out / f"{name}_B_img2img{int(st*100)}.png")
            base.save(out / f"{name}_cycle_img2img{int(st*100)}.gif", save_all=True,
                      append_images=[b.resize(base.size)], duration=420, loop=0)
            cells.append(label(b, f"B - img2img {st}"))
            print(f"  img2img {name} strength={st}", flush=True)
        cw, ch = cells[0].size
        sheet = Image.new("RGB", (cw * len(cells), ch), (255, 255, 255))
        for i, c in enumerate(cells):
            sheet.paste(c, (i * cw, 0))
        sheet.save(out / f"SHEET_{name}.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
