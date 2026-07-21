"""Stage 4 — turn a Plan into an image on disk.

Backends:
  - "placeholder": draws the segment text on a coloured frame (PIL). No GPU,
    no keys. Lets us validate timing + compositing + sync immediately.
  - "diffusion":  Stable Diffusion / SDXL via diffusers. Enabled on the
    university GPU in a later step.
"""
from __future__ import annotations
import textwrap
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from .models import Plan


def _load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_placeholder(plan: Plan, out_path: Path, size=(1280, 720)) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Colour keyed off the segment index so consecutive frames differ visibly.
    hue = (plan.index * 47) % 360
    bg = _hsl_to_rgb(hue, 0.35, 0.20)
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)

    font = _load_font(48)
    label_font = _load_font(28)
    draw.text((40, 30), f"[{plan.index}] {plan.start:.1f}s–{plan.end:.1f}s",
              fill=(200, 200, 210), font=label_font)

    caption = plan.subject or plan.text
    wrapped = textwrap.fill(caption, width=32)
    draw.multiline_text((size[0] // 2, size[1] // 2), wrapped,
                        fill=(240, 240, 245), font=font, anchor="mm",
                        align="center", spacing=12)
    img.save(out_path)
    return out_path


def _hsl_to_rgb(h, s, l):
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def visualize(plans: List[Plan], work_dir: Path, backend: str = "placeholder",
              size=(1280, 720)) -> List[Plan]:
    img_dir = work_dir / "images"
    for p in plans:
        if not p.visualize:
            p.image_path = None
            continue
        out = img_dir / f"seg_{p.index:04d}.png"
        if backend == "placeholder":
            render_placeholder(p, out, size=size)
        elif backend == "diffusion":
            raise NotImplementedError(
                "diffusion backend not wired yet — run on the university GPU. "
                "Use --visualizer placeholder for now."
            )
        else:
            raise ValueError(f"unknown visualizer backend: {backend}")
        p.image_path = str(out)
    return plans
