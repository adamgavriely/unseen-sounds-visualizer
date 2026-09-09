"""Stage 6 - Visual Augmentation Generation + alongside-video compositing.

v1 prototype backend = RETRIEVE: fetch a free Creative-Commons image (Openverse,
no key) for each augmented sound, cover-crop it, and record attribution. v2 will
add a 'diffusion' backend (SDXL/FLUX). 'placeholder' draws a labelled panel and
needs nothing.

Compositing shows the augmentation image for the currently-active sound event
*alongside* the original video (side-by-side), time-aligned, with the original
audio kept. See docs/project_notes.tex sec:placement for the placement rationale.
"""
from __future__ import annotations

import io
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from src.types import AugmentationSpec
from src.labels import search_query

_OPENVERSE = "https://api.openverse.org/v1/images/"
_HEADERS = {"User-Agent": "MscFinalProject/0.1 (academic research)"}


# ----------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------
_FONT_CACHE = {}
# PIL's built-in bitmap font is ~11 px and unreadable on a 720 px panel, and the two
# platforms have different fonts: Windows ships Arial, while the cluster gets DejaVu
# from the conda-forge fontconfig packages. Try both, then fall back.
_FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _font(size: int):
    """A scalable font at `size`, or PIL's default if no TrueType file is found."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont
    import glob
    candidates = list(_FONT_PATHS)
    # conda envs install fonts under <prefix>/fonts or <prefix>/share/fonts
    import sys
    candidates += glob.glob(f"{sys.prefix}/fonts/**/*.ttf", recursive=True)
    candidates += glob.glob(f"{sys.prefix}/share/fonts/**/*.ttf", recursive=True)
    for path in candidates:
        try:
            f = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _cover_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = max(tw, int(w * scale)), max(th, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _caption(img: Image.Image, text: str) -> Image.Image:
    if not text:
        return img
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle([0, h - 58, w, h], fill=(0, 0, 0, 175))
    d.text((16, h - 44), text, font=_font(max(18, h // 26)), fill=(240, 240, 245))
    return img


def _sound_glyph(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Stamp a sound indicator onto the image, so the panel reads without decoding.

    A photograph of a fire engine says "there is a fire engine"; it does not say "you
    are HEARING one". The viewer has to infer that the panel is about sound at all,
    which is exactly the decoding step Adam asked to remove -- and the DHH
    visualization literature is consistent that the cue should be immediate rather than
    interpreted (sec:litreview). So the source and the fact-of-sound are combined in one
    simple picture: the retrieved photograph of the source, plus the standard radiating
    arcs, plus the source named in words.

    Deliberately drawn rather than generated. It has to be identical every time, legible
    over an arbitrary photograph, and cheap; a diffusion model gives none of those.
    """
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    r = max(7, int(min(w, h) * 0.038))
    # Put the mark against the thing that is making the sound -- beside the bird, not
    # parked in the corner of the frame. Adam's example was sound drawn at the bird's
    # mouth; the subject's silhouette is findable on a plain ground, so the arcs go just
    # outside its upper edge. Photographs fill the frame and fall back to the corner.
    box = _subject_bbox(img)
    if box:
        # emitter just past the subject's edge, arcs opening AWAY from it: sound comes
        # out of the bird, so the rings must expand outward, not wrap back around it
        cx = min(w - int(r * 4.6), box[2] + int(r * 1.1))
        cy = max(int(r * 2.2), box[1] + (box[3] - box[1]) // 4)
    else:
        cx, cy = w - int(r * 5.2), int(r * 4.2)

    # On a pictogram the ground is plain and light, so dark arcs read best and a disc
    # behind them would be visual noise. Over a photograph nothing can be assumed, so
    # white arcs on a dark disc are the only reliable option.
    r_, g_, b_ = img.convert("RGB").getpixel((min(cx, w - 1), min(cy, h - 1)))
    on_light = (r_ + g_ + b_) / 3 > 140 and box is not None
    ink = (28, 32, 44, 255) if on_light else (255, 255, 255, 235)
    if not on_light:
        pad = int(r * 3.6)
        d.ellipse([cx - pad, cy - pad, cx + pad, cy + pad], fill=(0, 0, 0, 120))
    # louder sounds get one more arc: the panel shows intensity without a number
    for i in range(3 if strength >= 0.45 else 2):
        rr = r * (1.3 + 0.78 * i)
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=-52, end=52,
              fill=ink, width=max(2, r // 3))
    d.ellipse([cx - r * 0.38, cy - r * 0.38, cx + r * 0.38, cy + r * 0.38], fill=ink)
    return img


_PIPE = None


# "line art" is taken literally: the first attempt produced a clean bird silhouette on a
# background of black stripes, and a good fire engine boxed in by spurious bars. The
# words that actually work are the stock-photo ones -- isolated, white background,
# sticker -- which name the RESULT rather than the drawing technique.
ICON_STYLE = ("a single {subject}, simple flat vector illustration, bold solid shapes, "
              "isolated on a plain empty white background, centered with wide empty "
              "margins around it, sticker style, high contrast, instantly recognisable, "
              "no text")
ICON_NEGATIVE = ("photograph, photorealistic, realistic, 3d render, text, letters, words, "
                 "watermark, logo, caption, stripes, lines, bars, grid, frame, border, "
                 "pattern, background decoration, scenery, clutter, multiple objects, "
                 "small details, blurry")


def icon_prompt(subject: str) -> str:
    """Ask for a pictogram rather than a picture.

    The generator ablation had SDXL rendering "a clear, simple illustration of: Siren"
    and losing to a stock photograph, because it produced a plausible SCENE whose
    subject the viewer -- and the describing VLM -- had to work out. A panel beside a
    video gets a glance, not a study; what it needs is the least ambiguous possible
    depiction of one thing.

    So the prompt asks for the thing diffusion is reliably good at and photographs are
    not: one large object, flat, few shapes, plain background, nothing else in frame.
    That also makes the subject's silhouette findable in code, which is what lets the
    sound indicator be placed against the object instead of parked in a corner.
    """
    return ICON_STYLE.format(subject=subject)


def _subject_bbox(img: Image.Image, tol: int = 28):
    """Where the drawn thing is, assuming a plain background.

    Icon-style output sits on a near-uniform ground, so anything differing from the
    corner colour is the subject. Returns None when that assumption does not hold (a
    photograph fills the frame), and the caller falls back to a fixed corner.
    """
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        # The prompt asks for a white ground, so "not near-white" is the subject. This
        # survives the faint gradients diffusion leaves behind, which a corner-colour
        # comparison does not -- that version rejected 7 of 8 real generations.
        mask = rgb.convert("L").point(lambda v: 255 if v < 238 else 0)
        box = mask.getbbox()
        if not box:
            return None
        cover = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
        ink = mask.histogram()[255] / float(w * h)
        # A photograph covers the frame; so does a generation that filled the background
        # with decoration. Either way there is no clean edge to put the mark against.
        if cover > 0.92 or ink > 0.62:
            return None
        return box
    except Exception:
        return None


def _diffusion_image(path: Path, prompt: str, size=(1024, 1024),
                     model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                     device: str = "cuda") -> bool:
    """v2-b backend: generate the augmentation with SDXL (GPU). One pipeline is
    kept loaded across calls -- model load dominates cost, generation is ~2 s."""
    global _PIPE
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        if _PIPE is None:
            _PIPE = StableDiffusionXLPipeline.from_pretrained(
                model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None, use_safetensors=True)
            _PIPE = _PIPE.to(device)
            _PIPE.set_progress_bar_config(disable=True)
        img = _PIPE(prompt=prompt, negative_prompt=ICON_NEGATIVE,
                    width=size[0], height=size[1],
                    num_inference_steps=28, guidance_scale=8.0).images[0]
        img.save(path)
        return True
    except Exception as e:
        print(f"       [stage6] diffusion failed ({type(e).__name__}: {e}); "
              f"falling back to placeholder")
        return False


def _placeholder_image(path: Path, caption: str, size=(1024, 1024)) -> None:
    img = Image.new("RGB", size, color=(20, 22, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, size[0] - 8, size[1] - 8], outline=(90, 100, 140), width=3)
    d.text((28, 28), "[augmentation placeholder]", fill=(150, 160, 190))
    _caption(img, caption)
    img.save(path)


# ----------------------------------------------------------------------
# Openverse retrieval (salvaged from the legacy pipeline)
# ----------------------------------------------------------------------
def _http_get(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _search(query: str, page_size: int = 8) -> list:
    qs = urllib.parse.urlencode({"q": query, "page_size": page_size, "mature": "false"})
    return json.loads(_http_get(f"{_OPENVERSE}?{qs}")).get("results", [])


def _query_variants(query: str) -> list:
    words = query.split()
    variants, seen = [], set()
    for n in (len(words), 6, 4, 3, 2, 1):
        v = " ".join(words[:n]).strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            variants.append(v)
    return variants


def _retrieve(query: str, out_path: Path, size: Tuple[int, int]) -> Optional[dict]:
    """Download a relevant CC image to out_path; return attribution dict or None."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = query.split()[0].lower() if query else ""
    for q in _query_variants(query):
        try:
            results = _search(q)
        except Exception:
            continue
        # prefer results whose title actually mentions the keyword (mild on-topic bias)
        results.sort(key=lambda r: 0 if kw and kw in (r.get("title") or "").lower() else 1)
        for r in results:
            url = r.get("url") or r.get("thumbnail")
            if not url:
                continue
            try:
                img = Image.open(io.BytesIO(_http_get(url))).convert("RGB")
                _caption(_sound_glyph(_cover_crop(img, size)), query).save(out_path)
                return {"requested": query, "matched_query": q, "title": r.get("title"),
                        "creator": r.get("creator"), "license": r.get("license"),
                        "license_url": r.get("license_url"),
                        "source": r.get("foreign_landing_url") or url}
            except Exception:
                continue
    return None


# ----------------------------------------------------------------------
# Stage 6 entry: produce one image per augmented sound
# ----------------------------------------------------------------------
def generate_augmentations(specs: List[AugmentationSpec], work_dir: Path,
                           backend: str = "retrieve", size=(1024, 1024),
                           model: str = "", device: str = "cpu") -> List[AugmentationSpec]:
    out_dir = work_dir / "augmentations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("aug_*.png"):
        old.unlink()

    credits, n = [], 0
    for spec in specs:
        if not spec.augment:
            continue
        path = out_dir / f"aug_{spec.index:03d}.png"
        query = search_query(spec.subject or spec.event_label)
        if backend == "retrieve":
            meta = _retrieve(query, path, size)
            if meta:
                spec.image_path = str(path)
                spec.backend = "retrieve"
                credits.append({"index": spec.index, "label": spec.event_label, **meta})
            else:                                   # graceful fallback
                _placeholder_image(path, query, size)
                spec.image_path = str(path)
                spec.backend = "placeholder"
        elif backend == "diffusion":                # v2-b, university GPU
            # Always the pictogram prompt: what Stage 5 stored is the subject, and the
            # style is this backend's business, not the gate's.
            prompt = icon_prompt(spec.subject or query)
            if _diffusion_image(path, prompt, size, model=model, device=device):
                spec.image_path = str(path)
                spec.backend = "diffusion"
            else:
                _placeholder_image(path, query, size)
                spec.image_path = str(path)
                spec.backend = "placeholder"
        else:                                       # 'placeholder'
            _placeholder_image(path, spec.image_prompt or query, size)
            spec.image_path = str(path)
            spec.backend = "placeholder"
        n += 1

    (work_dir / "credits.json").write_text(
        json.dumps(credits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"       [stage6] {backend}: produced {n} augmentation image(s); "
          f"{len(credits)} retrieved (rest placeholder).")
    return specs


# ----------------------------------------------------------------------
# alongside-video compositor
#
# Panel layout follows the DHH-visualization evidence (notes, "Rendering
# improvements"): each augmented sound gets a FIXED slot for the whole clip
# (moving/popping visuals distract DHH viewers), and an active slot's opacity
# scales with detector confidence (loudness -> visual weight). TODO: temporal
# fade at slot activation (needs per-frame rendering, not a concat slideshow).
# ----------------------------------------------------------------------
MAX_SLOTS = 3   # granularity requirement (notes sec:granularity): >=3 simultaneous
                # sources never observed on the benchmark; more would split attention


def _slot_order(specs: List[AugmentationSpec]) -> List[str]:
    """One slot per sound label, ordered by first appearance; stable all clip.
    Capped at MAX_SLOTS keeping the highest-confidence sources."""
    aug = [s for s in specs if s.augment and s.image_path]
    order = []
    for s in sorted(aug, key=lambda s: s.start):
        if s.event_label not in order:
            order.append(s.event_label)
    if len(order) > MAX_SLOTS:
        best = {}
        for s in aug:
            best[s.event_label] = max(best.get(s.event_label, 0.0), s.confidence)
        keep = sorted(order, key=lambda lb: -best[lb])[:MAX_SLOTS]
        order = [lb for lb in order if lb in keep]
    return order


def _timeline(specs: List[AugmentationSpec], duration: float):
    """Contiguous segments over [0,duration]; each carries {label: active spec}."""
    ev = [s for s in specs if s.augment and s.image_path]
    bounds = sorted({0.0, duration} | {s.start for s in ev}
                    | {min(s.end, duration) for s in ev})
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 0.05:
            continue
        mid = (a + b) / 2
        active = {}
        for s in ev:
            if s.start <= mid < s.end:
                active[s.event_label] = s
        segs.append((active, b - a))
    return segs or [({}, duration)]


def _opacity(confidence: float) -> float:
    """Confidence -> visual weight: faint sounds render translucent, strong ones
    solid (evidence: SoundVizVR loudness encoding / Fortnite distance-as-opacity)."""
    return 0.45 + 0.55 * max(0.0, min(1.0, confidence / 0.6))


def _chip_color(label: str) -> tuple:
    palette = [(214, 93, 76), (76, 145, 214), (98, 180, 106), (206, 164, 66),
               (160, 108, 208), (72, 180, 178)]
    return palette[hash(label) % len(palette)]


def _render_slot(canvas: Image.Image, box: tuple, spec: Optional[AugmentationSpec],
                 label: str, mode: str, idle_image: Optional[str] = None) -> None:
    """Draw one fixed slot: image/chip when its sound is active, dimmed when not.

    An inactive slot used to be a black rectangle, which a viewer reads as a broken
    player rather than as "this sound is not currently present". Keeping the image
    visible but heavily dimmed conveys "heard a moment ago" and keeps the panel
    legible, while still making the active moment obvious.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    d = ImageDraw.Draw(canvas, "RGBA")
    if spec is None:                       # inactive
        if idle_image and mode != "minimal":
            img = _cover_crop(Image.open(idle_image).convert("RGB"), (w, h)).convert("RGBA")
            img.putalpha(48)               # ~19%: present but clearly not active
            base = Image.new("RGBA", (w, h), (16, 18, 24, 255))
            canvas.paste(Image.alpha_composite(base, img).convert("RGB"), (x0, y0))
            d.text((x0 + 14, y1 - 34), label, font=_font(max(15, h // 14)),
                   fill=(140, 148, 165))
        else:
            d.rectangle(box, fill=(16, 18, 24))
            d.text((x0 + 14, (y0 + y1) // 2 - 10), label, font=_font(max(15, h // 14)),
                   fill=(96, 103, 118))
    elif mode == "minimal":                # active chip: color band + label, no imagery
        d.rectangle(box, fill=(24, 26, 34))
        r, g, b = _chip_color(label)
        a = int(255 * _opacity(spec.confidence))
        d.rectangle([x0, y0, x0 + 10, y1], fill=(r, g, b, a))
        d.text((x0 + 24, (y0 + y1) // 2 - 10), label, font=_font(max(16, h // 12)),
               fill=(230, 232, 240, a))
    else:                                  # active image, opacity = confidence weight
        # Glyph BEFORE caption: the caption bar spans the full width, and drawing it
        # first makes the subject's bounding box the whole frame, which parks the sound
        # mark in the corner instead of against the thing making the sound.
        img = _caption(
            _sound_glyph(_cover_crop(Image.open(spec.image_path).convert("RGB"), (w, h)),
                         spec.confidence), label).convert("RGBA")
        img.putalpha(int(255 * _opacity(spec.confidence)))
        base = Image.new("RGBA", (w, h), (16, 18, 24, 255))
        canvas.paste(Image.alpha_composite(base, img).convert("RGB"), (x0, y0))
    d.line([x0, y1 - 1, x1, y1 - 1], fill=(40, 44, 56))


def composite_alongside(video_path: Path, specs: List[AugmentationSpec],
                        out_path: Path, duration: float,
                        panel: int = 720, fps: int = 25,
                        mode: str = "full") -> Path:
    """Side-by-side: original video (left) + time-aligned augmentation panel (right).

    mode: "full" (imagery in stable slots) | "minimal" (label chips) |
    "off" (no panel: re-encode the original as-is, the control condition)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "off":
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(out_path)], check=True, capture_output=True)
        return out_path
    work = out_path.parent / f"_{out_path.stem}_panels"
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.png"):
        old.unlink()

    # build the panel slideshow inputs: fixed slot per sound, stacked vertically
    slots = _slot_order(specs)
    # one image per slot, so an inactive slot can still show its (dimmed) picture
    slot_image = {}
    for sp in specs:
        if sp.augment and sp.image_path and sp.event_label not in slot_image:
            slot_image[sp.event_label] = sp.image_path
    segs = _timeline(specs, duration)
    lines = []
    for i, (active, dur) in enumerate(segs):
        p = work / f"p{i:04d}.png"
        canvas = Image.new("RGB", (panel, panel), (16, 18, 24))
        if not slots:
            # No augmentation at all is a DECISION, not a failure: say so at a size
            # that is actually readable, or the blank panel looks like a broken player.
            d = ImageDraw.Draw(canvas)
            big, small = _font(max(22, panel // 22)), _font(max(16, panel // 32))
            l1, l2 = "no off-screen sound detected", "nothing to add for this scene"
            for text, fnt, dy, col in ((l1, big, -26, (185, 192, 208)),
                                       (l2, small, 18, (120, 127, 143))):
                w = d.textlength(text, font=fnt)
                d.text(((panel - w) / 2, panel / 2 + dy), text, font=fnt, fill=col)
        else:
            sh = panel // len(slots)
            for k, label in enumerate(slots):
                y1 = panel if k == len(slots) - 1 else (k + 1) * sh
                _render_slot(canvas, (0, k * sh, panel, y1), active.get(label),
                             label, mode, slot_image.get(label))
        canvas.save(p)
        # concat resolves 'file' paths relative to concat.txt's own dir -> use basenames
        lines += [f"file '{p.name}'", f"duration {dur:.3f}"]
    lines.append(f"file 'p{len(segs)-1:04d}.png'")
    concat = work / "concat.txt"
    concat.write_text("\n".join(lines), encoding="utf-8")

    panel_mp4 = work / "panel.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-r", str(fps), "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(panel_mp4)],
                   check=True, capture_output=True)

    # scale original to panel height and hstack the two; keep original audio.
    # normalise fps + SAR on BOTH inputs first, else the hstack can stall/balloon
    # on a framerate/aspect mismatch (source VP8 30fps vs panel 25fps).
    fc = (f"[0:v]scale=-2:{panel},fps={fps},setsar=1[l];"
          f"[1:v]scale={panel}:{panel},fps={fps},setsar=1[r];"
          f"[l][r]hstack=inputs=2[v]")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(panel_mp4),
        "-filter_complex", fc,
        "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path)],
        check=True, capture_output=True)
    return out_path
