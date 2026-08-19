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
    d.rectangle([0, h - 52, w, h], fill=(0, 0, 0, 170))
    d.text((16, h - 38), text, fill=(240, 240, 245))
    return img


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
                _caption(_cover_crop(img, size), query).save(out_path)
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
        else:                                       # 'placeholder' (or diffusion TODO)
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
def _slot_order(specs: List[AugmentationSpec]) -> List[str]:
    """One slot per sound label, ordered by first appearance; stable all clip."""
    order = []
    for s in sorted([s for s in specs if s.augment and s.image_path],
                    key=lambda s: s.start):
        if s.event_label not in order:
            order.append(s.event_label)
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
                 label: str, mode: str) -> None:
    """Draw one fixed slot: image/chip when its sound is active, dim label when not."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    d = ImageDraw.Draw(canvas, "RGBA")
    if spec is None:                       # inactive: dark slot, dim label placeholder
        d.rectangle(box, fill=(16, 18, 24))
        d.text((x0 + 14, (y0 + y1) // 2 - 6), label, fill=(70, 76, 90))
    elif mode == "minimal":                # active chip: color band + label, no imagery
        d.rectangle(box, fill=(24, 26, 34))
        r, g, b = _chip_color(label)
        a = int(255 * _opacity(spec.confidence))
        d.rectangle([x0, y0, x0 + 10, y1], fill=(r, g, b, a))
        d.text((x0 + 24, (y0 + y1) // 2 - 6), label, fill=(230, 232, 240, a))
    else:                                  # active image, opacity = confidence weight
        img = _caption(_cover_crop(Image.open(spec.image_path).convert("RGB"), (w, h)),
                       label).convert("RGBA")
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
    segs = _timeline(specs, duration)
    lines = []
    for i, (active, dur) in enumerate(segs):
        p = work / f"p{i:04d}.png"
        canvas = Image.new("RGB", (panel, panel), (16, 18, 24))
        if not slots:
            d = ImageDraw.Draw(canvas)
            d.text((panel // 2 - 40, panel // 2), "(no sound)", fill=(90, 96, 110))
        else:
            sh = panel // len(slots)
            for k, label in enumerate(slots):
                y1 = panel if k == len(slots) - 1 else (k + 1) * sh
                _render_slot(canvas, (0, k * sh, panel, y1), active.get(label),
                             label, mode)
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
