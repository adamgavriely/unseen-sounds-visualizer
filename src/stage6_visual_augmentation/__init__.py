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
# ----------------------------------------------------------------------
def _timeline(specs: List[AugmentationSpec], duration: float):
    """Contiguous segments over [0,duration]; each carries the active image+label
    (the most recently-started augmentation covering that time), or None."""
    ev = sorted([s for s in specs if s.augment and s.image_path], key=lambda s: s.start)
    bounds = sorted({0.0, duration} | {s.start for s in ev}
                    | {min(s.end, duration) for s in ev})
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 0.05:
            continue
        mid = (a + b) / 2
        active = [s for s in ev if s.start <= mid < s.end]
        chosen = max(active, key=lambda s: s.start) if active else None
        segs.append((chosen.image_path if chosen else None,
                     chosen.event_label if chosen else "", b - a))
    return segs or [(None, "", duration)]


def composite_alongside(video_path: Path, specs: List[AugmentationSpec],
                        out_path: Path, duration: float,
                        panel: int = 720, fps: int = 25) -> Path:
    """Side-by-side: original video (left) + time-aligned augmentation panel (right)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f"_{out_path.stem}_panels"
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.png"):
        old.unlink()

    # build the panel slideshow inputs
    segs = _timeline(specs, duration)
    lines = []
    for i, (img, label, dur) in enumerate(segs):
        p = work / f"p{i:04d}.png"
        if img:
            _caption(_cover_crop(Image.open(img).convert("RGB"), (panel, panel)), label).save(p)
        else:
            base = Image.new("RGB", (panel, panel), (16, 18, 24))
            d = ImageDraw.Draw(base)
            d.text((panel // 2 - 40, panel // 2), "(no sound)", fill=(90, 96, 110))
            base.save(p)
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
