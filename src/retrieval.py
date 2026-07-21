"""Free image retrieval via the Openverse API (Creative-Commons images, no key).

Given a text query, search Openverse, download the first result that opens as a
valid image, cover-crop it to the target size, and return attribution metadata
(title/creator/license/source) so it can be credited — important for a real,
non-hallucinated image of a named event.
"""
from __future__ import annotations
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

_OPENVERSE = "https://api.openverse.org/v1/images/"
_HEADERS = {"User-Agent": "MscFinalProject/0.1 (academic research)"}


def _http_get(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _search(query: str, page_size: int = 8) -> list[dict]:
    qs = urllib.parse.urlencode({"q": query, "page_size": page_size,
                                 "mature": "false"})
    data = _http_get(f"{_OPENVERSE}?{qs}")
    return json.loads(data).get("results", [])


def _cover_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = max(tw, int(w * scale)), max(th, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def fetch(query: str, out_path: Path, size: tuple[int, int]) -> Optional[dict]:
    """Download a relevant image to out_path. Returns attribution dict, or None."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        results = _search(query)
    except Exception:
        return None
    for r in results:
        url = r.get("url") or r.get("thumbnail")
        if not url:
            continue
        try:
            img = Image.open(io.BytesIO(_http_get(url))).convert("RGB")
            _cover_crop(img, size).save(out_path)
            return {
                "query": query,
                "title": r.get("title"),
                "creator": r.get("creator"),
                "license": r.get("license"),
                "license_url": r.get("license_url"),
                "source": r.get("foreign_landing_url") or url,
            }
        except Exception:
            continue
    return None
