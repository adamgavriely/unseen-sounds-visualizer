"""v1.4 display rules (benchmark/gold/coverage/, Oct 2026), applied to the display spans in _display_spans:

* FLASH_RULE: the gate sees about one frame per second, so a lightning or blast flash on screen is missed. A flash is a
  frame whose mean luminance exceeds the median of the previous second by > 5 robust sigmas (1.4826 x MAD, floor 2 grey
  levels), in a run of <= 4 frames. A Thunder picture is dropped if a flash lies in [start - 2 s, end]; an Explosion /
  Fireworks / Gunshot / Machine gun / Artillery fire picture if a flash lies within +-0.3 s of its start.
* PICTURE_BAN: labels never drawn (generic textures: Vehicle, Water, Engine, Rain, Wind, ...).
* HOLD_FLEXSED: a picture's end grows while FlexSED still hears its family (max over the FlexSED queries of the family
  >= the bar), gaps < 1 s bridged; never into the next picture of the same label (minus the merge gap) or past the clip.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

import config

THUNDER = {"Thunder", "Thunderstorm"}
BLAST = {"Explosion", "Fireworks", "Gunshot, gunfire", "Gunshot", "Machine gun", "Artillery fire"}
_FLASH = {}
_FLEX = {}


def scan_flashes(video_path: Path) -> List[float]:
    """flash times (s) of a video at its full frame rate"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ys = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ys.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2YUV)[:, :, 0].mean()))
    y = np.array(ys)
    w = max(1, int(round(fps)))
    hot = np.zeros(len(y), bool)
    for i in range(w, len(y)):
        base = y[i - w:i]
        med = np.median(base)
        sig = max(2.0, 1.4826 * np.median(np.abs(base - med)))
        hot[i] = y[i] > med + 5 * sig
    out, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            if j - i <= 4:
                out.append(round(i / fps, 3))
            i = j
        else:
            i += 1
    return out


def flash_times(clip: Optional[str]) -> List[float]:
    """config.FLASH_TIMES (set by pipeline.run for the clip being processed), else the cache config.FLASH_CACHE
    ({stem: {"flashes": [...]}}), else none"""
    live = getattr(config, "FLASH_TIMES", None)
    if live is not None and getattr(config, "GROUP_CLIP", None) == clip:
        return list(live)
    path = getattr(config, "FLASH_CACHE", None)
    if path and clip:
        if path not in _FLASH:
            p = Path(path)
            _FLASH[path] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return list(_FLASH[path].get(clip, {}).get("flashes", []))
    return []


def flash_drop(label: str, start: float, end: float, flashes: List[float]) -> bool:
    from src.labels import canonical
    if label in THUNDER or canonical(label) in THUNDER:
        return any(start - 2.0 <= t <= end for t in flashes)
    if label in BLAST or canonical(label) in BLAST:
        return any(abs(t - start) <= 0.3 for t in flashes)
    return False


def _flex(clip: str):
    if clip not in _FLEX:
        d = Path(getattr(config, "HOLD_FLEXSED_DIR", None) or (config.WORK_DIR / "flexsed_cache"))
        f = d / f"{clip}.npz"
        if not f.exists():
            _FLEX[clip] = None
        else:
            z = np.load(f, allow_pickle=False)
            fw = z["fw"].astype(np.float32)
            labs = [str(x) for x in z["labels"]]
            if "times" in z:                                     # as dev_candidates_check.load_fr
                t = z["times"].astype(np.float64)
            else:
                fw = fw.T                                        # stored [labels, frames]
                t = np.arange(fw.shape[0], dtype=np.float64) / float(z["fps"])
            _FLEX[clip] = (fw, t, labs)
    return _FLEX[clip]


_CURVES = {}


def _curves(clip: str):
    """config.HOLD_CURVES: {stem: {"labels": {label: {"flex": {"t": [...], "v": [...]}}}}}, the FlexSED family curves
    (max over the queries of the label's family), as stored for offline scoring in benchmark/gold/v14/"""
    path = getattr(config, "HOLD_CURVES", None)
    if not path:
        return None
    if path not in _CURVES:
        p = Path(path)
        _CURVES[path] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return (_CURVES[path].get(clip) or {}).get("labels")


def hold_ends(spans, duration: float, clip: Optional[str], gap: float):
    """spans: mutable [label, start, end, spec] lists; ends may only grow"""
    bar = getattr(config, "HOLD_FLEXSED", None)
    if bar is None or not clip:
        return spans
    fr = _flex(clip)
    curves = None
    if fr is None:
        curves = _curves(clip)                   # offline: precomputed family curves (benchmark/gold/v14/)
        if curves is None:
            return spans
    from benchmark.gold.score_per_sound import same_family
    fw, t, labs = fr if fr is not None else (None, None, None)
    step, bridge = 0.02, 1.0
    for sp in spans:
        lab, a, b = sp[0], sp[1], sp[2]
        if curves is not None:
            c = (curves.get(lab) or {}).get("flex")
            if not c or not c.get("t"):
                continue
            t, v = np.asarray(c["t"], float), np.asarray(c["v"], float)
        else:
            cols = [i for i, l in enumerate(labs) if same_family(l, lab)]
            if not cols:
                continue
            v = fw[:, cols].max(axis=1)
        nxt = [o[1] for o in spans if o[0] == lab and o[1] > a + 1e-9]
        cap = min([float(duration)] + [n - gap for n in nxt])
        x, last = b, None
        while x < cap:
            i = int(np.clip(np.searchsorted(t, x), 0, len(t) - 1))
            if i > 0 and abs(t[i - 1] - x) <= abs(t[i] - x):
                i -= 1
            if v[i] >= bar:
                last = x
            elif (last is None and x - b >= bridge) or (last is not None and x - last >= bridge):
                break
            x += step
        if last is not None:
            sp[2] = max(b, min(cap, max(a + float(getattr(config, "MIN_DWELL", 1.5)), last)))
    return spans
