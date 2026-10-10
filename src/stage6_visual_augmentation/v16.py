"""v1.6 onset rules (Oct 2026, benchmark/gold/coverage/), applied to the display spans in _display_spans after the v1.4 /
v1.5 rules. Each reads the picture's family curve of a detector (max over the detector's classes in the picture's family,
score_per_sound.same_family) around the picture's start; never where the picture sits in the clip.

* onset score of a detector: the family curve's max in [start - 0.25, start + 0.75] s; rise: the onset score minus the
  median of the curve over [start - 3, start - 0.5] s (empty window -> 0). A missing curve scores 0 and has no rise.
* COONSET_CONTEST (s): two pictures of different families that overlap in time and start <= this apart -> the one with the
  lower onset score (mean over BEATs, FlexSED, DASM) is dropped; a tie drops the lower stage-5 confidence.
* WEAK_NO_RISE (conf, rise): drop a picture whose stage-5 confidence is < conf and whose FlexSED rise is < rise.
* BEATS_NO_RISE: drop a picture whose BEATs rise is <= 0.
All three are decided on the same set of pictures and applied together.

Curves online: BEATs from stage 4's own framewise output (config.BEATS_FRAMES, set by detect_events for the clip in hand),
FlexSED from its frame cache (as HOLD_FLEXSED, v14._flex), DASM from config.LISTENER_DASM_DIR/<clip>.npz (as the F8 vote).
Offline: config.ONSET_CURVES (benchmark/gold/v14/detector_curves.json).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

import config

DETECTORS = ("beats", "flex", "dasm")
_OFF = {}
_DASM = {}


def _offline(clip: str):
    """config.ONSET_CURVES: {stem: {label: {det: [t0, dt, [v ...]] or [[t ...], [v ...]]}}}, the family curves stored for
    offline scoring; a clip it does not hold has no curves (never the live caches)"""
    path = getattr(config, "ONSET_CURVES", None)
    if not path:
        return None
    if path not in _OFF:
        p = Path(path)
        _OFF[path] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _OFF[path].get(clip, {})


def _dasm(clip: str):
    d = getattr(config, "LISTENER_DASM_DIR", None)
    if (d, clip) not in _DASM:
        f = Path(d) / f"{clip}.npz" if d else None
        if f is None or not f.exists():
            _DASM[(d, clip)] = None
        else:
            z = np.load(f)
            _DASM[(d, clip)] = (z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]])
    return _DASM[(d, clip)]


def _frames(clip: str):
    """{det: (fw [T, Q], times, labels) or None} from what the pipeline holds for this clip"""
    from src.stage6_visual_augmentation.v14 import _flex
    b = getattr(config, "BEATS_FRAMES", None)
    return {"beats": tuple(b[1:]) if b and b[0] == clip else None, "flex": _flex(clip), "dasm": _dasm(clip)}


def curves(label: str, clip: Optional[str], frames=None, stored=None):
    """{det: (t, v) or None}: the label's family curve of each detector"""
    from benchmark.gold.score_per_sound import same_family
    out = {}
    for det in DETECTORS:
        if stored is not None:
            c = (stored.get(label) or {}).get(det)
            if c and len(c) == 2:                                       # off a regular grid
                c = (np.asarray(c[0], float), np.asarray(c[1], float))
            elif c:
                c = (c[0] + c[1] * np.arange(len(c[2])), np.asarray(c[2], float))
            out[det] = c or None
            continue
        fr = (frames or {}).get(det)
        cols = [i for i, l in enumerate(fr[2]) if same_family(l, label)] if fr is not None else []
        out[det] = (np.asarray(fr[1], float), fr[0][:, cols].max(axis=1)) if cols else None
    return out


def onset(c, a: float):
    """(onset score, rise) of curve c = (t, v) at start a; (0, None) without a curve"""
    if c is None:
        return 0.0, None
    t, v = c
    w = (t >= a - 0.25) & (t <= a + 0.75)
    if not w.any():
        return 0.0, None
    at = float(v[w].max())
    pre = (t >= a - 3.0) & (t <= a - 0.5)
    return at, at - (float(np.median(v[pre])) if pre.any() else 0.0)


def onset_drops(spans, clip: Optional[str]):
    """spans: [label, start, end, spec] lists -> the same list without the pictures the three rules drop"""
    co = getattr(config, "COONSET_CONTEST", None)
    weak = getattr(config, "WEAK_NO_RISE", None)
    brise = getattr(config, "BEATS_NO_RISE", False)
    if not spans or not clip or (co is None and weak is None and not brise):
        return spans
    from benchmark.gold.score_per_sound import same_family
    stored = _offline(clip)
    frames = None if stored is not None else _frames(clip)
    f = []
    for sp in spans:
        c = curves(sp[0], clip, frames, stored)
        o = {det: onset(c[det], sp[1]) for det in DETECTORS}
        f.append({"score": sum(o[d][0] for d in DETECTORS) / len(DETECTORS), "flex": o["flex"][1], "beats": o["beats"][1],
                  "conf": float(getattr(sp[3], "confidence", 0.0) or 0.0)})
    drop = set()
    if co is not None:
        for i, p in enumerate(spans):
            for j in range(i + 1, len(spans)):
                q = spans[j]
                if same_family(p[0], q[0]) or min(p[2], q[2]) - max(p[1], q[1]) <= 0 or abs(p[1] - q[1]) > float(co):
                    continue
                drop.add(min((i, j), key=lambda k: (f[k]["score"], f[k]["conf"])))
    for i, x in enumerate(f):
        if weak is not None and x["conf"] < float(weak[0]) and (x["flex"] or 0.0) < float(weak[1]):
            drop.add(i)
        if brise and x["beats"] is not None and x["beats"] <= 0:
            drop.add(i)
    return [sp for i, sp in enumerate(spans) if i not in drop]


def gate_doubt_drops(spans, clip: Optional[str]):
    """v1.7 GATE_DOUBT_DASM (t): a picture whose sound the on-screen check voted "seen" (majority or describe) in at least
    one stretch, yet drawn (spec.gate_doubt), is dropped unless DASM's family curve reaches t somewhere inside the picture.
    No DASM curve for the family -> kept. benchmark/gold/coverage/opusC_rules.md"""
    t = getattr(config, "GATE_DOUBT_DASM", None)
    if t is None or not spans or not clip:
        return spans
    stored = _offline(clip)
    frames = None if stored is not None else _frames(clip)
    out = []
    for sp in spans:
        if getattr(sp[3], "gate_doubt", False):
            c = curves(sp[0], clip, frames, stored)["dasm"]
            if c is not None:
                w = (c[0] >= sp[1]) & (c[0] <= sp[2])
                if w.any() and float(c[1][w].max()) < float(t):
                    continue
        out.append(sp)
    return out
