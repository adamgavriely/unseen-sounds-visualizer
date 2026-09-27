"""Amendment 22, Stage 0 (docs/prereg_v4.md): the detector round's pick, on the 280 AudioSet-Strong calibration clips,
from cached frame scores, following src/stage4_audio_event_detection.detect_events step by step:

  BEATs spans at AED_THRESHOLD 0.175 (display later at 0.35) -> FlexSED spans at its bar -> twin rule (a same-family
  BEATs span within 1 s absorbs the FlexSED span; cell F: only a displayable BEATs span absorbs) -> tier 2 on
  FlexSED-only spans -> FlexSED veto 0.3 (clip peak) -> PANNs veto 0.05 on FlexSED-only spans -> tier 3 (weak BEATs spans
  promoted) -> display bar 0.35.

Scoring as benchmark/audioset_stage4_report.py (consequential onset-recall in [-0.5, +1.0] s, false spans/min).
Pick rule: the cell with the highest consequential onset-recall among cells whose false spans/min <= the shipped cell's
on the same clips. (The 4.56 of the D6 report came from a simplified stack without the weak-twin absorption; the
comparison here is against the shipped stack computed by this same code.)

    python benchmark/detector_round_stage0.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_stage4_report as R
from src.labels import canonical
from src.stage4_audio_event_detection import _extract_events

CELLS = {  # name: (fbar, tier2 (b, p, win) or None, tier3 (f, p, win) or None, weak_twin)
    "shipped (0.8)":        (0.8, None, None, "absorb"),
    "A bar 0.6":            (0.6, None, None, "absorb"),
    "B bar 0.5":            (0.5, None, None, "absorb"),
    "C bar 0.6 + tier 2":   (0.6, (0.1, 0.05, 1.0), None, "absorb"),
    "D bar 0.5 + tier 2":   (0.5, (0.1, 0.05, 1.0), None, "absorb"),
    "E D + tier 3":         (0.5, (0.1, 0.05, 1.0), (0.3, 0.05, 1.0), "absorb"),
    "F twin fix (0.8)":     (0.8, None, None, "ignore"),
}
DISP, AED, HYS = 0.35, 0.175, 1.0


def near(fr, e, thr, win):
    fw, ts, labs = fr
    m = (ts >= e.start - win) & (ts <= e.end + win)
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(e.label)]
    return bool(cols) and bool(m.any()) and float(fw[m][:, cols].max()) >= thr


def stack(cid, fbar, t2, t3, weak):
    b = R.load(R.E.WIN / "beats" / f"{cid}.npz"); f = R.load(R.FLEX / f"{cid}.npz"); p = R.load(R.PANNS / f"{cid}.npz")
    events = _extract_events(b[0], b[1], b[2], AED, None, config.AED_MIN_DUR, low=AED * HYS)
    fev = _extract_events(f[0], f[1], f[2], fbar, None, config.AED_MIN_DUR, low=fbar * HYS)
    key = lambda e: canonical(e.label)
    fresh = []
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if weak == "ignore":
            tw = [x for x in tw if x.confidence >= DISP]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
        else:
            fresh.append(e)
    if t2 and fresh:
        fresh = [e for e in fresh if near((b[0], b[1], b[2]), e, t2[0], t2[2]) or near((p[0], p[1], p[2]), e, t2[1], t2[2])]
    flex_ids = {id(e) for e in fresh}
    events = events + fresh
    fpeak = {}
    for i, lab in enumerate(f[2]):
        fpeak[canonical(lab)] = max(fpeak.get(canonical(lab), 0.0), float(f[0][:, i].max()))
    events = [e for e in events if fpeak.get(key(e), 1.0) >= 0.3]
    ppeak = {}
    for i, lab in enumerate(p[2]):
        ppeak[canonical(lab)] = max(ppeak.get(canonical(lab), 0.0), float(p[0][:, i].max()))
    events = [e for e in events if id(e) not in flex_ids or ppeak.get(key(e), 1.0) >= 0.05]
    if t3:
        for e in events:
            if id(e) not in flex_ids and e.confidence < DISP and (
                    near((f[0], f[1], f[2]), e, t3[0], t3[2]) or near((p[0], p[1], p[2]), e, t3[1], t3[2])):
                e.confidence = DISP
    return [e for e in events if e.confidence >= DISP]


def main():
    cl = [c for c in R.E.clips() if (R.FLEX / f"{c['id']}.npz").exists() and (R.E.WIN / "beats" / f"{c['id']}.npz").exists()
          and (R.PANNS / f"{c['id']}.npz").exists()]
    res = {}
    print(f"{len(cl)} calibration clips")
    for name, (fb, t2, t3, weak) in CELLS.items():
        r = R.score(name, [(c, stack(c["id"], fb, t2, t3, weak)) for c in cl]); res[name] = r
        print(f"{name:22s} conseq onset-recall {r['onset_recall_conseq']:6.1%}  conseq recall {r['conseq_recall']:6.1%}  "
              f"masked {r['masked_conseq_recall']:6.1%} ({r['masked_conseq_n']})  false/min {r['fp_per_min']:5.2f}")
    base = res["shipped (0.8)"]["fp_per_min"]
    ok = {k: v for k, v in res.items() if k != "shipped (0.8)" and v["fp_per_min"] <= base}
    pick = max(ok, key=lambda k: ok[k]["onset_recall_conseq"]) if ok else None
    if pick and ok[pick]["onset_recall_conseq"] <= res["shipped (0.8)"]["onset_recall_conseq"]:
        pick = None                                  # nothing beats the shipped cell at its false-span rate
    print(f"PICK (highest onset-recall at false/min <= shipped {base:.2f}): {pick}")
    out = _ROOT / "benchmark" / "detector_round_stage0.json"
    out.write_text(json.dumps({"clips": len(cl), "cells": res, "pick": pick}, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
