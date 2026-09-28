"""Week plan B.4 / D6 (docs/WEEK_PLAN_2026-09-26.md): the shipped stage-4 stack on the 280-clip AudioSet-Strong
calibration set. DESCRIPTIVE, NO SELECTION: every bar is the shipped one (set on Adam's DEV gold); the per-family
FlexSED bars fitted on this set were never adopted, so the set is out of sample for the shipped configuration.

Five rows, built from the cached frame scores with stage 4's own rules (src/stage4_audio_event_detection/__init__.py:
span extractor, the twin rule keeping the earlier start, the one-sided FlexSED veto, the PANNs veto on FlexSED-only
spans); the occlusion onset refinement is not re-run here (it needs the audio model's gradients), so onsets are the
extractor's. Scoring is benchmark/audioset_detector_eval.py's (match = same label or family, overlap >= 0.5 s or half
the event; false span = overlaps no labelled event of its family), plus onset-recall in the picture metric's window.

  A  BEATs 0.35              B  FlexSED 0.8 alone          C  union (A + B, twin rule)
  D  C + FlexSED veto 0.3    E  D + PANNs veto 0.05 (the shipped stack)

    python benchmark/audioset_stage4_report.py            # GPU for the PANNs pass (cached), then CPU
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_detector_eval as E
from src.labels import canonical, is_salient_nonspeech, is_music
from src.stage4_audio_event_detection import _extract_events, _infer

FLEX = PANNS = PEF = None


def use_set(name: str):
    """all cache paths follow the set: 'calib' (the 280), 'heldout' (amendment 24) or 'fresh' (the confirmation set,
    docs/prereg_fresh_confirm_set.md: caches only until a candidate passes the 415)"""
    global FLEX, PANNS, PEF
    E.use_set(name)
    FLEX = _ROOT / "data" / "work" / f"flexsed_{name}"
    PANNS = E.WIN / "panns"
    PEF = E.WIN / "pe_frame"


use_set("calib")
OUT = _ROOT / "benchmark" / "audioset_stage4_report.json"
EARLY, LATE = 0.5, 1.0


def panns_cache(device="cuda"):
    PANNS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for c in E.clips():
            dst = PANNS / f"{c['id']}.npz"
            if dst.exists():
                continue
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(E.VIDEOS / f"{c['id']}.mp4"), "-vn", "-ac", "1",
                            "-ar", "32000", str(wav)], check=True)
            fw, times, labels = _infer(wav, device)
            np.savez_compressed(dst, fw=np.asarray(fw, np.float16), times=np.asarray(times, np.float32), labels=np.array(labels))


def load(p, transpose_fps=False):
    z = np.load(p, allow_pickle=False)
    fw = z["fw"].astype(np.float32); labels = [str(x) for x in z["labels"]]
    if "times" in z:
        return fw, z["times"].astype(np.float64), labels
    fw = fw.T
    return fw, np.arange(fw.shape[0]) / float(z["fps"]), labels


def spans(fw, times, labels, bar, low):
    return _extract_events(fw, times, labels, bar, None, config.AED_MIN_DUR, low=low)


def rows_for(cid, fbar=0.8):
    bfw, bt, bl = load(E.WIN / "beats" / f"{cid}.npz")
    beats = spans(bfw, bt, bl, config.DISPLAY_THRESHOLD, config.DISPLAY_THRESHOLD * 0.5)
    ffw, ft, fl = load(FLEX / f"{cid}.npz")
    flex = spans(ffw, ft, fl, fbar, fbar * float(getattr(config, "AED_HYSTERESIS", 1.0)))
    key = lambda e: canonical(e.label)
    union = [dataclasses.replace(e) for e in beats]
    fresh = []
    for e in flex:
        twin = [b for b in union if key(b) == key(e) and b.start - 1.0 <= e.end and e.start - 1.0 <= b.end]
        if twin:
            for b in twin:
                b.start = min(b.start, e.start)
        else:
            fresh.append(e)
    union = union + fresh
    fpeak = {}
    for i, lab in enumerate(fl):
        fpeak[canonical(lab)] = max(fpeak.get(canonical(lab), 0.0), float(ffw[:, i].max()))
    veto1 = [e for e in union if fpeak.get(key(e), 1.0) >= 0.3]
    pfw, _pt, pl = load(PANNS / f"{cid}.npz")
    ppeak = {}
    for i, lab in enumerate(pl):
        ppeak[canonical(lab)] = max(ppeak.get(canonical(lab), 0.0), float(pfw[:, i].max()))
    flex_only = {id(e) for e in fresh}
    veto2 = [e for e in veto1 if id(e) not in flex_only or ppeak.get(key(e), 1.0) >= 0.05]
    return {"A BEATs 0.35": beats, "B FlexSED 0.8": flex, "C union": union, "D + FlexSED veto 0.3": veto1,
            "E + PANNs veto 0.05 (shipped)": veto2}


def score(name, per_clip):
    hits = {"masked_conseq": 0, "conseq": 0, "all": 0}; tot = dict.fromkeys(hits, 0)
    on_hit = on_tot = fp = 0; minutes = 0.0; onset = []
    for c, ev in per_clip:
        ev = [e for e in ev if is_salient_nonspeech(e.label) and not is_music(e.label)]
        minutes += c["duration"] / 60.0
        gold = [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"])]
        for g in gold:
            m = [e for e in ev if E._same(e.label, g["label"]) and E._overlap_ok(e.start, e.end, g["start"], g["end"])]
            for k in ["all"] + (["conseq"] if g["consequential"] else []) + (["masked_conseq"] if g["consequential"] and g["masked"] else []):
                tot[k] += 1; hits[k] += bool(m)
            if g["consequential"]:
                on_tot += 1
                on_hit += any(E._same(e.label, g["label"]) and g["start"] - EARLY <= e.start <= g["start"] + LATE for e in ev)
            if m and g["consequential"]:
                onset.append(min(e.start for e in m) - g["start"])
        for e in ev:
            if not any(E._same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"]):
                fp += 1
    on = np.array(onset)
    return {**{f"{k}_recall": hits[k] / max(1, tot[k]) for k in tot}, **{f"{k}_n": tot[k] for k in tot},
            "onset_recall_conseq": on_hit / max(1, on_tot), "fp_per_min": fp / max(1e-6, minutes),
            "onset_mae": float(np.abs(on).mean()) if len(on) else None}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fbars", nargs="+", type=float, default=[0.8], help="FlexSED bars to report (the detector round's grid)")
    ap.add_argument("--set", default="calib", help="calib (the 280) or heldout (a new, disjoint AudioSet-Strong set)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.set != "calib":
        use_set(a.set)
    panns_cache(a.device)
    cl = [c for c in E.clips() if (FLEX / f"{c['id']}.npz").exists() and (E.WIN / "beats" / f"{c['id']}.npz").exists()]
    per = {}
    for fb in a.fbars:
        for c in cl:
            for name, ev in rows_for(c["id"], fb).items():
                if name.startswith("A "):
                    if fb != a.fbars[0]:
                        continue                      # BEATs alone does not depend on the FlexSED bar
                elif len(a.fbars) > 1 or fb != 0.8:
                    name = f"{name[:1]} FlexSED {fb:g}: " + name[2:]
                per.setdefault(name, []).append((c, ev))
    res = {"clips": len(cl), "note": "descriptive, no selection; shipped bars; onset refinement not re-run", "rows": {}}
    print(f"{len(cl)} AudioSet-Strong calibration clips (10 s each), non-speech non-music events")
    print(f"{'row':32s} {'masked-conseq':>14} {'conseq':>7} {'all':>7} {'onset-recall':>12} {'false/min':>9} {'onset MAE':>9}")
    for name, pc in per.items():
        r = score(name, pc); res["rows"][name] = r
        print(f"{name:32s} {r['masked_conseq_recall']:9.1%} ({r['masked_conseq_n']}) {r['conseq_recall']:7.1%} {r['all_recall']:7.1%} "
              f"{r['onset_recall_conseq']:12.1%} {r['fp_per_min']:9.2f} {r['onset_mae'] if r['onset_mae'] is None else round(r['onset_mae'], 2):>9}")
    out = Path(a.out) if a.out else OUT
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
