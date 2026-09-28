"""Detectors on gold slice B (AudioSet-Strong eval clips with a consequential sound buried
under speech/music; benchmark/gold/audioset_slice.py). Declared in docs/prereg_v4.md before
any number was seen: no pass bar, one table -- BEATs (v3), PretrainedSED BEATs-strong (v4
candidate) and FLAM-v2 (calibrated, failed attempt) on the same clips, same span rule.

Why this slice: every event in these clips is human-timed, and ALL sounds present are
labelled (AudioSet-Strong is exhaustive per clip), so a span that overlaps no labelled
event of its family is a real false alarm -- the FP criterion DCASE could only approximate.
The eval split is held out from the training of all three detectors.

    python -m benchmark.audioset_detector_eval --cache beats     # env msproj, GPU
    python -m benchmark.audioset_detector_eval --cache psed      # env psed
    python -m benchmark.audioset_detector_eval --cache flam      # env sota
    python -m benchmark.audioset_detector_eval --eval            # env msproj

Measures per detector: recall of consequential events that are masked (the slice's reason
to exist), recall of all consequential events, recall of every non-speech/music event,
false spans per minute, and the onset error of matched events. A detected span matches an
event if its label is the event's label or a descendant/family member and they overlap
>= 0.5 s (or half the event when shorter).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

SLICE = _ROOT / "benchmark" / "gold" / "audioset_slice.json"
VIDEOS = _ROOT / "data" / "input" / "audioset_strong"
WIN = _ROOT / "benchmark" / "audioset_windows"


def use_set(name: str):
    """switch the module to another AudioSet-Strong set: 'sliceB' (default), 'calib' (the 280) or 'heldout'
    (amendment 24's new set). An unknown name is an error, never a silent fall-back to slice B."""
    global SLICE, VIDEOS, WIN, OUT
    if name == "heldout":
        SLICE = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
        VIDEOS = _ROOT / "data" / "input" / "audioset_heldout"
        WIN = _ROOT / "benchmark" / "audioset_heldout_windows"
        OUT = _ROOT / "benchmark" / "audioset_heldout_eval.json"
    elif name == "calib":
        SLICE = _ROOT / "benchmark" / "gold" / "audioset_calib.json"
        VIDEOS = _ROOT / "data" / "input" / "audioset_calib"
        WIN = _ROOT / "benchmark" / "audioset_calib_windows"
        OUT = _ROOT / "benchmark" / "audioset_calib_eval.json"
    elif name == "fresh":             # docs/prereg_fresh_confirm_set.md: read only after a candidate passes the 415
        SLICE = _ROOT / "benchmark" / "gold" / "audioset_fresh.json"
        VIDEOS = _ROOT / "data" / "input" / "audioset_fresh"
        WIN = _ROOT / "benchmark" / "audioset_fresh_windows"
        OUT = _ROOT / "benchmark" / "audioset_fresh_eval.json"
    elif name == "sliceB":
        SLICE = _ROOT / "benchmark" / "gold" / "audioset_slice.json"
        VIDEOS = _ROOT / "data" / "input" / "audioset_strong"
        WIN = _ROOT / "benchmark" / "audioset_windows"
        OUT = _ROOT / "benchmark" / "audioset_detector_eval.json"
    else:
        raise ValueError(f"unknown AudioSet set {name!r}")
OUT = _ROOT / "benchmark" / "audioset_detector_eval.json"
BAR = 0.35            # config.DISPLAY_THRESHOLD; FLAM-v2 is rescaled to it, PSED's bar is its DCASE-chosen one
MIN_DUR = 0.5


def clips():
    d = json.loads(SLICE.read_text(encoding="utf-8"))
    return [c for c in d["clips"] if (VIDEOS / f"{c['id']}.mp4").exists()]


# ----------------------------------------------------------------------------- caches
def cache(which: str):
    items = [VIDEOS / f"{c['id']}.mp4" for c in clips()]
    out = WIN / which; out.mkdir(parents=True, exist_ok=True)
    if which == "beats":
        import subprocess, tempfile
        from src.stage4_audio_event_detection.beats_infer import infer_beats
        with tempfile.TemporaryDirectory() as td:
            n = 0
            for src in items:
                dst = out / f"{src.stem}.npz"
                if dst.exists():
                    continue
                wav = Path(td) / "a.wav"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
                fw, times, labels = infer_beats(wav, "cuda")
                np.savez_compressed(dst, fw=fw.astype(np.float16), times=np.asarray(times, dtype=np.float32), labels=np.array(labels)); n += 1
    elif which == "psed":
        from src.stage4_audio_event_detection.psed_infer import cache_clips
        n = cache_clips(items, out_dir=out)
    elif which == "flam":
        from src.stage4_audio_event_detection.flam_infer import cache_clips
        n = cache_clips(items, out_dir=out)
    print(f"[cache] {which}: {len(items)} clips, {n} scored now -> {out}", flush=True)


# ----------------------------------------------------------------------------- eval
def _load(p):
    z = np.load(p, allow_pickle=False)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


def _spans(fw, times, labels, bar):
    from src.stage4_audio_event_detection import _extract_events
    return _extract_events(fw, times, labels, bar, None, MIN_DUR, low=bar * 0.5)


def _same(label: str, event_label: str) -> bool:
    from src.labels import canonical, is_descendant
    return (label == event_label or canonical(label) == canonical(event_label)
            or is_descendant(label, event_label) or is_descendant(event_label, label))


def _overlap_ok(a, b, s, e):
    return min(b, e) - max(a, s) >= min(0.5, 0.5 * (e - s))


def evaluate_one(which: str, bar: float, calib=None):
    from src.labels import is_salient_nonspeech, is_music
    from src.stage4_audio_event_detection.flam_infer import rescale
    hits = {"masked_conseq": 0, "conseq": 0, "all": 0}; tot = {"masked_conseq": 0, "conseq": 0, "all": 0}
    fp = 0; minutes = 0.0; onset = []
    for c in clips():
        p = WIN / which / f"{c['id']}.npz"
        if not p.exists():
            continue
        fw, times, labels = _load(p)
        if calib:
            fw = rescale(fw, labels, calib, bar)
        ev = [e for e in _spans(fw, times, labels, bar) if is_salient_nonspeech(e.label) and not is_music(e.label)]
        minutes += c["duration"] / 60.0
        gold = [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"])]
        for g in gold:
            m = [e for e in ev if _same(e.label, g["label"]) and _overlap_ok(e.start, e.end, g["start"], g["end"])]
            keys = ["all"] + (["conseq"] if g["consequential"] else []) + (["masked_conseq"] if g["consequential"] and g["masked"] else [])
            for k in keys:
                tot[k] += 1; hits[k] += bool(m)
            if m and g["consequential"]:
                onset.append(min(e.start for e in m) - g["start"])
        for e in ev:
            if not any(_same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"]):
                fp += 1
    on = np.array(onset)
    return {"bar": bar, "clips": sum(1 for c in clips() if (WIN / which / f"{c['id']}.npz").exists()),
            **{f"{k}_recall": hits[k] / max(1, tot[k]) for k in tot}, **{f"{k}_n": tot[k] for k in tot},
            "fp_per_min": fp / max(1e-6, minutes), "onset_mae": float(np.abs(on).mean()) if len(on) else None,
            "onset_within_0.5": float((np.abs(on) <= 0.5).mean()) if len(on) else None, "onset_n": int(len(on))}


def evaluate():
    res = {"when": datetime.now().isoformat(timespec="minutes"), "n_clips": len(clips())}
    res["beats"] = evaluate_one("beats", BAR)
    ps = _ROOT / "benchmark" / "psed_setting.json"
    if (WIN / "psed").exists():
        pbar = json.loads(ps.read_text(encoding="utf-8"))["bar_chosen_on_dcase"] if ps.exists() else BAR
        res["psed"] = evaluate_one("psed", pbar)
    cal = _ROOT / "benchmark" / "flam_calibration.json"
    if (WIN / "flam").exists() and cal.exists():
        res["flam_v2"] = evaluate_one("flam", BAR, json.loads(cal.read_text(encoding="utf-8"))["bars"])
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'detector':10s} {'clips':>5} {'masked-conseq':>14} {'conseq':>8} {'all':>8} {'FP/min':>7} {'onset MAE':>10} {'<=0.5s':>7}")
    for k in ("beats", "psed", "flam_v2"):
        if k in res:
            r = res[k]
            print(f"{k:10s} {r['clips']:5d} {r['masked_conseq_recall']:13.1%} ({r['masked_conseq_n']}) {r['conseq_recall']:7.1%} {r['all_recall']:7.1%} "
                  f"{r['fp_per_min']:7.2f} {str(round(r['onset_mae'], 2)) if r['onset_mae'] is not None else '-':>10} "
                  f"{str(round(r['onset_within_0.5'], 2)) if r['onset_within_0.5'] is not None else '-':>7}")
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", choices=("beats", "psed", "flam"))
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--set", choices=("sliceB", "calib", "heldout", "fresh"), default="sliceB")
    a = ap.parse_args()
    use_set(a.set)
    if a.cache: cache(a.cache)
    if a.eval: evaluate()


if __name__ == "__main__":
    main()
