"""Stage-4 benchmark on Adam's gold set alone (amendment 7, 2026-09-22).

Adam: "we MUST improve the detection ... report also per category ... only work with the gold set I
annotated (you can split train/dev but only on what I did, keep variety)". So: every number here
comes from his 139 clips, split once by benchmark/gold/split.py into DEV (79) and TEST (60) with the
four categories, both populations and every sourcing wave represented in each half. Selection on DEV
only; TEST is read once, at the end.

Measured per detector configuration, per category and per half:
  onset recall   share of needed sounds (off screen, importance >= 2) with a same-family detection
                 starting within [-0.5, +1.0] s of the true onset -- the picture metric's own window
  found recall   share of needed sounds the detector names anywhere while they sound (a timing-free
                 upper bound: the gap between the two is the onset problem)
  false labels   detections per clip naming a family that no gold sound of that clip has
  onset error    median |detected onset - true onset| over found sounds

Configurations: any cached frame-wise detector (BEATs, PANNs, FlexSED), at any bar, alone or in
union (each model keeps its own bar; a union is the set of both models' events).

    python benchmark/gold/detector_bench.py --configs beats@0.35 beats@0.15 flexsed@0.3 beats@0.15+flexsed@0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.split import load as load_split

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
CACHES = {"beats": _ROOT / "benchmark" / "gold" / "beats_fw",
          "panns": _ROOT / "benchmark" / "gold" / "panns_fw",
          "flexsed": _ROOT / "data" / "work" / "flexsed_cache",
          "flexsedp": _ROOT / "data" / "work" / "flexsed_prompt_cache"}   # the 3-prompt ensemble
EARLY, LATE = 0.5, 1.0
SNAP = False
MIN_DUR = 0.2


HYST = None         # (rise, fall): onset = the first frame crossing `rise` after being below `fall`
RANKTOP = None      # (low, high): a span peaking below `high` is admitted only if its label is the
                    # strongest of all labels at its own peak frame
LOW = None          # absolute hysteresis floor; None = bar/2 (the pipeline's rule)


def events(stem: str, det: str, bar: float):
    """(label, start, end, peak) spans of one cached detector above `bar`, with hysteresis at bar/2"""
    p = CACHES[det] / f"{stem}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    fw = z["fw"].astype(np.float32)
    labels = [str(x) for x in z["labels"]]
    if "times" in z:
        times = z["times"].astype(np.float64)
    else:                                    # FlexSED: a fixed frame rate, one row per query
        fw = fw.T                            # [T, n_labels]
        times = np.arange(fw.shape[0]) / float(z["fps"])
    low = float(LOW) if LOW is not None else 0.5 * bar
    if HYST:
        low = float(HYST[1])                 # the span starts where the score leaves the quiet band
    out = []
    for j, lab in enumerate(labels):
        v = fw[:, j]
        on = v >= low
        if not on.any():
            continue
        idx = np.flatnonzero(on)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in splits:
            if run.size == 0:
                continue
            peak = float(v[run].max())
            if peak < bar:
                continue
            a, b = float(times[run[0]]), float(times[run[-1]])
            if HYST:
                # the onset is the first frame that rises through `rise`, not the first frame above
                # the display bar: a sound that fades in was being stamped late (Fables, 2026-09-22)
                up = run[np.flatnonzero(v[run] >= float(HYST[0]))]
                if up.size:
                    a = float(times[up[0]])
                    back = run[(run <= up[0])]
                    quiet = back[np.flatnonzero(v[back] < float(HYST[1]))]
                    if quiet.size:
                        a = float(times[min(quiet[-1] + 1, run[-1])])
            if b - a < MIN_DUR:
                b = a + MIN_DUR
            out.append((lab, a, b, peak, int(run[int(np.argmax(v[run]))])))
    if RANKTOP:
        # a weak detection is admitted only if its label is the strongest one at its own peak frame
        # (Fables, 2026-09-22: a rank rule instead of a lower global bar)
        keep = []
        for lab, a, b, peak, f in out:
            if peak >= float(RANKTOP[1]):
                keep.append((lab, a, b, peak)); continue
            j = labels.index(lab)
            if fw[f, j] >= fw[f].max() - 1e-6:
                keep.append((lab, a, b, peak))
        return keep
    return [(lab, a, b, peak) for lab, a, b, peak, _ in out]


_AUDIO = {}


def snap_onsets(stem: str, ev, win: float = 2.0):
    """Move each detection's start to the nearest spectral-flux novelty peak inside it (Fables D1+D2,
    2026-09-22: 14 points of recall are lost to onsets, not to detection). The rule is symmetric --
    it is applied to every detector and to both systems -- and it can only move a start, never add
    or remove a detection."""
    import librosa
    from benchmark.gold.detector_dry import clip_path, wav_for
    if stem not in _AUDIO:
        p = clip_path(stem + ".mp4") or clip_path(stem + ".webm") or clip_path(stem)
        if p is None:
            return ev
        y, sr = librosa.load(str(wav_for(p)), sr=16000)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
        t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=256)
        peaks = librosa.util.peak_pick(env, pre_max=6, post_max=6, pre_avg=12, post_avg=12, delta=float(np.median(env)), wait=8)
        _AUDIO[stem] = t[peaks] if len(peaks) else np.array([])
    pk = _AUDIO[stem]
    if pk.size == 0:
        return ev
    out = []
    for l, a, b, c in ev:
        cand = pk[(pk >= a - 0.2) & (pk <= min(b, a + win))]
        out.append((l, float(cand[0]) if cand.size else a, b, c))
    return out


def parse(cfg: str):
    """'beats@0.15+flexsed@0.3' -> [('beats', 0.15), ('flexsed', 0.3)]"""
    return [(p.split("@")[0], float(p.split("@")[1])) for p in cfg.split("+")]


def depictable(lab: str) -> bool:
    from src.labels import is_salient_nonspeech, is_music
    return is_salient_nonspeech(lab) and not is_music(lab)


def score(gold, stems, cfg):
    per = defaultdict(lambda: {"need": 0, "onset": 0, "found": 0, "fa": 0, "clips": 0, "err": []})
    for stem in stems:
        snds = gold[stem]
        ev = []
        for det, bar in parse(cfg):
            e = events(stem, det, bar)
            if e is None:
                ev = None; break
            e = [x for x in e if depictable(x[0])]
            if SNAP:
                e = snap_onsets(stem, e)
            ev += e
        if ev is None:
            continue
        cat = S.category(snds)
        for key in (cat, "ALL"):
            per[key]["clips"] += 1
        for s in snds:
            if not (s["needed"] and s["importance"] >= 2):
                continue
            hit_on = any(S.same_family(l, s["label"]) and (s["start"] - EARLY <= a <= s["start"] + LATE) for l, a, b, _ in ev)
            near = [a for l, a, b, _ in ev if S.same_family(l, s["label"]) and b > s["start"] - 1 and a < s["end"] + 1]
            for key in (cat, "ALL"):
                per[key]["need"] += 1
                per[key]["onset"] += int(hit_on)
                per[key]["found"] += int(bool(near))
                if near:
                    per[key]["err"].append(min(abs(a - s["start"]) for a in near))
        # a false label: a detected family that no gold sound of this clip has
        fams = {l for l, a, b, _ in ev}
        for l in fams:
            if not any(S.same_family(l, s["label"]) for s in snds):
                for key in (cat, "ALL"):
                    per[key]["fa"] += 1
    return per


def line(name, d):
    if not d["need"]:
        return f"  {name:12s} clips {d['clips']:3d} | needed   0 |                       | false labels/clip {d['fa'] / max(1, d['clips']):.2f}"
    return (f"  {name:12s} clips {d['clips']:3d} | needed {d['need']:3d} | onset-recall {d['onset'] / d['need']:.2f} "
            f"found-recall {d['found'] / d['need']:.2f} | median onset err {np.median(d['err']) if d['err'] else float('nan'):.2f}s "
            f"| false labels/clip {d['fa'] / max(1, d['clips']):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--halves", nargs="+", default=["dev"], help="dev (selection) / test (read once) / all")
    ap.add_argument("--ranktop", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"), help="spans peaking below HIGH must be the top label at their peak frame")
    ap.add_argument("--hyst", nargs=2, type=float, default=None, metavar=("RISE", "FALL"), help="onset = first frame through RISE after being below FALL")
    ap.add_argument("--low", type=float, default=None, help="absolute hysteresis floor for the span (default bar/2): a lower floor starts the span earlier")
    ap.add_argument("--snap", action="store_true", help="re-anchor every detection's onset to the nearest novelty peak inside it")
    ap.add_argument("--out", default=str(_ROOT / "benchmark" / "gold" / "detector_bench.json"))
    a = ap.parse_args()
    config.use_v4("59")
    global SNAP, LOW, HYST
    SNAP = bool(a.snap); LOW = a.low; HYST = tuple(a.hyst) if a.hyst else None
    global RANKTOP
    RANKTOP = tuple(a.ranktop) if a.ranktop else None
    gold = S.load_gold([GOLD])
    dev, test = load_split()
    halves = {"dev": dev, "test": test, "all": set(gold)}
    res = {}
    for half in a.halves:
        stems = sorted(s for s in gold if s in halves[half])
        print(f"===== {half.upper()} ({len(stems)} clips)")
        for cfg in a.configs:
            per = score(gold, stems, cfg)
            if not per:
                print(f"  {cfg}: no cache"); continue
            print(f"[{cfg}]")
            for key in ("ALL", "mixed", "unseen", "seen", "no_ambient"):
                if key in per:
                    print(line(key, per[key]))
            res[f"{half}|{cfg}"] = {k: {kk: (list(map(float, vv)) if kk == "err" else vv) for kk, vv in v.items()} for k, v in per.items()}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    main()
