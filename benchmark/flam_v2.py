"""FLAM, second attempt (docs/prereg_v4.md §4): descriptive vocabulary, per-query
calibration on DCASE dev-train-tau, the same five bars as the first attempt on
dev-test-tau and the labelled dev detections.

    python -m benchmark.flam_v2 --fetch                 # login node: dev-train-tau wavs (remotezip)
    python -m benchmark.flam_v2 --cache                  # GPU, env sota: raw scores, three sets
    python -m benchmark.flam_v2 --calibrate --eval       # any env with numpy

Calibration rule (declared before the run): for query q, the lowest bar in
{0.05 .. 0.95} whose spans on the calibration audio that overlap no gold event of a
matching class number <= FP_BUDGET / |Q| per minute; none -> 0.95.
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

from src.stage4_audio_event_detection.flam_queries import LABELS
from benchmark.flam_detector import BAR, MASKING, MIN_DUR

WIN_DIR = _ROOT / "benchmark" / "flam_v2_windows"
CALIB = _ROOT / "benchmark" / "flam_calibration.json"
OUT = _ROOT / "benchmark" / "flam_v2_setting.json"
DCASE = _ROOT / "data" / "dcase2025_task3"
CAL_SPLIT, EVAL_SPLIT = "dev-train-tau", "dev-test-tau"
N_CAL = 300
GRID = [round(b, 2) for b in np.arange(0.05, 0.96, 0.05)]
FP_BUDGET = BAR["fp_per_min"]            # 5.2, BEATs' rate, shared equally by the queries
SHIP_BAR = 0.35                          # config.DISPLAY_THRESHOLD


# ----------------------------------------------------------------------------- sets
def cal_stems():
    """One clip per mix from dev-train-tau, sorted, capped -- never the evaluation split."""
    seen, out = set(), []
    for m in sorted((DCASE / "metadata_dev" / CAL_SPLIT).glob("*.csv")):
        key = m.stem.split("_deg")[0]
        if key in seen:
            continue
        seen.add(key); out.append(m.stem)
        if len(out) >= N_CAL:
            break
    return out


def eval_stems():
    from benchmark.eval_dcase_onset import select_events
    chosen, _p, _n = select_events(DCASE, EVAL_SPLIT, 300, 7)
    return sorted({c[1][0] for c in chosen})


def fetch():
    from benchmark.eval_dcase_onset import fetch_audio
    n = fetch_audio(DCASE, CAL_SPLIT, cal_stems())
    print(f"[fetch] {CAL_SPLIT}: {n} wavs fetched, {len(cal_stems())} wanted")


def cache():
    from src.stage4_audio_event_detection.flam_infer import cache_clips
    from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip
    sets = {
        "dcase_train": [DCASE / "stereo_dev" / CAL_SPLIT / f"{s}.wav" for s in cal_stems()],
        "dcase_test": [DCASE / "stereo_dev" / EVAL_SPLIT / f"{s}.wav" for s in eval_stems()],
        "dev": [_find_clip(f.stem) for f in sorted((CACHE_DIR / "dev").glob("*.json"))],
    }
    for name, items in sets.items():
        items = [p for p in items if p is not None and Path(p).exists()]
        n = cache_clips(items, out_dir=WIN_DIR / name)
        print(f"[cache] {name}: {len(items)} clips, {n} scored now", flush=True)


# ----------------------------------------------------------------------------- rule
def col_spans(s, times, bar):
    """the shipping rule on one column: max >= bar, extended through >= bar/2, >= MIN_DUR."""
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
    out = []
    if s.max() < bar:
        return out
    active = s >= bar / 2
    i, n = 0, len(s)
    while i < n:
        if not active[i]:
            i += 1; continue
        j = i
        while j < n and active[j]:
            j += 1
        if (s[i:j] >= bar).any() and times[j - 1] + hop - times[i] >= MIN_DUR:
            out.append((float(times[i]), float(times[j - 1] + hop)))
        i = j
    return out


def _load(p):
    z = np.load(p, allow_pickle=False)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


def _gold(split, stem):
    from benchmark.eval_dcase_visibility import events_from_csv
    return [(c, s, e) for c, s, e, _f in events_from_csv(DCASE / "metadata_dev" / split / f"{stem}.csv")]


def _in_family(label, c):
    from benchmark.eval_dcase_onset import FAMILY
    from src.labels import is_descendant
    return c in FAMILY and any(label == f or is_descendant(label, f) for f in FAMILY[c])


# ----------------------------------------------------------------------------- calibrate
def calibrate():
    files = sorted((WIN_DIR / "dcase_train").glob("*.npz"))
    fp = {b: np.zeros(len(LABELS)) for b in GRID}
    minutes = 0.0
    for p in files:
        fw, times, labels = _load(p)
        assert labels == LABELS
        gold = _gold(CAL_SPLIT, p.stem)
        hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
        minutes += (times[-1] + hop - times[0]) / 60.0
        for c, lab in enumerate(labels):
            ok = [(s, e) for cl, s, e in gold if _in_family(lab, cl)]
            for b in GRID:
                for a, z in col_spans(fw[:, c], times, b):
                    if not any(min(z, e) - max(a, s) > 0 for s, e in ok):
                        fp[b][c] += 1
    budget = FP_BUDGET / len(LABELS)
    bars, rates = {}, {}
    for c, lab in enumerate(LABELS):
        ok = [b for b in GRID if fp[b][c] / minutes <= budget]
        bars[lab] = min(ok) if ok else GRID[-1]
        rates[lab] = {str(b): round(float(fp[b][c] / minutes), 4) for b in GRID}
    out = {"when": datetime.now().isoformat(timespec="minutes"), "split": CAL_SPLIT, "clips": len(files),
           "minutes": round(minutes, 2), "fp_budget_per_query_per_min": budget, "ship_bar": SHIP_BAR,
           "bars": bars, "fp_per_min_by_bar": rates}
    CALIB.write_text(json.dumps(out, indent=1), encoding="utf-8")
    hi = sum(1 for v in bars.values() if v >= GRID[-1])
    print(f"[calibrate] {len(files)} clips, {minutes:.1f} min, budget {budget:.4f}/min/query; "
          f"median bar {np.median(list(bars.values())):.2f}, {hi} queries at the ceiling -> {CALIB}")
    for lab, b in sorted(bars.items(), key=lambda kv: kv[1]):
        print(f"   {b:.2f}  {lab}")


# ----------------------------------------------------------------------------- evaluate
def evaluate():
    from src.stage4_audio_event_detection.flam_infer import rescale
    from benchmark.gate_dev_sweep import load
    from src.labels import canonical, is_salient_nonspeech, is_music
    calib = json.loads(CALIB.read_text(encoding="utf-8"))["bars"]
    keep = [c for c, l in enumerate(LABELS) if is_salient_nonspeech(l) and not is_music(l)]

    def spans_of(fw, times, labels):
        fw = rescale(fw, labels, calib, SHIP_BAR)
        return [(labels[c], a, z, float(fw[:, c].max())) for c in keep for a, z in col_spans(fw[:, c], times, SHIP_BAR)]

    # 1. DCASE dev-test-tau: masked / clear recall and FP per minute at the calibrated bars
    from benchmark.eval_dcase_onset import select_events
    chosen, _p, _n = select_events(DCASE, EVAL_SPLIT, 300, 7)
    by_clip = {}
    for c, (stem, s, e) in chosen:
        by_clip.setdefault(stem, []).append((c, s, e))
    hits = {"masked": 0, "clear": 0}; tot = {"masked": 0, "clear": 0}; fp = 0; minutes = 0.0
    for stem, events in by_clip.items():
        p = WIN_DIR / "dcase_test" / f"{stem}.npz"
        if not p.exists():
            continue
        fw, times, labels = _load(p)
        sp = spans_of(fw, times, labels)
        hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
        minutes += (times[-1] + hop - times[0]) / 60.0
        all_gold = _gold(EVAL_SPLIT, stem)
        for c, s, e in events:
            k = "masked" if any(cl in MASKING and min(b, e) - max(a, s) > 0 for cl, a, b in all_gold) else "clear"
            tot[k] += 1
            if any(_in_family(l, c) and min(b, e) - max(a, s) >= 0.5 for l, a, b, _ in sp):
                hits[k] += 1
        for l, a, b, _ in sp:
            if not any(_in_family(l, c) and min(b, e) - max(a, s) > 0 for c, s, e in all_gold):
                fp += 1
    d = {"masked_recall": hits["masked"] / max(1, tot["masked"]), "masked_n": tot["masked"],
         "clear_recall": hits["clear"] / max(1, tot["clear"]), "clear_n": tot["clear"],
         "fp_per_min": fp / max(1e-6, minutes), "minutes": minutes, "clips": len(by_clip)}

    # 2. dev: labelled real / phantom detections
    lab = json.loads((_ROOT / "benchmark" / "dev_phantoms.json").read_text(encoding="utf-8"))
    sounds = {r["clip"]: r["sounds"] for r in load("dev")}
    real_kept = real_n = ph_gone = ph_n = 0; no_query = []
    for it in lab:
        p = WIN_DIR / "dev" / f"{it['clip']}.npz"
        snd = next((s for s in sounds.get(it["clip"], []) if s["label"] == it["label"]), None)
        if not p.exists() or snd is None:
            continue
        fw, times, labels = _load(p)
        fires = any(canonical(l) == it["label"] and min(b, snd["end"]) - max(a, snd["start"]) > 0
                    for l, a, b, _ in spans_of(fw, times, labels))
        if it["label"] not in {canonical(l) for l in LABELS}:
            no_query.append(it["label"])
        if it["phantom"]:
            ph_n += 1; ph_gone += (not fires)
        else:
            real_n += 1; real_kept += fires
    passed = (d["masked_recall"] >= BAR["masked_recall"] and d["clear_recall"] >= BAR["clear_recall"]
              and d["fp_per_min"] <= BAR["fp_per_min"] and real_kept >= BAR["real_kept"] and ph_gone >= BAR["phantoms_gone"])
    out = {"when": datetime.now().isoformat(timespec="minutes"), "model": "openflam v1-base, descriptive vocabulary, per-query bars",
           "queries": len(LABELS), "calibration": str(CALIB.name), "dcase_test": d,
           "dev": {"real_kept": [real_kept, real_n], "phantoms_gone": [ph_gone, ph_n],
                   "phantom_labels_without_query": sorted(set(no_query))},
           "bars": BAR, "passed": passed}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[dcase] masked {d['masked_recall']:.1%} (>= 24.5%) clear {d['clear_recall']:.1%} (>= 32.6%) "
          f"FP/min {d['fp_per_min']:.2f} (<= 5.2) over {d['minutes']:.1f} min")
    print(f"[dev] real kept {real_kept}/{real_n} (>= 21) | phantoms gone {ph_gone}/{ph_n} (>= 40); "
          f"phantom labels with no query: {len(set(no_query))}")
    print(f"[result] {'PASSED' if passed else 'FAILED'} -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.fetch: fetch()
    if a.cache: cache()
    if a.calibrate: calibrate()
    if a.eval: evaluate()


if __name__ == "__main__":
    main()
