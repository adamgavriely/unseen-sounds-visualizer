"""PretrainedSED BEATs-strong as the detector: the declared evaluation (docs/prereg_psed.md).

    python -m benchmark.psed_eval --cache          # GPU, env psed (+ ffmpeg on PATH): DCASE test, dev, benchmark clips
    python -m benchmark.psed_eval --eval           # env msproj

The bar is chosen on DCASE gold at BEATs' false-positive rate (the loosest grid bar with
FP/min <= 5.2), as in the first FLAM attempt; then the bars of the pre-registration.
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

from benchmark.flam_detector import dcase_score, spans, BEATS_BAR, WIN_DIR as BEATS_WIN, MIN_DUR

WIN_DIR = _ROOT / "benchmark" / "psed_windows"
OUT = _ROOT / "benchmark" / "psed_setting.json"
DCASE = _ROOT / "data" / "dcase2025_task3"
GRID = [round(b, 2) for b in np.arange(0.10, 0.96, 0.05)]
BAR = {"masked_recall": 0.095, "clear_recall": 0.326, "fp_per_min": 5.2, "real_kept": 21, "phantoms_gone": 40,
       "onset_mae": 0.35, "onset_within_0.5": 0.80, "onset_matched": 66}


def cache(clip_dir=None):
    from src.stage4_audio_event_detection.psed_infer import cache_clips, CACHE_DIR
    if clip_dir:                       # one folder only (the gold-set shards, slurm/job_gold.sh)
        items = sorted(p for p in Path(clip_dir).iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"))
        n = cache_clips(items, out_dir=CACHE_DIR)
        print(f"[cache] {clip_dir}: {len(items)} clips, {n} scored now -> {CACHE_DIR}", flush=True)
        return
    from benchmark.flam_v2 import eval_stems, EVAL_SPLIT
    from benchmark.gate_dev_sweep import CACHE_DIR as VOTES, _find_clip
    sets = {
        "dcase_test": ([DCASE / "stereo_dev" / EVAL_SPLIT / f"{s}.wav" for s in eval_stems()], WIN_DIR / "dcase_test"),
        "dev": ([_find_clip(f.stem) for f in sorted((VOTES / "dev").glob("*.json"))], WIN_DIR / "dev"),
        "benchmark": (sorted(p for p in (_ROOT / "data" / "input" / "benchmark").rglob("*")
                             if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")), CACHE_DIR),
    }
    for name, (items, out) in sets.items():
        items = [p for p in items if p is not None and Path(p).exists()]
        n = cache_clips(items, out_dir=out)
        print(f"[cache] {name}: {len(items)} clips, {n} scored now -> {out}", flush=True)


def _load(p):
    z = np.load(p, allow_pickle=False)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


def timing(bar: float):
    """Onset error of matched DCASE events under the hysteresis rule (eval_dcase_onset)."""
    import config
    from benchmark.eval_dcase_onset import select_events, nearest_onset
    from src.stage4_audio_event_detection import _extract_events
    chosen, _p, _n = select_events(DCASE, "dev-test-tau", 300, 7)
    low = bar * float(getattr(config, "AED_HYSTERESIS", 0.5))
    errs = []
    cache_ev = {}
    for c, (stem, s, e) in chosen:
        p = WIN_DIR / "dcase_test" / f"{stem}.npz"
        if not p.exists():
            continue
        if stem not in cache_ev:
            fw, times, labels = _load(p)
            cache_ev[stem] = _extract_events(fw, times, labels, bar, None, MIN_DUR, low=low)
        o = nearest_onset(cache_ev[stem], c, s)
        if o is not None:
            errs.append(o - s)
    err = np.array(errs)
    return {"n": int(len(err)), "mean": float(err.mean()) if len(err) else None,
            "mae": float(np.abs(err).mean()) if len(err) else None,
            "within_0.5": float((np.abs(err) <= 0.5).mean()) if len(err) else None}


def evaluate():
    from benchmark.gate_dev_sweep import load
    from src.labels import canonical
    beats = dcase_score(BEATS_WIN / "dcase", BEATS_BAR)
    scores = {b: dcase_score(WIN_DIR / "dcase_test", b) for b in GRID}
    ok = [b for b in GRID if scores[b]["fp_per_min"] <= BAR["fp_per_min"]]
    bar = min(ok) if ok else GRID[-1]
    d = scores[bar]
    print(f"[dcase] BEATs @0.35: masked {beats['masked_recall']:.1%} clear {beats['clear_recall']:.1%} FP/min {beats['fp_per_min']:.1f}")
    for b in GRID:
        s = scores[b]; print(f"[dcase] PSED @{b:.2f}: masked {s['masked_recall']:.1%} clear {s['clear_recall']:.1%} FP/min {s['fp_per_min']:.1f}{'  <- chosen' if b == bar else ''}")
    t = timing(bar)
    lab = json.loads((_ROOT / "benchmark" / "dev_phantoms.json").read_text(encoding="utf-8"))
    sounds = {r["clip"]: r["sounds"] for r in load("dev")}
    real_kept = real_n = ph_gone = ph_n = 0
    for it in lab:
        p = WIN_DIR / "dev" / f"{Path(it['clip']).stem}.npz"
        snd = next((s for s in sounds.get(it["clip"], []) if s["label"] == it["label"]), None)
        if not p.exists() or snd is None:
            continue
        fw, times, labels = _load(p)
        fires = any(canonical(l) == it["label"] and min(b, snd["end"]) - max(a, snd["start"]) > 0
                    for l, a, b, _ in spans(fw, times, labels, bar))
        if it["phantom"]:
            ph_n += 1; ph_gone += (not fires)
        else:
            real_n += 1; real_kept += fires
    passed = (d["masked_recall"] >= BAR["masked_recall"] and d["clear_recall"] >= BAR["clear_recall"]
              and d["fp_per_min"] <= BAR["fp_per_min"] and real_kept >= BAR["real_kept"] and ph_gone >= BAR["phantoms_gone"]
              and t["n"] >= BAR["onset_matched"] and t["mae"] is not None and t["mae"] <= BAR["onset_mae"]
              and t["within_0.5"] >= BAR["onset_within_0.5"])
    out = {"when": datetime.now().isoformat(timespec="minutes"), "model": "PretrainedSED BEATs_strong_1, frame-level",
           "bar_chosen_on_dcase": bar, "beats": beats, "psed": d, "timing": t,
           "grid": {str(b): scores[b] for b in GRID},
           "dev": {"real_kept": [real_kept, real_n], "phantoms_gone": [ph_gone, ph_n]}, "bars": BAR, "passed": bool(passed)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[timing] matched {t['n']} (>= 66)  MAE {t['mae']} (<= 0.35)  within 0.5 s {t['within_0.5']} (>= 0.80)")
    print(f"[dev] real kept {real_kept}/{real_n} (>= 21) | phantoms gone {ph_gone}/{ph_n} (>= 40)")
    print(f"[result] bar {bar:.2f}: masked {d['masked_recall']:.1%} (>= 9.5%) clear {d['clear_recall']:.1%} (>= 32.6%) "
          f"FP/min {d['fp_per_min']:.1f} -> {'PASSED' if passed else 'FAILED'} -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--clip-dir", default=None)
    a = ap.parse_args()
    if a.cache: cache(a.clip_dir)
    if a.eval: evaluate()


if __name__ == "__main__":
    main()
