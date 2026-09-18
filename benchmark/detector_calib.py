"""Choose every detector's bar on the AudioSet-Strong calibration set (docs/prereg_v4.md,
"Calibration on AudioSet-Strong"): the loosest grid bar whose false spans per minute do not
exceed BEATs' at 0.35 on the same clips. Then the slice-B table at those bars.

    python -m benchmark.detector_calib --cache-beats      # env msproj, GPU
    python -m benchmark.detector_calib --cache-psed       # env psed, GPU: the five backbones
    python -m benchmark.detector_calib --choose           # env msproj: bars -> benchmark/detector_calib.json, slice-B table
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_detector_eval as E
from src.stage4_audio_event_detection.psed_infer import BACKBONES, ensemble_of, cache_clips

OUT = _ROOT / "benchmark" / "detector_calib.json"
GRID = [round(b, 2) for b in np.arange(0.05, 0.96, 0.05)]


def cache_psed():
    E.use_set("calib")
    items = [E.VIDEOS / f"{c['id']}.mp4" for c in E.clips()]
    for b in BACKBONES:
        n = cache_clips(items, out_dir=E.WIN / "psed_backbones" / b, backbone=b)
        print(f"[calib] {b}: {len(items)} clips, {n} scored now", flush=True)


def build_ensembles():
    """mean of five for the calibration set; the single PSED-BEATs cache under the evaluator's name"""
    E.use_set("calib")
    out = E.WIN / "psed_ens"; out.mkdir(parents=True, exist_ok=True)
    root = E.WIN / "psed_backbones"
    n = 0
    for f in (root / "BEATs").glob("*.npz"):
        e = ensemble_of(f.stem, root)
        if e is None:
            continue
        fw, t, labels = e
        np.savez_compressed(out / f.name, fw=fw.astype(np.float16), times=t.astype(np.float32), labels=np.array(labels)); n += 1
    dst = E.WIN / "psed"; dst.mkdir(parents=True, exist_ok=True)
    for f in (root / "BEATs").glob("*.npz"):
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
    print(f"[calib] ensemble: {n} clips averaged")


def choose():
    build_ensembles()
    E.use_set("calib")
    ref = E.evaluate_one("beats", 0.35)
    print(f"[calib] BEATs @0.35: masked {ref['masked_conseq_recall']:.1%} all {ref['all_recall']:.1%} FP/min {ref['fp_per_min']:.2f} on {ref['clips']} clips")
    bars = {"beats": 0.35}
    table = {}
    for name in ("psed", "psed_ens"):
        rows = {b: E.evaluate_one(name, b) for b in GRID}
        ok = [b for b in GRID if rows[b]["fp_per_min"] <= ref["fp_per_min"]]
        bars[name] = min(ok) if ok else GRID[-1]
        table[name] = {str(b): {k: rows[b][k] for k in ("masked_conseq_recall", "all_recall", "fp_per_min")} for b in GRID}
        r = rows[bars[name]]
        print(f"[calib] {name:9s} bar {bars[name]:.2f}: masked {r['masked_conseq_recall']:.1%} all {r['all_recall']:.1%} FP/min {r['fp_per_min']:.2f}")
    E.use_set("sliceB")
    sb = {}
    for name, bar in bars.items():
        if not (E.WIN / name).exists():
            continue
        r = E.evaluate_one(name, bar); sb[name] = r
        print(f"[sliceB] {name:9s} @{bar:.2f}: masked {r['masked_conseq_recall']:.1%} conseq {r['conseq_recall']:.1%} all {r['all_recall']:.1%} FP/min {r['fp_per_min']:.2f} onset MAE {r['onset_mae']:.2f}")
    OUT.write_text(json.dumps({"when": datetime.now().isoformat(timespec="minutes"),
                               "rule": "loosest bar with FP/min <= BEATs@0.35 on the calibration set",
                               "calib_clips": ref["clips"], "beats_ref": ref, "bars": bars, "grid": table, "sliceB": sb},
                              indent=1), encoding="utf-8")
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-beats", action="store_true")
    ap.add_argument("--cache-psed", action="store_true")
    ap.add_argument("--choose", action="store_true")
    a = ap.parse_args()
    if a.cache_beats:
        E.use_set("calib"); E.cache("beats")
    if a.cache_psed:
        cache_psed()
    if a.choose:
        choose()


if __name__ == "__main__":
    main()
