"""The five-backbone PretrainedSED average as the detector (docs/prereg_psed_ensemble.md).

    python -m benchmark.psed_ensemble --cache        # GPU, env psed: the 4 other backbones on DCASE test, dev, slice B, benchmark
    python -m benchmark.psed_ensemble --eval         # env msproj: build the averages, choose the bar on DCASE, slice-B table, rule

Caches: benchmark/psed_ens_windows/<backbone>/<set>/<stem>.npz for dcase_test, dev, sliceB
(BEATs' own caches are copied in from psed_windows / audioset_windows so all five sit
together); data/work/psed_ens_cache/<backbone>/<stem>.npz for the benchmark clips.
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

from benchmark.flam_detector import dcase_score, spans
from benchmark.psed_eval import GRID, WIN_DIR as PSED_WIN, DCASE, BAR as PSED_BARS, timing as psed_timing
from benchmark import audioset_detector_eval as SB
from src.stage4_audio_event_detection.psed_infer import BACKBONES, ENSEMBLE_CACHE, ensemble_of, cache_clips

WIN = _ROOT / "benchmark" / "psed_ens_windows"
OUT = _ROOT / "benchmark" / "psed_ensemble_setting.json"
# declared before the run (docs/prereg_psed_ensemble.md)
RULE = {"sliceB_masked_recall_min": 0.657, "sliceB_fp_per_min_max": 2.59, "dev_reals_min": 6}


def sets():
    from benchmark.flam_v2 import eval_stems, EVAL_SPLIT
    from benchmark.gate_dev_sweep import CACHE_DIR as VOTES, _find_clip
    return {
        "dcase_test": [DCASE / "stereo_dev" / EVAL_SPLIT / f"{s}.wav" for s in eval_stems()],
        "dev": [_find_clip(f.stem) for f in sorted((VOTES / "dev").glob("*.json"))],
        "sliceB": [SB.VIDEOS / f"{c['id']}.mp4" for c in SB.clips()],
        "benchmark": sorted(p for p in (_ROOT / "data" / "input" / "benchmark").rglob("*")
                            if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")),
    }


def cache():
    for name, items in sets().items():
        items = [p for p in items if p is not None and Path(p).exists()]
        for b in BACKBONES:
            out = (ENSEMBLE_CACHE / b) if name == "benchmark" else (WIN / b / name)
            if b == "BEATs":
                # reuse the existing BEATs-strong caches
                src = {"dcase_test": PSED_WIN / "dcase_test", "dev": PSED_WIN / "dev",
                       "sliceB": SB.WIN / "psed", "benchmark": _ROOT / "data" / "work" / "psed_cache"}[name]
                out.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*.npz"):
                    if not (out / f.name).exists():
                        shutil.copy(f, out / f.name)
                continue
            n = cache_clips(items, out_dir=out, backbone=b)
            print(f"[cache] {name} / {b}: {len(items)} clips, {n} scored now", flush=True)


def build_averages():
    """write the mean-of-five caches next to the per-backbone ones"""
    for name in ("dcase_test", "dev", "sliceB"):
        out = WIN / "ensemble" / name; out.mkdir(parents=True, exist_ok=True)
        stems = [f.stem for f in (WIN / "BEATs" / name).glob("*.npz")]
        n = 0
        for stem in stems:
            # ensemble_of expects root/<backbone>/<stem>.npz; our layout is root/<backbone>/<set>/<stem>.npz
            e = ensemble_of(f"{name}/{stem}", WIN)
            if e is None:
                continue
            fw, t, labels = e
            np.savez_compressed(out / f"{stem}.npz", fw=fw.astype(np.float16), times=t.astype(np.float32), labels=np.array(labels)); n += 1
        print(f"[ensemble] {name}: {n} clips averaged")
    # slice B evaluator reads audioset_windows/<name>/: link the ensemble in
    dst = SB.WIN / "psed_ens"; dst.mkdir(parents=True, exist_ok=True)
    for f in (WIN / "ensemble" / "sliceB").glob("*.npz"):
        shutil.copy(f, dst / f.name)


def evaluate():
    from benchmark.gate_dev_sweep import load
    from src.labels import canonical
    build_averages()
    # 1. the bar, by the same DCASE rule as the single model
    scores = {b: dcase_score(WIN / "ensemble" / "dcase_test", b) for b in GRID}
    ok = [b for b in GRID if scores[b]["fp_per_min"] <= PSED_BARS["fp_per_min"]]
    bar = min(ok) if ok else GRID[-1]
    d = scores[bar]
    print(f"[dcase] ensemble @{bar:.2f}: masked {d['masked_recall']:.1%} clear {d['clear_recall']:.1%} FP/min {d['fp_per_min']:.1f}")
    # 2. slice B at that bar, next to the single model
    single = SB.evaluate_one("psed", json.loads((_ROOT / "benchmark" / "psed_setting.json").read_text(encoding="utf-8"))["bar_chosen_on_dcase"])
    ens = SB.evaluate_one("psed_ens", bar)
    for name, r in (("PSED-BEATs", single), ("ensemble x5", ens)):
        print(f"[sliceB] {name:12s} masked {r['masked_conseq_recall']:.1%} conseq {r['conseq_recall']:.1%} all {r['all_recall']:.1%} FP/min {r['fp_per_min']:.2f} onset MAE {r['onset_mae']:.2f}")
    # 3. dev reals / phantoms
    lab = json.loads((_ROOT / "benchmark" / "dev_phantoms.json").read_text(encoding="utf-8"))
    sounds = {r["clip"]: r["sounds"] for r in load("dev")}
    real_kept = real_n = ph_gone = ph_n = 0
    for it in lab:
        p = WIN / "ensemble" / "dev" / f"{Path(it['clip']).stem}.npz"
        snd = next((s for s in sounds.get(it["clip"], []) if s["label"] == it["label"]), None)
        if not p.exists() or snd is None:
            continue
        z = np.load(p, allow_pickle=False)
        fw, times, labels = z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]
        fires = any(canonical(l) == it["label"] and min(b2, snd["end"]) - max(a, snd["start"]) > 0
                    for l, a, b2, _ in spans(fw, times, labels, bar))
        if it["phantom"]:
            ph_n += 1; ph_gone += (not fires)
        else:
            real_n += 1; real_kept += fires
    passed = (ens["masked_conseq_recall"] >= RULE["sliceB_masked_recall_min"] and ens["fp_per_min"] <= RULE["sliceB_fp_per_min_max"]
              and real_kept >= RULE["dev_reals_min"])
    out = {"when": datetime.now().isoformat(timespec="minutes"), "backbones": list(BACKBONES), "bar_chosen_on_dcase": bar,
           "dcase": d, "sliceB_single": single, "sliceB_ensemble": ens,
           "dev": {"real_kept": [real_kept, real_n], "phantoms_gone": [ph_gone, ph_n]}, "rule": RULE, "passed": bool(passed)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[dev] real kept {real_kept}/{real_n} (>= 6) | phantoms gone {ph_gone}/{ph_n}")
    print(f"[result] {'PASSED' if passed else 'FAILED'} -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.cache: cache()
    if a.eval: evaluate()


if __name__ == "__main__":
    main()
