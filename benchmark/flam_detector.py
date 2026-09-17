"""FLAM as the stage-4 detector, evaluated exactly like the seven earlier attempts
(docs/prereg_flam.md, committed before this file).

Queries = the 527 AudioSet label names BEATs uses (taken from a cached BEATs window file), so
the output has the same shape as benchmark/beats_windows: framewise[frames, 527], times,
labels. Windows of 10 s at 48 kHz (FLAM's input); padding masked; score = sigmoid of the
local similarity. Spans use the shipping rule with a bar chosen on DCASE gold at BEATs'
false-positive rate.

    python -m benchmark.flam_detector --cache dcase --cache dev      # GPU, env 'sota'
    python -m benchmark.flam_detector --eval                          # login node, env 'sota' or msproj
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

WIN_DIR = _ROOT / "benchmark" / "beats_windows"
OUT_DIR = _ROOT / "benchmark" / "flam_windows"
OUT = _ROOT / "benchmark" / "flam_setting.json"
SR = 48000
WIN = 480000
BEATS_BAR, MIN_DUR = 0.35, 0.5
BAR = {"masked_recall": 0.245, "clear_recall": 0.326, "fp_per_min": 5.2, "real_kept": 21, "phantoms_gone": 40}
MASKING = {0, 1, 8, 9}


def label_names():
    z = next(WIN_DIR.glob("dev/*.npz"))
    return [str(x) for x in np.load(z, allow_pickle=False)["labels"]]


# ----------------------------------------------------------------------------- cache
def _wav(src: Path, td: Path) -> Path:
    wav = td / (src.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    return wav


def score_file(model, names, wav: Path, batch: int = 64):
    """framewise [frames, 527] over the whole file, times, using FLAM's local similarity."""
    import torch, librosa
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    n = len(audio)
    fws, times = [], []
    with torch.inference_mode():
        for start in range(0, max(1, n), WIN):
            chunk = audio[start:start + WIN]
            valid = len(chunk) / SR
            if valid < 0.5:
                break
            x = torch.from_numpy(np.pad(chunk, (0, WIN - len(chunk)))).float().unsqueeze(0).to("cuda")
            cols = []
            for i in range(0, len(names), batch):
                q = names[i:i + batch]
                sim = model.get_local_similarity(x.repeat(len(q), 1), q, method="unbiased")   # [q, frames]
                cols.append(torch.sigmoid(sim).float().cpu().numpy().T)                        # [frames, q]
            fw = np.concatenate(cols, axis=1)                                                  # [frames, 527]
            T = fw.shape[0]; hop = 10.0 / T
            keep = int(np.ceil(valid / hop))
            fws.append(fw[:keep]); times.append(start / SR + np.arange(keep) * hop)
    return np.concatenate(fws), np.concatenate(times)


def cache(split: str):
    import openflam
    names = label_names()
    model = openflam.OpenFLAM(model_name="v1-base", default_ckpt_path=str(Path.home() / ".cache" / "openflam")).to("cuda").eval()
    out = OUT_DIR / split; out.mkdir(parents=True, exist_ok=True)
    if split == "dcase":
        from benchmark.eval_dcase_onset import select_events
        root = _ROOT / "data" / "dcase2025_task3"
        chosen, _p, _n = select_events(root, "dev-test-tau", 300, 7)
        items = [(s, root / "stereo_dev" / "dev-test-tau" / f"{s}.wav") for s in sorted({c[1][0] for c in chosen})]
    else:
        from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip
        items = [(f.stem, _find_clip(f.stem)) for f in sorted((CACHE_DIR / split).glob("*.json"))]
    print(f"[flam] {split}: {len(items)} clips", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for i, (stem, src) in enumerate(items, 1):
            dst = out / (stem + ".npz")
            if dst.exists() or src is None or not Path(src).exists():
                continue
            fw, times = score_file(model, names, _wav(Path(src), Path(td)))
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=times.astype(np.float32), labels=np.array(names))
            if i % 25 == 0:
                print(f"[flam] {i}/{len(items)}", flush=True)
    print(f"[flam] {split} done", flush=True)


# ----------------------------------------------------------------------------- rule
def spans(fw, times, labels, bar: float):
    """the shipping rule: max >= bar, extended through >= bar/2, at least MIN_DUR."""
    from src.labels import is_salient_nonspeech, is_music
    out = []
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.25
    for c, lab in enumerate(labels):
        if not is_salient_nonspeech(lab) or is_music(lab):
            continue
        s = fw[:, c]
        if s.max() < bar:
            continue
        active = s >= bar / 2
        i, n = 0, len(s)
        while i < n:
            if not active[i]:
                i += 1; continue
            j = i
            while j < n and active[j]:
                j += 1
            if (s[i:j] >= bar).any() and times[j - 1] + hop - times[i] >= MIN_DUR:
                out.append((lab, float(times[i]), float(times[j - 1] + hop), float(s[i:j].max())))
            i = j
    return out


def _load(p: Path):
    z = np.load(p, allow_pickle=False)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


# ----------------------------------------------------------------------------- dcase
def dcase_score(win_dir: Path, bar: float):
    from benchmark.eval_dcase_onset import select_events, FAMILY
    from benchmark.eval_dcase_visibility import events_from_csv
    from src.labels import is_descendant
    root = _ROOT / "data" / "dcase2025_task3"
    chosen, _p, _n = select_events(root, "dev-test-tau", 300, 7)
    by_clip = {}
    for c, (stem, s, e) in chosen:
        by_clip.setdefault(stem, []).append((c, s, e))

    def in_family(label, c):
        return any(label == f or is_descendant(label, f) for f in FAMILY[c])

    def masked(stem, s, e):
        return any(c in MASKING and min(b, e) - max(a, s) > 0
                   for c, a, b, _f in events_from_csv(root / "metadata_dev" / "dev-test-tau" / f"{stem}.csv"))

    hits = {"masked": 0, "clear": 0}; tot = {"masked": 0, "clear": 0}; fp = 0; minutes = 0.0
    for stem, events in by_clip.items():
        p = win_dir / (stem + ".npz")
        if not p.exists():
            continue
        fw, times, labels = _load(p)
        sp = spans(fw, times, labels, bar)
        minutes += (times[-1] - times[0] + (times[1] - times[0] if len(times) > 1 else 0.25)) / 60.0
        for c, s, e in events:
            k = "masked" if masked(stem, s, e) else "clear"; tot[k] += 1
            if any(in_family(l, c) and min(b, e) - max(a, s) >= 0.5 for l, a, b, _ in sp):
                hits[k] += 1
        for l, a, b, _ in sp:
            if not any(in_family(l, c) and min(b, e) - max(a, s) > 0 for c, s, e in events):
                fp += 1
    return {"bar": bar, "masked_recall": hits["masked"] / max(1, tot["masked"]), "masked_n": tot["masked"],
            "clear_recall": hits["clear"] / max(1, tot["clear"]), "clear_n": tot["clear"], "fp_per_min": fp / max(1e-6, minutes)}


def evaluate():
    from benchmark.gate_dev_sweep import load
    from src.labels import canonical
    # 1. the bar: the highest bar whose DCASE false-positive rate is at or below BEATs'
    beats = dcase_score(WIN_DIR / "dcase", BEATS_BAR)
    grid = [round(b, 2) for b in np.arange(0.10, 0.96, 0.05)]
    scores = {b: dcase_score(OUT_DIR / "dcase", b) for b in grid}
    ok = [b for b in grid if scores[b]["fp_per_min"] <= BAR["fp_per_min"]]
    bar = min(ok) if ok else max(grid)           # the loosest bar that still meets the FP rate
    d = scores[bar]
    print(f"[dcase] BEATs @0.35: masked {beats['masked_recall']:.1%} clear {beats['clear_recall']:.1%} FP/min {beats['fp_per_min']:.1f}")
    for b in grid:
        s = scores[b]; print(f"[dcase] FLAM @{b:.2f}: masked {s['masked_recall']:.1%} clear {s['clear_recall']:.1%} FP/min {s['fp_per_min']:.1f}{'  <- chosen' if b == bar else ''}")
    # 2. dev: labelled real / phantom detections
    lab = json.loads((_ROOT / "benchmark" / "dev_phantoms.json").read_text(encoding="utf-8"))
    sounds = {r["clip"]: r["sounds"] for r in load("dev")}
    real_kept = real_n = ph_gone = ph_n = 0
    for it in lab:
        p = OUT_DIR / "dev" / (it["clip"] + ".npz")
        if not p.exists():
            continue
        snd = next((s for s in sounds.get(it["clip"], []) if s["label"] == it["label"]), None)
        if snd is None:
            continue
        fw, times, labels = _load(p)
        fires = any(canonical(l) == it["label"] and min(b, snd["end"]) - max(a, snd["start"]) > 0 for l, a, b, _ in spans(fw, times, labels, bar))
        if it["phantom"]:
            ph_n += 1; ph_gone += (not fires)
        else:
            real_n += 1; real_kept += fires
    passed = (d["masked_recall"] >= BAR["masked_recall"] and d["clear_recall"] >= BAR["clear_recall"] and d["fp_per_min"] <= BAR["fp_per_min"]
              and real_kept >= BAR["real_kept"] and ph_gone >= BAR["phantoms_gone"])
    out = {"when": datetime.now().isoformat(timespec="minutes"), "model": "openflam v1-base, 527 AudioSet names as queries",
           "bar_chosen_on_dcase": bar, "beats": beats, "flam": d, "grid": {str(b): scores[b] for b in grid},
           "dev": {"real_kept": [real_kept, real_n], "phantoms_gone": [ph_gone, ph_n]}, "bars": BAR, "passed": passed}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[dev] real kept {real_kept}/{real_n} (bar >= 21) | phantoms gone {ph_gone}/{ph_n} (bar >= 40)")
    print(f"[result] bar {bar:.2f}: masked {d['masked_recall']:.1%} (>= 24.5%) clear {d['clear_recall']:.1%} (>= 32.6%) FP/min {d['fp_per_min']:.1f} -> {'PASSED' if passed else 'FAILED'} -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", choices=("dev", "test", "dcase"), default=[])
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    for s in a.cache:
        cache(s)
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
