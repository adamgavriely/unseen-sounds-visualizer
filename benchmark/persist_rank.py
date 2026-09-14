"""Detector decision rule: persistence + within-window rank, instead of max-over-windows.

Why (Fable 3, four-reviewer round, 2026-09-15). The shipping rule fires a class when its
MAX over the 0.25 s-hop windows reaches 0.35. A maximum over ~60 noisy sigmoids is an
extreme-value statistic: one window of "whale" at 0.36 makes a phantom, while rain that
sits at 0.23 in every window under Music 0.81 never fires. Both faults are one rule.

Rule. A class is detected over a run of consecutive windows in which it (i) scores at
least FLOOR and (ii) ranks within the top K of the non-speech, non-music classes of that
window; the run must last at least RUN seconds. Rank neutralises masking (rain at 0.23
wins its window even under Speech 0.84); persistence kills single-window spikes. Same
BEATs, no training, no hand rules: three numbers chosen on DCASE gold, frozen before dev.

Measurement, declared before running. DCASE gold (255 events with onsets): recall = a
gold event has an emitted span of the right family overlapping it by >= 0.5 s; false
positives = emitted non-speech spans in a clip whose family matches no gold event of that
clip, per minute of audio. Grid: RUN in {1.0, 1.5, 2.0} s, K in {2, 3, 4}, FLOOR in
{0.10, 0.15, 0.20}. Pick the cell with the highest recall among those with FP/min at or
below the shipping rule's; it must beat shipping recall by >= 10 points or the rule is
dropped. Dev (judge-visible axis only): >= 40 of the 77 phantoms in dev_phantoms.json no
longer emitted, <= 1 of the 20 unseen dev clips losing every currently shown sound.

    python -m benchmark.persist_rank --cache dev --cache test --cache dcase   # GPU
    python -m benchmark.persist_rank --sweep --eval                            # CPU
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical, is_salient_nonspeech, is_music

WIN_DIR = _ROOT / "benchmark" / "beats_windows"
SETTING = _ROOT / "benchmark" / "persist_rank_setting.json"
GRID = [(run, k, floor) for run in (1.0, 1.5, 2.0) for k in (2, 3, 4) for floor in (0.10, 0.15, 0.20)]
SHIP_BAR = 0.35
SHIP_LOW = 0.175
MIN_DUR = 0.5
HOP = 0.25


# ----------------------------------------------------------------------------- cache
def _wav(video: Path, td: Path) -> Path:
    wav = td / (video.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    str(wav), "-loglevel", "error"], check=True)
    return wav


def cache(split: str):
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    out = WIN_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    if split == "dcase":
        from benchmark.eval_dcase_onset import select_events
        root = _ROOT / "data" / "dcase2025_task3"
        chosen, _p, _n = select_events(root, "dev-test-tau", 300, 7)
        items = [(s, root / "stereo_dev" / "dev-test-tau" / f"{s}.wav") for s in sorted({c[1][0] for c in chosen})]
    else:
        from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip
        items = [(f.stem, _find_clip(f.stem)) for f in sorted((CACHE_DIR / split).glob("*.json"))]
    print(f"[win] {split}: {len(items)} clips", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for i, (stem, src) in enumerate(items, 1):
            dst = out / (stem + ".npz")
            if dst.exists() or src is None or not Path(src).exists():
                continue
            fw, times, labels = infer_beats(_wav(Path(src), Path(td)), config.DEVICE)
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=times, labels=np.array(labels))
            if i % 25 == 0:
                print(f"[win] {i}/{len(items)}", flush=True)
    print(f"[win] {split} done", flush=True)


# ----------------------------------------------------------------------------- rules
def _load(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["fw"].astype(np.float32), z["times"], list(z["labels"])


def spans_ship(fw, times, labels):
    """The shipping rule: max >= 0.35, extended through >= 0.175, min 0.5 s."""
    out = []
    dt = HOP
    for c, lab in enumerate(labels):
        s = fw[:, c]
        if s.max() < SHIP_BAR:
            continue
        active = s >= SHIP_LOW
        i, n = 0, len(s)
        while i < n:
            if not active[i]:
                i += 1; continue
            j = i
            while j < n and active[j]:
                j += 1
            if (s[i:j] >= SHIP_BAR).any() and times[j - 1] + dt - times[i] >= MIN_DUR:
                out.append((lab, float(times[i]), float(times[j - 1] + dt), float(s[i:j].max())))
            i = j
    return out


def spans_rule(fw, times, labels, run: float, k: int, floor: float):
    comp = np.array([is_salient_nonspeech(l) and not is_music(l) for l in labels])
    fwc = np.where(comp[None, :], fw, -1.0)
    # rank of every class in every window among competitors (1 = best)
    order = np.argsort(-fwc, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(fw.shape[0])[:, None]
    rank[rows, order] = np.arange(fw.shape[1])[None, :] + 1
    ok = (fw >= floor) & (rank <= k) & comp[None, :]
    out = []
    need = int(round(run / HOP))
    for c, lab in enumerate(labels):
        col = ok[:, c]
        i, n = 0, len(col)
        while i < n:
            if not col[i]:
                i += 1; continue
            j = i
            while j < n and col[j]:
                j += 1
            if j - i >= need:
                out.append((lab, float(times[i]), float(times[j - 1] + HOP), float(fw[i:j, c].max())))
            i = j
    return out


def _fam(label: str) -> str:
    return canonical(label)


# ----------------------------------------------------------------------------- dcase
def dcase_score(rule):
    from benchmark.eval_dcase_onset import select_events, FAMILY, CLASSES
    from src.labels import is_descendant
    root = _ROOT / "data" / "dcase2025_task3"
    chosen, _p, _n = select_events(root, "dev-test-tau", 300, 7)
    by_clip = {}
    for c, (stem, s, e) in chosen:
        by_clip.setdefault(stem, []).append((c, s, e))

    def in_family(label, c):
        return any(label == f or is_descendant(label, f) for f in FAMILY[c])

    hits = total = 0; fp = 0; minutes = 0.0
    for stem, events in by_clip.items():
        p = WIN_DIR / "dcase" / (stem + ".npz")
        if not p.exists():
            continue
        fw, times, labels = _load(p)
        spans = rule(fw, times, labels)
        minutes += (times[-1] + HOP) / 60.0 if len(times) else 0
        for c, s, e in events:
            total += 1
            if any(in_family(l, c) and min(b, e) - max(a, s) >= 0.5 for l, a, b, _ in spans):
                hits += 1
        for l, a, b, _ in spans:
            if not any(in_family(l, c) and min(b, e) - max(a, s) > 0 for c, s, e in events):
                fp += 1
    return {"recall": hits / max(1, total), "n": total, "fp_per_min": fp / max(1e-6, minutes)}


def sweep():
    ship = dcase_score(spans_ship)
    print(f"[sweep] shipping rule: recall {ship['recall']:.3f} (n={ship['n']})  FP/min {ship['fp_per_min']:.2f}")
    rows = []
    for run, k, floor in GRID:
        r = dcase_score(lambda fw, t, l: spans_rule(fw, t, l, run, k, floor))
        rows.append({"run": run, "k": k, "floor": floor, **r})
        print(f"   run {run:.1f} k {k} floor {floor:.2f}: recall {r['recall']:.3f}  FP/min {r['fp_per_min']:.2f}")
    ok = [r for r in rows if r["fp_per_min"] <= ship["fp_per_min"] + 1e-9]
    best = max(ok, key=lambda r: r["recall"]) if ok else None
    passed = best is not None and best["recall"] >= ship["recall"] + 0.10
    print(f"[sweep] best at <= shipping FP/min: {best}; DCASE bar (+10 pts recall) {'PASSED' if passed else 'FAILED'}")
    SETTING.write_text(json.dumps({"shipping": ship, "grid": rows, "chosen": best, "dcase_passed": passed,
                                   "frozen_at": datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- dev
def evaluate():
    from benchmark.gate_dev_sweep import load, decide, SHOULD_SHOW
    st = json.loads(SETTING.read_text(encoding="utf-8"))
    ch = st["chosen"]
    if ch is None:
        print("[eval] no cell passed DCASE; nothing to evaluate"); return
    gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    bar, rule, kinds = gate["bar"], gate["rule"], gate["kinds"]
    phantoms = json.loads((_ROOT / "benchmark" / "dev_phantoms.json").read_text(encoding="utf-8"))
    ph = {(p["clip"], p["label"]) for p in phantoms if p["phantom"]}

    def emitted(rec):
        p = WIN_DIR / "dev" / (rec["clip"] + ".npz")
        if not p.exists():
            return None
        fw, times, labels = _load(p)
        return [(_fam(l), a, b) for l, a, b, _ in spans_rule(fw, times, labels, ch["run"], ch["k"], ch["floor"])]

    def still(sound, spans):
        return any(f == sound["label"] and min(b, sound["end"]) - max(a, sound["start"]) > 0 for f, a, b in spans)

    ph_removed = ph_seen = 0; unseen_lost = unseen_total = 0
    for rec in load("dev"):
        sp = emitted(rec)
        if sp is None:
            continue
        d = decide(rec, bar, rule, kinds)
        shown = [s for s in rec["sounds"] if s["label"] in d["shown"]]
        for s in shown:
            if (rec["clip"], s["label"]) in ph:
                ph_seen += 1; ph_removed += (not still(s, sp))
        if rec["tag"] in SHOULD_SHOW and d["show"]:
            unseen_total += 1
            if not any(still(s, sp) for s in shown):
                unseen_lost += 1
    passed = ph_removed >= 40 and unseen_lost <= 1
    print(f"[eval] cell {ch}: phantoms no longer emitted {ph_removed}/{ph_seen}; unseen clips losing every shown sound "
          f"{unseen_lost}/{unseen_total}; dev bar {'PASSED' if passed else 'FAILED'}")
    st["dev"] = {"phantoms_removed": [ph_removed, ph_seen], "unseen_lost": [unseen_lost, unseen_total], "passed": passed}
    SETTING.write_text(json.dumps(st, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", choices=("dev", "test", "dcase"), default=[])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    for sp in a.cache:
        cache(sp)
    if a.sweep:
        sweep()
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
