"""Separate first, detect second: BEATs on the Demucs "other" stem instead of the mix.

Why (Adam, 2026-09-15 evening; Fable consulted). Seven of the eleven test clips where the
gated system wrongly stayed silent are masked misses: the ambient sound scored 0.22-0.31
while Speech (0.84) or Music (0.81) owned the window. The earlier deferral only considered
removing vocals. htdemucs yields four stems -- vocals, drums, bass, other -- so scoring the
"other" stem (mix minus vocals, drums, bass) attacks both speech- and music-masking at once.
Known risks, from our own use of Demucs in the independent reference: tonal sounds (sirens,
alarms) are pulled into the vocals stem; "other" still carries melodic instruments; BEATs was
trained on mixtures, so every score shifts and the 0.35 bar is not calibrated for stems.

Rule under test: residual-only. BEATs on the "other" stem, the shipping rule unchanged
(max >= 0.35, extended through >= 0.175, min 0.5 s), speech and music classes ignored. Not
"max over mix and stem": that can only add detections and would add phantoms by construction.

Measurement, declared before the run (no test-set contact):
  DCASE gold (the 255-event sample of eval_dcase_onset, same seed): recall on the gold
    events that OVERLAP a speech, music or instrument gold event in the same clip
    ("masked" events), and on the rest ("clear"); false-positive spans per family. Mix
    numbers come from the cached windows (benchmark/beats_windows/dcase), stem numbers
    from the new cache (benchmark/beats_windows_sep/dcase).
  Dev (benchmark/dev_phantoms.json, 77 phantom / 23 real detections, labelled before any
    filter): how many of the 23 real detections are still emitted on the stem; how many
    of the 77 phantoms are no longer emitted; how many NEW families are emitted on the
    stem that the mix never emitted in that clip.
PASS iff  (a) masked-event recall rises by >= 10 points over the mix,
     and  (b) no family's DCASE false-positive count rises by more than 2,
     and  (c) at most 2 of the 23 real dev detections are lost.
Fail on any one and the method stays future work, reported with these numbers.

    python -m benchmark.separate_detect --cache dcase --cache dev     # GPU (~1-2 h)
    python -m benchmark.separate_detect --eval                        # login node
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
import config
from src.labels import canonical, is_salient_nonspeech, is_music
from benchmark.persist_rank import WIN_DIR, _load, spans_ship, HOP

SEP_DIR = _ROOT / "benchmark" / "beats_windows_sep"
OUT = _ROOT / "benchmark" / "separate_detect.json"
MASKING_CLASSES = {0, 1, 8, 9}        # female speech, male speech, music, musical instrument
BAR = {"masked_recall_gain": 0.10, "fp_per_family_rise": 2, "real_lost": 2}


# ----------------------------------------------------------------------------- cache
def _other_stem(src: Path, td: Path) -> Path:
    """Demucs htdemucs four-stem separation; the 'other' stem at 16 kHz mono."""
    wav = td / (src.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav),
                    "-loglevel", "error"], check=True)
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", config.DEVICE,
                    "-o", str(td / "sep"), str(wav)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    other = td / "sep" / "htdemucs" / wav.stem / "other.wav"
    mono = td / (src.stem + "_other16k.wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(other), "-ac", "1", "-ar", "16000", str(mono),
                    "-loglevel", "error"], check=True)
    return mono


def cache(split: str):
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    out = SEP_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    if split == "dcase":
        from benchmark.eval_dcase_onset import select_events
        root = _ROOT / "data" / "dcase2025_task3"
        chosen, _p, _n = select_events(root, "dev-test-tau", 300, 7)
        items = [(s, root / "stereo_dev" / "dev-test-tau" / f"{s}.wav") for s in sorted({c[1][0] for c in chosen})]
    else:
        from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip
        items = [(f.stem, _find_clip(f.stem)) for f in sorted((CACHE_DIR / split).glob("*.json"))]
    print(f"[sep] {split}: {len(items)} clips", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for i, (stem, src) in enumerate(items, 1):
            dst = out / (stem + ".npz")
            if dst.exists() or src is None or not Path(src).exists():
                continue
            try:
                fw, times, labels = infer_beats(_other_stem(Path(src), Path(td)), config.DEVICE)
            except Exception as e:                       # one bad clip must not kill the cache
                print(f"  ! {stem}: {type(e).__name__}: {e}", flush=True); continue
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=times, labels=np.array(labels))
            if i % 25 == 0:
                print(f"[sep] {i}/{len(items)}", flush=True)
    print(f"[sep] {split} done", flush=True)


# ----------------------------------------------------------------------------- dcase
def _spans(path: Path):
    fw, times, labels = _load(path)
    return [x for x in spans_ship(fw, times, labels) if is_salient_nonspeech(x[0]) and not is_music(x[0])]


def dcase_eval():
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
        for c, a, b, _f in events_from_csv(root / "metadata_dev" / "dev-test-tau" / f"{stem}.csv"):
            if c in MASKING_CLASSES and min(b, e) - max(a, s) > 0:
                return True
        return False

    res = {}
    for name, win in (("mix", WIN_DIR / "dcase"), ("stem", SEP_DIR / "dcase")):
        hits = {"masked": 0, "clear": 0}; total = {"masked": 0, "clear": 0}
        fp = Counter(); clips = 0
        for stem, events in by_clip.items():
            p = win / (stem + ".npz")
            if not p.exists() or not (WIN_DIR / "dcase" / (stem + ".npz")).exists() \
                    or not (SEP_DIR / "dcase" / (stem + ".npz")).exists():
                continue                                   # score both rules on the same clips
            clips += 1
            spans = _spans(p)
            for c, s, e in events:
                k = "masked" if masked(stem, s, e) else "clear"
                total[k] += 1
                if any(in_family(l, c) and min(b, e) - max(a, s) >= 0.5 for l, a, b, _ in spans):
                    hits[k] += 1
            for l, a, b, _ in spans:
                if not any(in_family(l, c) and min(b, e) - max(a, s) > 0 for c, s, e in events):
                    fp[canonical(l)] += 1
        res[name] = {"clips": clips,
                     "masked_recall": hits["masked"] / max(1, total["masked"]), "masked_n": total["masked"],
                     "clear_recall": hits["clear"] / max(1, total["clear"]), "clear_n": total["clear"],
                     "fp_by_family": dict(fp), "fp_total": sum(fp.values())}
    rise = {f: res["stem"]["fp_by_family"].get(f, 0) - res["mix"]["fp_by_family"].get(f, 0)
            for f in set(res["stem"]["fp_by_family"]) | set(res["mix"]["fp_by_family"])}
    res["fp_rise_max"] = max(rise.values()) if rise else 0
    res["fp_rise_by_family"] = {f: r for f, r in rise.items() if r > 0}
    res["masked_gain"] = res["stem"]["masked_recall"] - res["mix"]["masked_recall"]
    return res


# ----------------------------------------------------------------------------- dev
def dev_eval():
    labelled = json.loads((_ROOT / "benchmark" / "dev_phantoms.json").read_text(encoding="utf-8"))
    from benchmark.gate_dev_sweep import load
    spans_by_clip = {}
    sounds_by_clip = {rec["clip"]: rec["sounds"] for rec in load("dev")}

    def spans(clip, win):
        p = win / (clip + ".npz")
        return None if not p.exists() else [(canonical(l), a, b) for l, a, b, _ in _spans(p)]

    def still(lab, start, end, sp):
        return any(f == lab and min(b, end) - max(a, start) > 0 for f, a, b in sp)

    real_kept = real_seen = ph_removed = ph_seen = 0
    new_families = 0; clips = 0
    seen_clips = set()
    for item in labelled:
        sp = spans(item["clip"], SEP_DIR / "dev")
        if sp is None:
            continue
        snd = next((s for s in sounds_by_clip.get(item["clip"], []) if s["label"] == item["label"]), None)
        if snd is None:
            continue
        kept = still(item["label"], snd["start"], snd["end"], sp)
        if item["phantom"]:
            ph_seen += 1; ph_removed += (not kept)
        else:
            real_seen += 1; real_kept += kept
        seen_clips.add(item["clip"])
    for clip in sorted(seen_clips):
        mix = spans(clip, WIN_DIR / "dev"); sep = spans(clip, SEP_DIR / "dev")
        if mix is None or sep is None:
            continue
        clips += 1
        new_families += len({f for f, _, _ in sep} - {f for f, _, _ in mix})
    return {"clips": clips, "real_kept": [real_kept, real_seen], "phantoms_removed": [ph_removed, ph_seen],
            "new_families_on_stem": new_families, "real_lost": real_seen - real_kept}


def evaluate():
    d = dcase_eval(); v = dev_eval()
    passed = (d["masked_gain"] >= BAR["masked_recall_gain"] and d["fp_rise_max"] <= BAR["fp_per_family_rise"]
              and v["real_lost"] <= BAR["real_lost"])
    out = {"when": datetime.now().isoformat(timespec="minutes"), "rule": "BEATs on htdemucs 'other' stem, shipping rule, residual only",
           "bar": BAR, "dcase": d, "dev": v, "passed": passed}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    m, s_ = d["mix"], d["stem"]
    print(f"[dcase] {m['clips']} clips | masked events n={m['masked_n']}: recall mix {m['masked_recall']:.1%} -> stem "
          f"{s_['masked_recall']:.1%} (gain {d['masked_gain']:+.1%}, bar +10 pts) | clear events n={m['clear_n']}: "
          f"{m['clear_recall']:.1%} -> {s_['clear_recall']:.1%} | FP spans {m['fp_total']} -> {s_['fp_total']}, "
          f"largest per-family rise {d['fp_rise_max']} (bar <= 2) {d['fp_rise_by_family']}")
    print(f"[dev] {v['clips']} clips | real detections kept {v['real_kept'][0]}/{v['real_kept'][1]} (bar: lose <= 2) | "
          f"phantoms no longer emitted {v['phantoms_removed'][0]}/{v['phantoms_removed'][1]} | new families on stem "
          f"{v['new_families_on_stem']}")
    print(f"[result] {'PASSED' if passed else 'FAILED'} -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", choices=("dev", "dcase"), default=[])
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    for split in a.cache:
        cache(split)
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
