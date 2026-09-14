"""Calibrate the CLAP second opinion (src/stage4_audio_event_detection/clap_check.py).

Three steps, the first two on a GPU, the third on a laptop:

  --dcase   For the 255 DCASE 2025 gold events of the onset evaluation (same seed), rank
            the gold class's family among CLAP's families for the 2 s window at the gold
            onset. Prints recall@k for k = 1..30 and freezes k = the smallest value with
            recall >= 0.95 into benchmark/clap_setting.json. Gold decides k; nothing from
            the benchmark's own clips does.
  --ranks SPLIT   For every cached gate-vote clip of the split (benchmark/gate_votes/),
            add CLAP's family rank to each sound at or above the display bar.
  --eval    With k frozen, add one row to the dev cost table: the current gate setting
            plus the CLAP filter (missed / redundant / cost, plus how many shown sounds
            the filter removes on seen/no-ambient clips and on unseen clips). Decided on
            dev before test is opened; then lists the test clips whose shown set changes.

    python -m benchmark.calibrate_clap_k --dcase --ranks dev --ranks test    # GPU
    python -m benchmark.calibrate_clap_k --eval                               # CPU
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
from src.stage4_audio_event_detection import clap_check as C

SETTING = _ROOT / "benchmark" / "clap_setting.json"
RECALL_TARGET = 0.95
MAX_K = 30           # of 328 families: above this the check no longer filters anything
KS = list(range(1, 31))


def _load_audio(path: Path):
    import librosa
    return librosa.load(str(path), sr=16000, mono=True)


def _wav_from(video: Path, td: Path) -> Path:
    wav = td / (video.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    str(wav), "-loglevel", "error"], check=True)
    return wav


# ----------------------------------------------------------------------------- dcase
def dcase():
    from benchmark.eval_dcase_onset import select_events, FAMILY, CLASSES
    root = _ROOT / "data" / "dcase2025_task3"
    chosen, _pool, _n = select_events(root, "dev-test-tau", 300, 7)
    ranks = []
    with tempfile.TemporaryDirectory() as td:
        cache = {}
        for i, (c, (stem, s, e)) in enumerate(chosen, 1):
            src = root / "stereo_dev" / "dev-test-tau" / f"{stem}.wav"
            if not src.exists():
                continue
            if stem not in cache:
                cache[stem] = _load_audio(_wav_from(src, Path(td)))
            audio, sr = cache[stem]
            scores = C.family_scores(audio, sr, s, config.DEVICE, window=min(C.MAX_WINDOW, max(2.0, e - s)))
            # the gold class maps to one or more AudioSet labels; the best-ranked family counts
            r = min((C.family_rank(lab, scores) or 999) for lab in FAMILY[c])
            ranks.append({"stem": stem, "class": CLASSES[c], "start": s, "rank": r})
            if i % 50 == 0:
                print(f"[clap-k] {i}/{len(chosen)}", flush=True)
    rk = np.array([r["rank"] for r in ranks])
    recall = {k: float((rk <= k).mean()) for k in KS}
    print("[clap-k] recall@k on DCASE gold (%d events):" % len(rk))
    print("   " + "  ".join(f"k{k}={recall[k]:.2f}" for k in KS if k <= 15 or k % 5 == 0))
    k = next((k for k in KS if recall[k] >= RECALL_TARGET), None)
    print(f"[clap-k] smallest k with recall >= {RECALL_TARGET:.0%}: {k}  (must be <= {MAX_K}, declared before the run)")
    if k is not None and k > MAX_K:
        k = None
    per = {}
    for r in ranks:
        per.setdefault(r["class"], []).append(r["rank"])
    for cl, v in sorted(per.items()):
        print(f"   {cl[:22]:22s} n={len(v):3d} median rank {np.median(v):.0f}  "
              f"recall@{k}={np.mean(np.array(v) <= (k or 999)):.2f}")
    SETTING.write_text(json.dumps({"top_k": k, "recall_target": RECALL_TARGET, "recall_at_k": recall,
                                   "n_events": int(len(rk)), "per_class_median_rank":
                                   {cl: float(np.median(v)) for cl, v in per.items()},
                                   "frozen_at": datetime.now().isoformat(timespec="seconds"),
                                   "events": ranks}, indent=1), encoding="utf-8")
    print(f"[clap-k] frozen -> {SETTING}")


# ----------------------------------------------------------------------------- ranks
def ranks(split: str):
    from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip
    bar = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
    files = sorted((CACHE_DIR / split).glob("*.json"))
    print(f"[clap-ranks] {split}: {len(files)} clips", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(files, 1):
            rec = json.loads(f.read_text(encoding="utf-8"))
            todo = [s for s in rec["sounds"] if s["confidence"] >= bar]   # recomputed: subtree + burst window
            if not todo:
                continue
            video = _find_clip(rec["clip"])
            if video is None:
                continue
            audio, sr = _load_audio(_wav_from(video, Path(td)))
            for s in todo:
                scores = C.family_scores(audio, sr, s["start"], config.DEVICE,
                                         window=min(C.MAX_WINDOW, max(2.0, s["end"] - s["start"])))
                s["clap_rank"] = C.family_rank(s["label"], scores)
                top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
                s["clap_top3"] = [t[0] for t in top]
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            if i % 25 == 0:
                print(f"[clap-ranks] {i}/{len(files)}", flush=True)
    print(f"[clap-ranks] {split} done", flush=True)


# ----------------------------------------------------------------------------- eval
def evaluate():
    from benchmark.gate_dev_sweep import load, decide, score, SHOULD_SHOW, SILENT_RIGHT
    k = json.loads(SETTING.read_text(encoding="utf-8"))["top_k"]
    gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    bar, rule, kinds = gate["bar"], gate["rule"], gate["kinds"]

    def filtered(rec):
        r = dict(rec)
        r["sounds"] = [s for s in rec["sounds"]
                       if s.get("clap_rank") is None or k is None or s["clap_rank"] <= k]
        return r

    dev = load("dev")
    before = score(dev, bar, rule, kinds)
    after = score([filtered(r) for r in dev], bar, rule, kinds)
    removed_neg = removed_pos = kept_neg = kept_pos = 0
    for rec in dev:
        d = decide(rec, bar, rule, kinds)
        for s in rec["sounds"]:
            if s["label"] not in d["shown"] or s.get("clap_rank") is None:
                continue
            drop = s["clap_rank"] > k
            if rec["tag"] in SILENT_RIGHT:
                removed_neg += drop; kept_neg += (not drop)
            else:
                removed_pos += drop; kept_pos += (not drop)
    print(f"[clap-eval] k={k}; dev cost table row (missed -4 / redundant -1 per clip):")
    for lab, r in (("current", before), ("current + CLAP", after)):
        print(f"   {lab:16s} acc {r['accuracy']:.1%}  cost {r['cost_per_clip']:+.3f}  "
              f"missed {r['missed']}  redundant {r['redundant']}")
    print(f"   shown sounds removed by CLAP: {removed_neg}/{removed_neg + kept_neg} on seen/no-ambient clips, "
          f"{removed_pos}/{removed_pos + kept_pos} on unseen clips")
    test = load("test")
    flips = []
    for rec in test:
        a = decide(rec, bar, rule, kinds)
        b = decide(filtered(rec), bar, rule, kinds)
        if set(a["shown"]) != set(b["shown"]):
            flips.append({"clip": rec["clip"], "tag": rec["tag"], "before": a["shown"], "after": b["shown"],
                          "dropped": [(s["label"], s["clap_rank"], s.get("clap_top3")) for s in rec["sounds"]
                                      if s["label"] in a["shown"] and s.get("clap_rank") and s["clap_rank"] > k]})
    print(f"[clap-eval] test clips whose shown set changes: {len(flips)}")
    for f in flips:
        print(f"   {f['clip']:45s} {f['tag']:15s} {f['before']} -> {f['after']}")
    out = _ROOT / "benchmark" / "clap_eval.json"
    out.write_text(json.dumps({"top_k": k, "dev_before": before, "dev_after": after,
                               "removed": {"seen_or_no_ambient": [removed_neg, removed_neg + kept_neg],
                                           "unseen": [removed_pos, removed_pos + kept_pos]},
                               "test_flips": flips}, indent=1), encoding="utf-8")
    print(f"[clap-eval] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcase", action="store_true")
    ap.add_argument("--ranks", action="append", choices=("dev", "test"), default=[])
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.dcase:
        dcase()
    for sp in a.ranks:
        ranks(sp)
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
