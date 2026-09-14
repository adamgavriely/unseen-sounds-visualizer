"""Two-tagger agreement: keep a BEATs detection only if PANNs also hears its family.

Why. On the development split about eighty of the hundred pictures the visibility check
could not silence are confident BEATs detections with no source anywhere (a whale at a
Christmas market 0.36, roaring cats at a helicopter arrival 0.51, an electric toothbrush
in an alarm clip 0.73). A CLAP zero-shot second opinion could not be calibrated (50 %
recall at k = 30 on DCASE gold). PANNs CNN14 is a second AudioSet tagger with a different
architecture and front end, already wired as the alternate stage-4 backend; a label-space
confusion specific to one model should not survive the other. (Four-reviewer round and
chair, 2026-09-15: run this first.)

Rule (declared before running). For a BEATs detection with label L over [start, end],
PANNs' per-frame scores inside [start, end] are collapsed to ontology families (max over
the family's labels and over frames); L's family (or any family in L's ontology subtree,
as in clap_check) must rank within the TOP_K = 10 families, speech and music families
excluded from the competition since they are never shown. Rank, not score, so the two
models' calibrations do not matter.

Pass bar (declared). Dev: at least 40 of the 80 phantoms in benchmark/dev_phantoms.json
removed AND at most 1 of the 23 unseen dev clips loses its picture. DCASE gold: among
the gold events BEATs matched at the display bar (onset eval, hysteresis), at most 2 %
are removed. Pass -> v4 = v3 + filter, re-render/re-judge only test clips where a picture
is removed. Fail -> one paragraph, stop.

    python -m benchmark.panns_agree --ranks dev --ranks test --dcase     # GPU
    python -m benchmark.panns_agree --eval                               # CPU
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import is_salient_nonspeech
from src.stage4_audio_event_detection import clap_check as CC

TOP_K = 10
PHANTOMS = _ROOT / "benchmark" / "dev_phantoms.json"
OUT = _ROOT / "benchmark" / "panns_agree.json"


def _wav_from(video: Path, td: Path) -> Path:
    wav = td / (video.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "32000",
                    str(wav), "-loglevel", "error"], check=True)
    return wav


def _panns(wav: Path):
    from src.stage4_audio_event_detection import _infer
    fw, times, labels = _infer(wav, config.DEVICE)
    fams = [CC.family(l) for l in labels]
    compete = [is_salient_nonspeech(l) for l in labels]
    return fw, times, labels, fams, compete


def _families_of(label: str):
    """Ontology families for a cached label. The gate cache stores CONSOLIDATED names from
    labels.FAMILY ('Gunshot' for 'Gunshot, gunfire', 'Footsteps' for 'Walk, footsteps'), which
    are not ontology nodes; the first run removed every such label with rank 204 because
    no family matched. Map back to every ontology label that consolidates to the name."""
    from src.labels import FAMILY
    members = [k for k, v in FAMILY.items() if v == label] or []
    out = set(CC._subtree_families(label))
    for m in members:
        out.update(CC._subtree_families(m))
    return sorted(out)


def family_rank(fw, times, labels, fams, compete, label: str, start: float, end: float):
    sel = (times >= start) & (times <= max(end, start + 0.5))
    if not sel.any():
        return None
    peak = fw[sel].max(axis=0)                       # per label, max over the span
    scores = {}
    for l, f, c, p in zip(labels, fams, compete, peak):
        if not c:
            continue
        if p > scores.get(f, -1.0):
            scores[f] = float(p)
    mine = max((scores[f] for f in _families_of(label) if f in scores), default=None)
    if mine is None:
        return len(scores) + 1
    return 1 + sum(1 for v in scores.values() if v > mine)


def ranks(split: str):
    from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip
    bar = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
    files = sorted((CACHE_DIR / split).glob("*.json"))
    print(f"[panns] {split}: {len(files)} clips", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(files, 1):
            rec = json.loads(f.read_text(encoding="utf-8"))
            todo = [s for s in rec["sounds"] if s["confidence"] >= bar]
            if not todo:
                continue
            video = _find_clip(rec["clip"])
            if video is None:
                continue
            fw, times, labels, fams, compete = _panns(_wav_from(video, Path(td)))
            for s in todo:
                s["panns_rank"] = family_rank(fw, times, labels, fams, compete,
                                              s["label"], s["start"], s["end"])
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            if i % 25 == 0:
                print(f"[panns] {i}/{len(files)}", flush=True)
    print(f"[panns] {split} done", flush=True)


def dcase():
    """PANNs rank of the gold family for every gold event BEATs matched (onset eval)."""
    from benchmark.eval_dcase_onset import FAMILY, CLASSES
    root = _ROOT / "data" / "dcase2025_task3"
    onset = json.loads((_ROOT / "benchmark" / "eval_dcase_onset.json").read_text(encoding="utf-8"))
    matched = [r for r in onset["records"] if r["err"].get("hyst") is not None]
    cls_idx = {n: i for i, n in enumerate(CLASSES)}
    out = []
    with tempfile.TemporaryDirectory() as td:
        cache = {}
        for r in matched:
            stem = r["key"][0]
            src = root / "stereo_dev" / "dev-test-tau" / f"{stem}.wav"
            if not src.exists():
                continue
            if stem not in cache:
                cache[stem] = _panns(_wav_from(src, Path(td)))
            fw, times, labels, fams, compete = cache[stem]
            c = cls_idx[r["class"]]
            rk = min((family_rank(fw, times, labels, fams, compete, lab, r["gold_start"], r["gold_end"]) or 999)
                     for lab in FAMILY[c])
            out.append({"stem": stem, "class": r["class"], "rank": rk})
    kept = sum(1 for o in out if o["rank"] <= TOP_K)
    print(f"[panns-dcase] BEATs-matched gold events {len(out)}; kept by PANNs top-{TOP_K}: {kept} "
          f"({kept / max(1, len(out)):.0%}); removed {len(out) - kept}")
    (_ROOT / "benchmark" / "panns_agree_dcase.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def evaluate():
    from benchmark.gate_dev_sweep import load, decide, SHOULD_SHOW
    gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    bar, rule, kinds = gate["bar"], gate["rule"], gate["kinds"]
    phantoms = json.loads(PHANTOMS.read_text(encoding="utf-8"))
    ph = {(p["clip"], p["label"]) for p in phantoms if p["phantom"]}
    real = {(p["clip"], p["label"]) for p in phantoms if not p["phantom"]}

    def keep(s):
        return s.get("panns_rank") is None or s["panns_rank"] <= TOP_K

    dev = load("dev")
    ph_removed = ph_seen = real_removed = real_seen = 0
    unseen_lost = unseen_total = 0
    for rec in dev:
        d = decide(rec, bar, rule, kinds)
        for s in rec["sounds"]:
            if s["label"] not in d["shown"]:
                continue
            key = (rec["clip"], s["label"])
            if key in ph:
                ph_seen += 1; ph_removed += (not keep(s))
            elif key in real:
                real_seen += 1; real_removed += (not keep(s))
        if rec["tag"] in SHOULD_SHOW:
            unseen_total += 1
            filt = dict(rec); filt["sounds"] = [s for s in rec["sounds"] if keep(s)]
            if d["show"] and not decide(filt, bar, rule, kinds)["show"]:
                unseen_lost += 1
    print(f"[panns-eval] k={TOP_K}: phantoms removed {ph_removed}/{ph_seen}; "
          f"real-but-unrecognised removed {real_removed}/{real_seen}; "
          f"unseen dev clips losing their picture {unseen_lost}/{unseen_total}")
    passed = ph_removed >= 40 and unseen_lost <= 1
    test = load("test")
    flips = []
    for rec in test:
        d = decide(rec, bar, rule, kinds)
        gone = [(s["label"], s.get("panns_rank")) for s in rec["sounds"]
                if s["label"] in d["shown"] and not keep(s)]
        if gone:
            flips.append({"clip": rec["clip"], "tag": rec["tag"], "removed": gone,
                          "remaining": [l for l in d["shown"] if l not in {g[0] for g in gone}]})
    print(f"[panns-eval] dev bar {'PASSED' if passed else 'FAILED'}; test clips with a picture removed: {len(flips)}")
    for f in flips:
        print(f"   {f['clip']:45s} {f['tag']:15s} removed {f['removed']} -> {f['remaining']}")
    OUT.write_text(json.dumps({"top_k": TOP_K, "phantoms_removed": [ph_removed, ph_seen],
                               "real_removed": [real_removed, real_seen],
                               "unseen_lost": [unseen_lost, unseen_total], "passed": passed,
                               "test_flips": flips}, indent=1), encoding="utf-8")
    print(f"[panns-eval] -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", action="append", choices=("dev", "test"), default=[])
    ap.add_argument("--dcase", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    for sp in a.ranks:
        ranks(sp)
    if a.dcase:
        dcase()
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
