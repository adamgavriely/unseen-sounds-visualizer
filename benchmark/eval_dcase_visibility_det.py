"""Detector-grounded visibility (stage 2) on the DCASE 2025 visibility set: SAM 3 against
OWLv2, the same 258 events and frames the VLM checks used (benchmark/eval_dcase_visibility.py,
seed 7). Dev-only check declared in docs/prereg_v4.md ("stage 2 swap"). Each event: six frames
from [start-1 s, end+1 s]; a concept phrase per DCASE class (below, the same wording style
as owl.DETECT_QUERY); the detector's best presence score over the frames; visible iff score
>= its pipeline bar (SAM 3: 0.5, its own default; OWLv2: config.OWL_THRESHOLD = 0.20).

    python -m benchmark.eval_dcase_visibility_det --backend sam3
    python -m benchmark.eval_dcase_visibility_det --backend owlv2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.eval_dcase_visibility import CLASSES, events_from_csv

# concept phrase per DCASE class: the visible source of that sound (noun phrase, like owl.DETECT_QUERY)
PHRASE = {0: "a person talking", 1: "a person talking", 2: "a person clapping hands", 3: "a telephone",
          4: "a person laughing", 5: "a household appliance", 6: "a person walking", 7: "a door",
          8: "a musical instrument", 9: "a musical instrument", 10: "a water tap", 11: "a bell",
          12: "a door"}


def score_sam3(frames, phrase, device):
    import torch
    from src.stage2_video_understanding.sam3 import _load
    mdl, proc = _load(config.SAM3_MODEL, device)
    best = 0.0
    for img in frames:
        inputs = proc(images=[img], text=[phrase], return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(mdl.dtype)
        with torch.no_grad():
            out = mdl(**inputs)
        res = proc.post_process_instance_segmentation(out, threshold=0.05, mask_threshold=0.5,
                                                      target_sizes=inputs.get("original_sizes").tolist())[0]
        if len(res["scores"]):
            best = max(best, float(res["scores"].max()))
    return best


def score_owl(frames, phrase, device):
    import torch
    from src.stage2_video_understanding.owl import _load
    mdl, proc = _load(config.OWL_MODEL, device)
    best = 0.0
    for img in frames:
        inputs = proc(text=[[phrase]], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs)
        res = proc.post_process_grounded_object_detection(out, threshold=0.05,
                                                          target_sizes=torch.tensor([[img.height, img.width]]).to(device))[0]
        if len(res["scores"]):
            best = max(best, float(res["scores"].max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("sam3", "owlv2"), required=True)
    ap.add_argument("--root", default=str(_ROOT / "data" / "dcase2025_task3"))
    ap.add_argument("--split", default="dev-test-tau")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = _ROOT / "benchmark" / f"eval_dcase_visibility_{a.backend}.json"
    bar = config.SAM3_THRESHOLD if a.backend == "sam3" else config.OWL_THRESHOLD
    scorer = score_sam3 if a.backend == "sam3" else score_owl
    root = Path(a.root)
    metas = sorted((root / "metadata_dev" / a.split).glob("*.csv"))
    random.seed(a.seed); random.shuffle(metas)
    pool = defaultdict(list)
    for m in metas:
        for c, s, e, frac in events_from_csv(m):
            if 0.2 < frac < 0.8:
                continue
            pool[(c, int(frac >= 0.5))].append((m.stem, s, e))
    per_cell = max(3, a.n // (len(CLASSES) * 2))
    chosen = []
    for key, items in pool.items():
        chosen += [(key, x) for x in items[:per_cell]]
    random.shuffle(chosen); chosen = chosen[:a.n]
    # same events as the VLM runs? (keys must match the reference file)
    ref = _ROOT / "benchmark" / "eval_dcase_visibility_q38_direct.json"
    if ref.exists():
        rk = {tuple(r["key"]) for r in json.loads(ref.read_text("utf-8"))["records"]}
        ck = {(stem, c, round(s, 1)) for (c, g), (stem, s, e) in chosen}
        print(f"[dcase-det] events: {len(chosen)}; overlap with the VLM run's events: {len(rk & ck)}/{len(rk)}")

    from src.stage2_video_understanding import _sample_frames_at
    done = {}
    if out.exists():
        done = {tuple(r["key"]): r for r in json.loads(out.read_text("utf-8")).get("records", [])}
    records = list(done.values())
    for (c, gold), (stem, s, e) in chosen:
        key = (stem, c, round(s, 1))
        if key in done:
            continue
        video = root / "video_dev" / a.split / (stem + ".mp4")
        n = 6; lo, hi = max(0.0, s - 1.0), min(5.0, e + 1.0)
        frames = _sample_frames_at(video, [lo + (hi - lo) * k / (n - 1) for k in range(n)])
        sc = scorer(frames, PHRASE[c], config.DEVICE)
        rec = {"key": list(key), "clip": stem, "class": CLASSES[c], "phrase": PHRASE[c], "start": s, "end": e,
               "gold_onscreen": gold, "score": round(sc, 4), "pred_visible": int(sc >= bar)}
        records.append(rec)
        print(f"  {CLASSES[c][:16]:16s} gold={'on ' if gold else 'off'} score={sc:.2f} pred={'vis' if rec['pred_visible'] else 'not'} "
              f"{'OK' if gold == rec['pred_visible'] else '--'}  {stem} {s:.1f}-{e:.1f}", flush=True)
        out.write_text(json.dumps({"records": records}, indent=1), encoding="utf-8")

    n = len(records)
    agree = sum(r["gold_onscreen"] == r["pred_visible"] for r in records)
    tp = sum(r["gold_onscreen"] == 1 and r["pred_visible"] == 1 for r in records)
    fn = sum(r["gold_onscreen"] == 1 and r["pred_visible"] == 0 for r in records)
    fp = sum(r["gold_onscreen"] == 0 and r["pred_visible"] == 1 for r in records)
    tn = sum(r["gold_onscreen"] == 0 and r["pred_visible"] == 0 for r in records)
    per = defaultdict(list)
    for r in records:
        per[r["class"]].append(r["gold_onscreen"] == r["pred_visible"])
    # bar-free view: AUROC of the score against the geometric gold
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score([r["gold_onscreen"] for r in records], [r["score"] for r in records])
    except Exception:
        auroc = None
    summary = {"backend": a.backend, "bar": bar, "n": n, "agreement": agree / n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
               "onscreen_recall": tp / max(1, tp + fn), "offscreen_recall": tn / max(1, tn + fp), "auroc": auroc,
               "per_class": {k: [sum(v), len(v)] for k, v in per.items()}}
    print(f"\n=== {a.backend} visibility vs DCASE 2025 gold ({n} events, bar {bar}) ===")
    print(f"  agreement {agree / n:.1%}   on-screen recall {tp / max(1, tp + fn):.1%}   off-screen recall {tn / max(1, tn + fp):.1%}   AUROC {auroc}")
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
