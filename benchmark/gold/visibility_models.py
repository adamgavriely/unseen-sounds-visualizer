"""Which model best answers OUR visibility question, judged against Adam's own labels? (2026-09-23)

SAM 3 was compared with OWLv2 once before (docs/prereg_v4.md, 2026-09-20) and lost. Adam's objection
is correct and this run exists because of it: that comparison used DCASE clips, where "visible"
means the source is geometrically inside the camera's field of view. That is not the question this
project asks. Ours is the annotator's: *can a viewer see the thing making this sound* -- a car behind
a wall is in frame and not visible; a crowd on screen is not the source of chanting from around the
corner.

So this asks the same three models the same question on OUR clips, against OUR labels:

    ground truth   the annotator's `visible` tick on each gold sound
    OWLv2          best presence score for that sound's concept phrase, in the stretches it sounds
    SAM 3          the same, with the successor model and the same phrases
    the VLM        what the shipping gate already decided, read from gate_votes.json

Reported: agreement with the annotator, and separately the two error directions, because they cost
different things. Saying "visible" when the annotator says not visible SUPPRESSES a picture the
viewer needed. Saying "not visible" when it is visible SHOWS a picture they did not need.

    python benchmark/gold/visibility_models.py --half dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

TAG = "v4b6"


def clip_path(stem):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / f"{stem}.mp4"
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "gold139" / "all").glob(stem + ".*"):
        return p
    return None


def raw_gold():
    """the annotator's own ticks, which load_gold collapses into `needed`"""
    d = json.loads(Path(GOLD).read_text(encoding="utf-8"))
    out = {}
    for c in d.get("clips", []):
        if not isinstance(c, dict) or c.get("bad") or not c.get("done"):
            continue
        out[Path(c["clip"]).stem] = [
            {"label": S.resolve_label(s.get("family") or s.get("label") or ""),
             "start": float(s.get("start") or 0), "end": float(s.get("end") or 0),
             "visible": bool(s.get("visible")), "importance": int(s.get("importance") or 2)}
            for s in c.get("sounds", []) if s.get("start") is not None and s.get("end") is not None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="dev")
    ap.add_argument("--fps", type=float, default=3.0)
    a = ap.parse_args()
    config.use_v4("59")
    import torch
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage2_video_understanding.owl import DETECT_QUERY, _load as load_owl

    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs[a.half]) & set(subs["bench"]))
    raw = raw_gold()
    root = _ROOT / "data" / "work" / f"protocol_proposed_{TAG}"

    owl_mdl, owl_proc = load_owl("google/owlv2-base-patch16-ensemble", "cuda")
    from src.stage2_video_understanding.sam3 import _load as load_sam3
    sam_mdl, sam_proc = load_sam3(config.SAM3_MODEL, "cuda")

    def owl_peak(frames, label):
        q = DETECT_QUERY.get(label)
        if q is None:
            return None
        best = 0.0
        for img in frames:
            inp = owl_proc(text=[[q]], images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = owl_mdl(**inp)
            r = owl_proc.post_process_grounded_object_detection(
                out, threshold=0.05, target_sizes=torch.tensor([[img.height, img.width]]).to("cuda"))[0]
            best = max(best, max([float(x) for x in r["scores"]], default=0.0))
        return best

    def sam_peak(frames, label):
        q = DETECT_QUERY.get(label)
        if q is None:
            return None
        best = 0.0
        for img in frames:
            inp = sam_proc(images=img, text=q, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = sam_mdl(**inp)
            r = sam_proc.post_process_instance_segmentation(
                out, threshold=0.05, mask_threshold=0.5,
                target_sizes=[(img.height, img.width)])[0]
            sc = r.get("scores")
            if sc is not None and len(sc):
                best = max(best, float(max(float(x) for x in sc)))
        return best

    rows = []
    for stem in stems:
        vp = clip_path(stem)
        if vp is None or stem not in raw:
            continue
        f = root / stem / "gate_votes.json"
        V = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        for s in raw[stem]:
            if s["importance"] < 2:
                continue
            lo, hi = s["start"], max(s["end"], s["start"] + 0.5)
            n = max(2, min(8, int((hi - lo) * a.fps)))
            times = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
            frames = _sample_frames_at(vp, times)
            if not frames:
                continue
            o, m = owl_peak(frames, s["label"]), sam_peak(frames, s["label"])
            if o is None or m is None:
                continue
            rec = [v for v in V if v.get("label") == s["label"]]
            vlm = None
            if rec:
                vlm = any(sum(1 for k in ("name", "ab", "desc") if v.get(k) is True) >= 2 for v in rec)
            rows.append({"clip": stem, "label": s["label"], "visible": s["visible"],
                         "owl": o, "sam3": m, "vlm": vlm})

    nv = sum(1 for r in rows if r["visible"])
    print(f"== {a.half}: {len(rows)} gold sounds, {nv} the annotator marked VISIBLE, {len(rows) - nv} not\n")
    print(f"{'model':16s} {'bar':>5s} {'agreement':>10s} {'says visible when it is':>24s} {'says visible when it is NOT':>28s}")

    def report(name, score_of, bars):
        best = None
        for bar in bars:
            tp = sum(1 for r in rows if r["visible"] and score_of(r) >= bar)
            fp = sum(1 for r in rows if not r["visible"] and score_of(r) >= bar)
            tn = sum(1 for r in rows if not r["visible"] and score_of(r) < bar)
            fn = sum(1 for r in rows if r["visible"] and score_of(r) < bar)
            acc = (tp + tn) / max(1, len(rows))
            print(f"{name:16s} {bar:5.2f} {acc:9.1%} {tp:>16d}/{nv:<7d} {fp:>20d}/{len(rows) - nv:<7d}")
            if best is None or acc > best[1]:
                best = (bar, acc)
        return best

    b_owl = report("OWLv2", lambda r: r["owl"], [0.10, 0.20, 0.30, 0.40])
    b_sam = report("SAM 3", lambda r: r["sam3"], [0.30, 0.50, 0.70, 0.80])
    vr = [r for r in rows if r["vlm"] is not None]
    if vr:
        tp = sum(1 for r in vr if r["visible"] and r["vlm"])
        tn = sum(1 for r in vr if not r["visible"] and not r["vlm"])
        nvv = sum(1 for r in vr if r["visible"])
        fp = sum(1 for r in vr if not r["visible"] and r["vlm"])
        print(f"{'the VLM (gate)':16s} {'-':>5s} {(tp + tn) / len(vr):9.1%} {tp:>16d}/{nvv:<7d} "
              f"{fp:>20d}/{len(vr) - nvv:<7d}   ({len(vr)} sounds it voted on)")

    print(f"\nbest OWLv2 {b_owl[1]:.1%} at bar {b_owl[0]}, best SAM 3 {b_sam[1]:.1%} at bar {b_sam[0]}")
    out = _ROOT / "benchmark" / "gold" / f"visibility_models_{a.half}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
