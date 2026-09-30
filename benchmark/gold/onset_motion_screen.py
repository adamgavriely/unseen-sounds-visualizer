"""Round 31 second idea, OM (onset-motion tiebreak; docs/prereg_round13_detector_push.md "Round 31 OM"), CPU only, on the
saved SHIP5 (= SHIP4+BTP) pictures of merged DEV. A gate stretch with exactly one "yes" among its three votes (name / a-b /
description) is re-decided as "seen" iff the video itself changes at the sound's onset: the mean absolute grey-level
difference between frames 0.25 s apart, averaged over [onset - 0.5, onset + 0.5] s, is >= K times the clip's median
frame-to-frame difference (same spacing, whole clip). The clip verdict stays "silent only if every stretch is seen".
Rescored with score_per_sound as the BTP screen. GO iff 0 current hits lost and wrong <= 31.

    python benchmark/gold/onset_motion_screen.py [--k 3.0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import btp_screen as B
from benchmark.gold import score_per_sound as S
from benchmark.gold import round13_dev as R
from benchmark.gold.subj_screen import spec_of, load_parts
from benchmark.gold.gate_group import video_of

OUT = _ROOT / "benchmark" / "gold" / "onset_motion_screen.json"
CACHE = _ROOT / "benchmark" / "gold" / "onset_motion_cache.json"
STEP, W = 0.25, 160


def grey_frames(video: Path, dur: float):
    """grey frames at STEP spacing over the clip, W px wide (ffmpeg, one pass)"""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={1 / STEP},scale={W}:-1", "-q:v", "3",
                        str(Path(td) / "f%05d.jpg")], check=True, capture_output=True)
        files = sorted(Path(td).glob("f*.jpg"))
        return np.stack([np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0 for f in files]) if files else None


def diffs(frames):
    return np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2))          # d[i] = change between frame i and i+1


def onset_change(d, onset):
    lo, hi = max(0, int((onset - 0.5) / STEP)), int((onset + 0.5) / STEP)
    seg = d[lo:hi + 1]
    return float(seg.mean()) if len(seg) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=3.0)
    a = ap.parse_args()
    P, roots = load_parts()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    rows = []
    for pt, st, g, pics in P:
        for lab, pa, pb, resc in pics:
            spec, votes = spec_of(roots[pt], st, lab, pa, pb)
            if spec is None:
                rows.append({"part": pt, "clip": st, "pic": [lab, pa, pb], "split": [], "flip": False}); continue
            split = [v for v in votes if not v["seen"] and sum(x is True for x in (v["name"], v["ab"], v["desc"])) == 1]
            r = {"part": pt, "clip": st, "pic": [lab, pa, pb], "spec": spec["event_label"], "n_stretch": len(votes),
                 "n_split": len(split), "split": [], "flip": False}
            if split:
                key = st
                if key not in cache:
                    v = video_of(pt, st, roots[pt])
                    fr = grey_frames(v, 0.0) if v else None
                    cache[key] = diffs(fr).tolist() if fr is not None and len(fr) > 1 else []
                    CACHE.write_text(json.dumps(cache), encoding="utf-8")
                d = np.asarray(cache[key], dtype=np.float32)
                med = float(np.median(d)) if len(d) else 0.0
                for v in split:
                    on = float(v["stretch"][0])
                    ch = onset_change(d, on) if len(d) else 0.0
                    ratio = ch / med if med > 0 else 0.0
                    v_seen = ratio >= a.k
                    r["split"].append({"stretch": v["stretch"], "votes": [v["name"], v["ab"], v["desc"]], "named": v["named"],
                                       "onset_change": round(ch, 4), "clip_median": round(med, 4), "ratio": round(ratio, 2), "seen_now": v_seen})
                # the clip verdict: silent only if EVERY stretch is seen (the shipped rule), with the flipped ones
                seen_all = all(v["seen"] or any(s["stretch"] == v["stretch"] and s["seen_now"] for s in r["split"]) for v in votes)
                r["flip"] = bool(votes) and seen_all
            rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r["clip"], []).append(r)
    base = {"dev": [], "dev2": []}; out = {"dev": [], "dev2": []}; lost = []; dropped = []
    for pt, st, g, pics in P:
        rr = by.get(st, [])
        keep = [p[:3] for p, r in zip(pics, rr) if not r["flip"]]
        dropped += [(st, r["pic"][0], round(r["pic"][1], 2)) for r in rr if r["flip"]]
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
        base[pt].append(r0); out[pt].append(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((st, r0["hit"] - r1["hit"]))
    Bm = B.summ(base["dev"] + base["dev2"]); X = B.summ(out["dev"] + out["dev2"])
    go = not lost and X["wrong"] <= 31
    print("BASE", B.fmt(Bm))
    print(f"OM k={a.k}: merged {B.fmt(X)} | DEV {B.fmt(B.summ(out['dev']))} | DEV2 {B.fmt(B.summ(out['dev2']))} hits lost {lost} -> {'GO' if go else 'STOP'}")
    print("   candidates (pictures with a one-yes stretch):", sum(1 for r in rows if r.get("n_split")))
    for r in rows:
        if r.get("n_split"):
            print("  ", r["clip"], r["pic"][0], round(r["pic"][1], 2), "flip" if r["flip"] else "keep", r["split"])
    print("   dropped:", dropped)
    OUT.write_text(json.dumps({"k": a.k, "base": Bm, "OM": X, "dropped": dropped, "hits_lost": lost, "GO": go, "rows": rows}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
