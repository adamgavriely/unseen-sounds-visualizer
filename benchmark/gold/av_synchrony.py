"""Does the picture CHANGE at the instant the sound starts? (Adam's audio-visual sync idea, 2026-09-23)

Every visibility experiment so far has asked a model "is the source on screen". Three of them landed
on exactly the break-even line, and the SAM 3 run showed why: a concept detector answers *presence*
("a crowd is visible") when the gate needs *source* ("is the crowd we can hear that crowd, or one
around the corner"). No current model is asked for source.

Synchrony is a different signal entirely, and it is the only one nobody has tried. A source that is
on screen AND making the sound should MOVE when the sound happens: a door swings as it bangs, a
hammer falls as it strikes. A bystander object that merely happens to be in frame should not.

Two design decisions make it a fair test rather than a motion detector:

  * **Relative, not absolute.** The change inside the detected box is divided by the change outside
    it. A panning camera moves everything, so it cancels; only change that is LOCAL to the object
    survives. This is why global frame-difference would not work.
  * **Measured only where the detector already sees the object.** If no box is found there is
    nothing to be synchronous with, and the answer would just repeat the detector's verdict. The
    question here is the one left over: among the stretches where an object IS visible, does
    synchrony tell the real source from the bystander?

The two groups are the ones every other experiment used:

    leaks  the annotator says the source is plainly visible and the gate drew a picture anyway.
           Synchrony should be HIGH -- the visible thing is the source.
    hits   sounds the gate correctly drew because the source is off screen. Any object the detector
           finds is a bystander, so synchrony should be LOW.

Go/no-go, fixed before the run and matching every other visibility amendment: a threshold on the
synchrony score must remove more than two leaks per needed sound lost.

    python benchmark/gold/av_synchrony.py --half dev
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
from benchmark.gold.error_taxonomy import bucket, GOLD, EARLY, LATE
from benchmark.gold.veto_sweep import keep

WIN = 1.5          # seconds either side of the onset
FPS = 10.0         # frames per second inside that window
ONSET_BAND = (-0.3, 0.5)   # "at the sound" means this much around the marked onset


def clip_path(stem):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / f"{stem}.mp4"
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "gold139" / "all").glob(stem + ".*"):
        return p
    return None


def synchrony(frames, times, box, t0):
    """How much more does the object's region change AT the onset than it does either side of it?

    frames: PIL images, evenly spaced; box: (x0, y0, x1, y1) in pixels; t0: the marked onset.
    Returns None when the box is degenerate or there is nothing to compare.
    """
    if len(frames) < 4 or box is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    g = [np.asarray(f.convert("L"), dtype=np.float32) for f in frames]
    h, w = g[0].shape
    x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    inside_n, outside_n = mask.sum(), (~mask).sum()
    if inside_n < 64 or outside_n < 64:
        return None
    ratios, mids = [], []
    for i in range(len(g) - 1):
        d = np.abs(g[i + 1] - g[i])
        ins = float(d[mask].mean())
        out = float(d[~mask].mean())
        # dividing by the background is what makes a panning camera cancel out
        ratios.append(ins / (out + 1e-3))
        mids.append(0.5 * (times[i] + times[i + 1]))
    ratios, mids = np.array(ratios), np.array(mids)
    at = (mids >= t0 + ONSET_BAND[0]) & (mids <= t0 + ONSET_BAND[1])
    away = ~at
    if at.sum() < 1 or away.sum() < 2:
        return None
    # the signal: local change at the onset, against this object's own usual level
    return float(ratios[at].max() - np.median(ratios[away]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="dev")
    ap.add_argument("--bar", type=float, default=0.2, help="OWLv2 score needed to have a box at all")
    a = ap.parse_args()
    config.use_v4("59")
    import torch
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage2_video_understanding.owl import DETECT_QUERY, _load

    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    root = _ROOT / "data" / "work" / "protocol_proposed_v4b6"
    mdl, proc = _load("google/owlv2-base-patch16-ensemble", "cuda")

    def best_box(img, label):
        q = DETECT_QUERY.get(label)
        if q is None:
            return None, 0.0
        inp = proc(text=[[q]], images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = mdl(**inp)
        r = proc.post_process_grounded_object_detection(
            out, threshold=0.05,
            target_sizes=torch.tensor([[img.height, img.width]]).to("cuda"))[0]
        if not len(r["scores"]):
            return None, 0.0
        k = int(torch.argmax(r["scores"]))
        return [float(v) for v in r["boxes"][k]], float(r["scores"][k])

    groups = {"leak": [], "hit": []}
    for stem in sorted(set(subs[a.half]) & set(subs["bench"])):
        pics = S.load_pictures(root, stem, "proposed")
        if pics is None:
            continue
        snds = gold[stem]
        for lab, x, y in [p for p in pics if keep(stem, p[0], 0.3, None)]:
            right = any(S.same_family(lab, s["label"]) and S.in_window(x, s["start"], EARLY, LATE)
                        and s["needed"] and s["importance"] >= 2 for s in snds)
            if right:
                groups["hit"].append((stem, lab, x))
                continue
            other = [s for s in snds if not S.same_family(lab, s["label"]) and s["start"] - 1.0 <= y and x <= s["end"] + 1.0]
            bk = bucket(lab, x, y, snds)
            if bk == "invented" and other:
                bk = "wrong-family"
            if bk == "gate-leak":
                groups["leak"].append((stem, lab, x))

    print(f"== {a.half}: {len(groups['leak'])} leaks, {len(groups['hit'])} hits\n")
    res = {"leak": [], "hit": []}
    for g, items in groups.items():
        for stem, lab, t0 in items:
            vp = clip_path(stem)
            if vp is None:
                continue
            n = int(2 * WIN * FPS) + 1
            times = [t0 - WIN + (2 * WIN) * i / (n - 1) for i in range(n)]
            frames = _sample_frames_at(vp, times)
            if len(frames) < 4:
                continue
            box, sc = best_box(frames[len(frames) // 2], lab)
            if box is None or sc < a.bar:
                continue                      # no object to be synchronous with
            v = synchrony(frames, times[:len(frames)], box, t0)
            if v is None:
                continue
            res[g].append({"clip": stem, "label": lab, "onset": round(t0, 2),
                           "owl": round(sc, 2), "sync": round(v, 3)})
            print(f"   [{g:4s}] {stem[:28]:28s} {lab[:16]:16s} owl {sc:.2f}  sync {v:+.3f}")

    L = np.array([r["sync"] for r in res["leak"]]) if res["leak"] else np.array([])
    H = np.array([r["sync"] for r in res["hit"]]) if res["hit"] else np.array([])
    print(f"\nstretches with a visible object to measure: {len(L)} leaks, {len(H)} hits")
    if len(L) and len(H):
        print(f"   leaks (source IS visible):  median {np.median(L):+.3f}  mean {L.mean():+.3f}")
        print(f"   hits  (source is off screen): median {np.median(H):+.3f}  mean {H.mean():+.3f}")
        print(f"\n   {'bar':>7s} {'leaks removed':>14s} {'hits lost':>10s} {'ratio':>8s}")
        best = None
        for bar in np.percentile(np.concatenate([L, H]), [10, 25, 40, 50, 60, 75, 90]):
            nl = int((L >= bar).sum()); nh = int((H >= bar).sum())
            r = (nl / nh) if nh else (float("inf") if nl else 0.0)
            flag = "CLEARS" if r > 2 and nl > 0 else ""
            if nl and (best is None or r > best[1]):
                best = (bar, r, nl, nh)
            print(f"   {bar:7.3f} {nl:14d} {nh:10d} {(f'{r:.2f}' if np.isfinite(r) else 'inf'):>8s}  {flag}")
        ok = best is not None and best[1] > 2
        print(f"\ngo/no-go (a threshold removing >2 leaks per sound lost): "
              f"{'PASS' if ok else 'FAIL -- synchrony does not separate the two'}")
    else:
        print("   not enough measurable stretches to decide")
    out = _ROOT / "benchmark" / "gold" / f"av_synchrony_{a.half}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
