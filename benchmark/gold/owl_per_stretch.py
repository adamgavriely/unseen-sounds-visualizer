"""Would a TIME-ALIGNED object-detector vote help the gate? (2026-09-23)

The concept pass already runs on every clip and its verdict is thrown away. It was tested once as a
second silencing vote and rejected -- it cost three sounds -- but that test was CLIP-level: OWLv2
reports what it finds anywhere in six frames spread over the whole clip, so a car parked in the
first second silenced an ambulance heard in the last.

This asks the question the rejected test could not: per STRETCH, over the same frames the VLM was
shown, does the object detector see the source? The cross-tabulation that decides it is

                        OWL sees it in this stretch     OWL does not
    VLM says visible              agree                  VLM alone
    VLM says NOT visible     <-- the cell that matters       agree

Everything in the bottom-left is a silence a time-aligned OWL vote would ADD. Each one is either a
gate leak removed (worth beta) or a needed sound lost (worth 4), so the rule clears the declared
operating point only if it removes more than two leaks per sound lost.

    python benchmark/gold/owl_per_stretch.py --half dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD, EARLY, LATE


def clip_path(stem):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / f"{stem}.mp4"
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "gold139" / "all").glob(stem + ".*"):
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="dev")
    ap.add_argument("--fps", type=float, default=4.0, help="frames per second inside the stretch")
    ap.add_argument("--bar", type=float, default=0.2, help="OWLv2 score that counts as seeing it")
    a = ap.parse_args()
    config.use_v4("59")
    import torch
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage2_video_understanding.owl import DETECT_QUERY, _load

    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs[a.half]) & set(subs["bench"]))
    root = _ROOT / "data" / "work" / "protocol_proposed_v4b6"
    mdl, proc = _load("google/owlv2-base-patch16-ensemble", "cuda")

    def owl_peak(vp, lab, lo, hi):
        q = DETECT_QUERY.get(lab)
        if q is None:
            return None
        n = max(2, int((hi - lo) * a.fps))
        times = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        best = 0.0
        for img in _sample_frames_at(vp, times):
            inp = proc(text=[[q]], images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = mdl(**inp)
            r = proc.post_process_grounded_object_detection(
                out, threshold=0.05,
                target_sizes=torch.tensor([[img.height, img.width]]).to("cuda"))[0]
            best = max(best, max([float(x) for x in r["scores"]], default=0.0))
        return best

    cell = Counter()
    would_silence = {"needed_hit": [], "leak_or_fa": []}
    for stem in stems:
        f = root / stem / "gate_votes.json"
        if not f.exists():
            continue
        vp = clip_path(stem)
        if vp is None:
            continue
        snds = gold[stem]
        for v in json.loads(f.read_text(encoding="utf-8")):
            lab = v["label"]
            lo, hi = float(v["stretch"][0]), float(v["stretch"][1])
            yes = sum(1 for k in ("name", "ab", "desc") if v.get(k) is True)
            vlm_visible = yes >= 2
            pk = owl_peak(vp, lab, lo - 1.0, hi + 1.0)
            if pk is None:
                continue
            owl_visible = pk >= a.bar
            cell[(vlm_visible, owl_visible)] += 1
            if not vlm_visible and owl_visible:
                # silencing this stretch: does it cost a needed sound, or remove a wrong picture?
                needed = any(S.same_family(lab, s["label"]) and s["needed"] and s["importance"] >= 2
                             and s["start"] - 1.0 <= hi and lo <= s["end"] + 1.0 for s in snds)
                (would_silence["needed_hit"] if needed else would_silence["leak_or_fa"]).append(
                    (stem, lab, round(lo, 1), round(pk, 2)))

    print(f"== {a.half}: {sum(cell.values())} (sound, stretch) decisions, OWL at {a.fps} fps, bar {a.bar}\n")
    print(f"{'':22s} {'OWL sees it':>12s} {'OWL blind':>12s}")
    print(f"{'VLM says visible':22s} {cell[(True, True)]:12d} {cell[(True, False)]:12d}")
    print(f"{'VLM says NOT visible':22s} {cell[(False, True)]:12d} {cell[(False, False)]:12d}")
    nh, nl = len(would_silence["needed_hit"]), len(would_silence["leak_or_fa"])
    print(f"\na time-aligned OWL vote would additionally silence {nh + nl} stretches:")
    print(f"   {nl} where no needed sound is present  -> wrong pictures removed")
    print(f"   {nh} where a NEEDED sound is present   -> sounds lost")
    if nh:
        print(f"   ratio {nl / nh:.2f} removed per sound lost  "
              f"({'CLEARS' if nl / nh > 2 else 'FAILS'} the beta=2 break-even of 2.0)")
    else:
        print("   no needed sound would be lost")
    for k in ("needed_hit", "leak_or_fa"):
        for x in would_silence[k][:8]:
            print(f"     {k:11s} {x[0][:28]:28s} {x[1][:18]:18s} at {x[2]:5.1f}s  owl {x[3]:.2f}")
    out = _ROOT / "benchmark" / "gold" / f"owl_per_stretch_{a.half}.json"
    out.write_text(json.dumps({"cells": {str(k): v for k, v in cell.items()},
                               "would_silence": would_silence}, indent=1), encoding="utf-8")
    print("\n->", out)


if __name__ == "__main__":
    main()
