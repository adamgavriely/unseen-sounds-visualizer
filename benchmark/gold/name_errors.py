"""Why the detector names a sound that is not the one sounding (2026-09-23).

The taxonomy says naming is 61% of our wrong pictures on the adopted detector (v4b6). But
"wrong name" hides three different failures, and they need three different fixes:

  texture      the sound IS there, the annotator just did not think it worth a tag (the traffic
               behind a city walk). The detector hears it all clip long: a flat, sustained score.
  mis-hearing  the sound is not there and could not be (a whale in a street). A brief spike.
  salience     a real, tagged sound of another family at the same moment -- a genuine confusion.

This splits every wrong-named picture by the evidence that produced it: which detector fired
(FlexSED's cached frame scores decide; a label FlexSED never lifts above its bar came from BEATs),
how the score is shaped over the clip (duty cycle = share of frames above the bar; peak), and the
clip's gold category. It also histograms the LATE pictures against the gate's 5-s stretch grid, to
tell an onset error from a gate-scheduling artefact.

    python benchmark/gold/name_errors.py --tag v4b6
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import bucket, GOLD, EARLY, LATE

FX = _ROOT / "data" / "work" / "flexsed_cache"
BAR = 0.8                      # config.V4["0"]["FLEXSED_BAR"]


def fx_scores(stem, label):
    """(duty cycle above the bar, peak, score at each frame) for one family, or None"""
    p = FX / f"{stem}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    labs = [str(x) for x in z["labels"]]
    if label not in labs:
        return None
    v = z["fw"][labs.index(label)].astype(np.float32)
    return float((v >= BAR).mean()), float(v.max()), v, float(z["fps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4b6")
    ap.add_argument("--system", default="proposed")
    ap.add_argument("--subset", default="bench")
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    root = _ROOT / "data" / "work" / f"protocol_{a.system}_{a.tag}"
    rows, lateness = [], []
    for stem in sorted(subs[a.subset]):
        pics = S.load_pictures(root, stem, a.system)
        if pics is None:
            continue
        snds = gold[stem]
        cat = S.category(snds)
        for lab, x, y in pics:
            if any(S.same_family(lab, s["label"]) and S.in_window(x, s["start"], EARLY, LATE)
                   and s["needed"] and s["importance"] >= 2 for s in snds):
                continue
            other = [s for s in snds if not S.same_family(lab, s["label"]) and s["start"] - 1.0 <= y and x <= s["end"] + 1.0]
            bk = bucket(lab, x, y, snds)
            if bk == "invented" and other:
                bk = "wrong-family"
            if bk == "late":
                # how late, and where in the gate's 5-s stretch grid did the onset sit?
                fam = [s for s in snds if S.same_family(lab, s["label"]) and s["needed"]]
                if fam:
                    s0 = min(fam, key=lambda s: abs(s["start"] - x))["start"]
                    lateness.append((stem, lab, round(x - s0, 2), round(s0 % 5.0, 2)))
                continue
            if bk not in ("invented", "wrong-family"):
                continue
            f = fx_scores(stem, lab)
            duty, peak = (f[0], f[1]) if f else (None, None)
            rows.append({"clip": stem, "label": lab, "bucket": bk, "cat": cat, "start": x,
                         "fx_duty": duty, "fx_peak": peak,
                         "detector": ("flexsed+" if (peak or 0) >= BAR else "beats-only") if f else "?"})
    # -- the split
    print(f"== {a.system} {a.tag}: {len(rows)} wrongly-named pictures")
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    print(f"{'label':34s} {'n':>3s} {'bucket':12s} {'duty':>6s} {'peak':>6s}  detector")
    agg = defaultdict(lambda: [0, [], [], Counter(), Counter()])
    for r in rows:
        g = agg[r["label"]]
        g[0] += 1
        if r["fx_duty"] is not None:
            g[1].append(r["fx_duty"]); g[2].append(r["fx_peak"])
        g[3][r["detector"]] += 1
        g[4][r["bucket"]] += 1
    for lab, g in sorted(agg.items(), key=lambda kv: -kv[1][0])[:22]:
        d = f"{np.mean(g[1]):.2f}" if g[1] else "  -"
        p = f"{np.mean(g[2]):.2f}" if g[2] else "  -"
        print(f"{lab[:34]:34s} {g[0]:3d} {g[4].most_common(1)[0][0]:12s} {d:>6s} {p:>6s}  {dict(g[3])}")
    # texture vs spike: a label above the bar for most of the clip is the noise of the place
    tex = [r for r in rows if (r["fx_duty"] or 0) >= 0.5]
    spike = [r for r in rows if r["fx_duty"] is not None and r["fx_duty"] < 0.1]
    print(f"\n  sustained (above bar >=50% of the clip, 'noise of the place'): {len(tex)} / {len(rows)}")
    print(f"  brief      (above bar <10% of the clip, a spike):               {len(spike)} / {len(rows)}")
    print(f"  by detector: {Counter(r['detector'] for r in rows).most_common()}")
    print(f"  by clip category: {Counter(r['cat'] for r in rows).most_common()}")
    # -- the late pictures against the 5-s stretch grid
    if lateness:
        d = np.array([l[2] for l in lateness])
        print(f"\n== {len(lateness)} late pictures: lateness median {np.median(d):.2f}s  "
              f"mean {d.mean():.2f}s  min {d.min():.2f}  max {d.max():.2f}")
        print("   lateness histogram (s):", Counter(int(x) for x in d).most_common())
        print("   onset position inside the 5-s gate stretch:", Counter(int(l[3]) for l in lateness).most_common())
        for l in sorted(lateness, key=lambda t: -t[2])[:8]:
            print(f"     {l[0][:26]:26s} {l[1][:22]:22s} late {l[2]:5.2f}s  onset at {l[3]:4.1f}s into its stretch")
    out = _ROOT / "benchmark" / "gold" / f"name_errors_{a.tag}.json"
    out.write_text(json.dumps({"rows": rows, "late": lateness}, indent=1), encoding="utf-8")
    print("\n->", out)


if __name__ == "__main__":
    main()
