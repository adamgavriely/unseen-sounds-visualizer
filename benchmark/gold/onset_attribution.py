"""The numbers the reviewers asked for before the onset rule is written up. (2026-09-24)

  1. each recovered hit: did the picture's START move into the window, or did a picture appear that
     was not there before (a gate or span change, not the onset rule itself)?
  2. which false alarms appeared or disappeared, per clip;
  3. whether any drawn start on a clip-start sound (annotator onset <= 0.5 s) moved;
  4. the largest LATER shift the rule made;
  5. reviewer C's primary: on the pre-gate set (needed sounds after 0.5 s with a same-family detector
     event), detector onsets recovered vs lost.

    python benchmark/gold/onset_attribution.py --base dev_repro_v31 --arm dev_mono_v31
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    g = S.load_gold([GOLD])
    dev = sorted(S.subsets_of(g)["dev"])
    W = lambda t: _ROOT / "data" / "work" / f"protocol_proposed_{t}"
    pics = lambda t, s: S.load_pictures(W(t), s, "proposed") or []

    def events(t, s):
        f = W(t) / s / "events.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

    def hits(ps, snd):
        return [p for p in ps if S.same_family(p[0], snd["label"]) and S.in_window(p[1], snd["start"], S.EARLY, S.LATE)]

    needed = lambda snd: snd["needed"] and snd["importance"] >= 2

    print("1. recovered hits, by mechanism")
    for s in dev:
        for snd in g[s]:
            if not needed(snd):
                continue
            hb, hm = hits(pics(a.base, s), snd), hits(pics(a.arm, s), snd)
            if hm and not hb:
                fam = [round(p[1], 2) for p in pics(a.base, s) if S.same_family(p[0], snd["label"])]
                near = [x for x in fam if abs(x - snd["start"]) < 3.0]
                how = "START MOVED into the window" if near else "NEW PICTURE (not a start that moved)"
                print(f"   {s[:28]:28s} {snd['label'][:16]:16s} at {snd['start']:5.1f}  base pictures of the family "
                      f"{fam} -> now {round(hm[0][1], 2)}   => {how}")

    fa = lambda t, s: (lambda r: r["visible"] + r["cross"] + r["phantom"])(S.score_clip(g[s], pics(t, s)))
    ch = [(s, fa(a.base, s), fa(a.arm, s)) for s in dev if fa(a.base, s) != fa(a.arm, s)]
    print(f"\n2. clips whose false-alarm count changed: {len(ch)}")
    for s, x, y in ch:
        print(f"   {s[:34]:34s} {x} -> {y}   base {[(p[0][:14], round(p[1], 1)) for p in pics(a.base, s)]}"
              f"   now {[(p[0][:14], round(p[1], 1)) for p in pics(a.arm, s)]}")

    def onsets(t):
        o = {}
        for s in dev:
            f = W(t) / s / "augmentations.json"
            if not f.exists():
                continue
            for sp in json.loads(f.read_text(encoding="utf-8")):
                if sp.get("augment"):
                    o[(s, sp["event_label"])] = min(x[0] for x in (sp.get("spans") or [[sp["start"], sp["end"]]]))
        return o
    ob, om = onsets(a.base), onsets(a.arm)
    moved = [(k, ob[k], om[k]) for k in set(ob) & set(om) if abs(ob[k] - om[k]) > 0.02]
    clip_start = [(k, x, y) for k, x, y in moved
                  if any(snd["start"] <= 0.5 and S.same_family(snd["label"], k[1]) and needed(snd) for snd in g[k[0]])
                  and x <= 1.0]
    print(f"\n3. drawn starts that moved: {len(moved)}; on a clip-start sound (annotator onset <= 0.5 s): "
          f"{[(k[0][:22], k[1], round(x, 2), round(y, 2)) for k, x, y in clip_start] or 'none'}")
    if moved:
        k, x, y = max(moved, key=lambda m: m[2] - m[1])
        print(f"4. largest later shift: {y - x:+.2f} s  ({k[0]}, {k[1]}: {x:.2f} -> {y:.2f})")

    rec = lost = n = 0
    for s in dev:
        eb, em = events(a.base, s), events(a.arm, s)
        for snd in g[s]:
            if not (needed(snd) and snd["start"] > 0.5):
                continue
            if not any(S.same_family(e["label"], snd["label"]) for e in eb + em):
                continue
            n += 1
            hb = any(S.same_family(e["label"], snd["label"]) and S.in_window(e["start"], snd["start"], S.EARLY, S.LATE) for e in eb)
            hm = any(S.same_family(e["label"], snd["label"]) and S.in_window(e["start"], snd["start"], S.EARLY, S.LATE) for e in em)
            rec += hm and not hb
            lost += hb and not hm
    print(f"\n5. pre-gate set (detector events before the gate): {n} needed sounds after 0.5 s with a same-family "
          f"event; onsets recovered {rec}, lost {lost}, net {rec - lost:+d}")


if __name__ == "__main__":
    main()
