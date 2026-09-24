"""Compare the timing arms against the run they are meant to improve. (2026-09-24)

One row per run. The headline numbers come from the scorer on the fixed gold set, so the denominator
is the same for every arm whatever the gate decides to draw; the onset error comes from the drawn
specs, which is a different and smaller set and is reported as a distribution, not as a score.

    python benchmark/gold/arm_compare.py --base dev_symgen_v30 \
        --arms dev_cap_v30 dev_mono_v30 dev_monocap_v30 dev_monostrong_v30

`--identical base,arm` additionally checks that two runs produced the SAME starts, which is how we
prove the instrumentation did not change behaviour when its switches are off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

NAMED = ["Siren", "Crowd", "Bird", "Gunshot", "Laughter"]      # the five early cases


def root_of(tag):
    return _ROOT / "data" / "work" / f"protocol_proposed_{tag}"


def scored(tag, gold):
    """the scorer's own per-clip rows, on the fixed gold set"""
    rows, per_sound = [], {}
    for stem, snds in gold.items():
        pics = S.load_pictures(root_of(tag), stem, "proposed")
        if pics is None:
            continue
        r = S.score_clip(snds, pics)
        r["clip"] = stem
        rows.append(r)
        # a per-sound hit flag, so before/after can be paired sound by sound
        for g in snds:
            if not (g["needed"] and g["importance"] >= 2):
                continue
            hit = any(S.same_family(l, g["label"]) and S.in_window(x, g["start"], S.EARLY, S.LATE)
                      for l, x, y in pics)
            per_sound[(stem, g["label"], round(g["start"], 2))] = hit
    return rows, per_sound


def onsets(tag, gold):
    out = {}
    for stem, snds in gold.items():
        f = root_of(tag) / stem / "augmentations.json"
        if not f.exists():
            continue
        for sp in json.loads(f.read_text(encoding="utf-8")):
            if not sp.get("augment"):
                continue
            spans = sp.get("spans") or [[sp["start"], sp["end"]]]
            a, b = min(x[0] for x in spans), max(x[1] for x in spans)
            cand = [s for s in snds if S.same_family(sp["event_label"], s["label"])]
            if not cand:
                continue
            g = min(cand, key=lambda s: abs(s["start"] - a))
            if abs(g["start"] - a) > 6.0:
                continue
            out[(stem, sp["event_label"])] = (a - g["start"], b - g["end"], a, g["start"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--identical", default="", help="tagA,tagB: assert the two produced equal starts")
    ap.add_argument("--subset", default="dev", help="gold subset: dev (default) or test")
    ap.add_argument("--boot", action="store_true", help="paired clip bootstrap (2000 draws, seed 0) vs the base")
    a = ap.parse_args()
    gold = S.load_gold([GOLD])
    dev = set(S.subsets_of(gold)[a.subset])
    gold = {k: v for k, v in gold.items() if k in dev}

    if a.identical:
        x, y = a.identical.split(",")
        ox, oy = onsets(x, gold), onsets(y, gold)
        shared = sorted(set(ox) & set(oy))
        moved = [(k, ox[k][2], oy[k][2]) for k in shared if abs(ox[k][2] - oy[k][2]) > 0.02]
        print(f"identity check {x} vs {y}: {len(shared)} shared spans, {len(moved)} with a different start")
        for k, p, q in moved[:10]:
            print(f"   {k[0][:28]:28s} {k[1][:14]:14s} {p:6.2f} -> {q:6.2f}")
        print()

    base_rows, base_hits = scored(a.base, gold)
    base_on = onsets(a.base, gold)
    print(f"{'run':22s} {'clips':>5s} {'hits':>5s} {'miss':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} "
          f"{'FA/clip':>8s} {'cost':>6s} | {'onset med':>9s} {'early':>6s} {'end med':>8s}")

    def line(tag, rows, hits, on):
        agg = S.aggregate(rows)
        d = np.array([v[0] for v in on.values()]) if on else np.array([0.0])
        e = np.array([v[1] for v in on.values()]) if on else np.array([0.0])
        print(f"{tag:22s} {len(rows):5d} {sum(r['hit'] for r in rows):5d} {sum(r['miss'] for r in rows):5d} "
              f"{agg['P']:6.3f} {agg['R']:6.3f} {agg['F1']:6.3f} {S.fa_per_clip(rows):8.2f} "
              f"{S.viewer_cost(rows):6.2f} | {np.median(d):+9.2f} {int((d < -0.5).sum()):6d} {np.median(e):+8.2f}")

    line(a.base, base_rows, base_hits, base_on)
    for tag in a.arms:
        if not root_of(tag).exists():
            print(f"{tag:22s} (not rendered)")
            continue
        rows, hits, on = (*scored(tag, gold), onsets(tag, gold))
        line(tag, rows, hits, on)

    print("\npaired, sound by sound, against the base (the gold set is fixed, so this is honest)")
    for tag in a.arms:
        if not root_of(tag).exists():
            continue
        _, hits = scored(tag, gold)
        shared = sorted(set(base_hits) & set(hits))
        gained = [k for k in shared if hits[k] and not base_hits[k]]
        lost = [k for k in shared if base_hits[k] and not hits[k]]
        print(f"   {tag:22s} recovered {len(gained):2d}   lost {len(lost):2d}   (of {len(shared)} needed sounds)")
        for k in lost:
            print(f"        LOST  {k[0][:28]:28s} {k[1][:16]:16s} at {k[2]}")
        for k in gained:
            print(f"        gained {k[0][:28]:28s} {k[1][:16]:16s} at {k[2]}")

    if a.boot:
        print("\npaired clip bootstrap against the base (2000 draws, seed 0): arm minus base")
        by_base = {r["clip"]: r for r in base_rows}
        for tag in a.arms:
            if not root_of(tag).exists():
                continue
            rows, _ = scored(tag, gold)
            by_arm = {r["clip"]: r for r in rows}
            clips = sorted(set(by_base) & set(by_arm))
            B = [by_base[c] for c in clips]
            A = [by_arm[c] for c in clips]
            stats = {"F1": lambda rs: S.aggregate(rs)["F1"], "P": lambda rs: S.aggregate(rs)["P"],
                     "R": lambda rs: S.aggregate(rs)["R"], "FA/clip": S.fa_per_clip, "cost": S.viewer_cost}
            rng = np.random.default_rng(0)
            idx = [rng.integers(0, len(clips), len(clips)) for _ in range(2000)]
            print(f"   {tag} ({len(clips)} clips)")
            for name, fn in stats.items():
                d = fn(A) - fn(B)
                bs = np.array([fn([A[i] for i in ix]) - fn([B[i] for i in ix]) for ix in idx])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                flag = "*" if lo > 0 or hi < 0 else " "
                print(f"      d{name:8s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}] {flag}")

    print("\nthe five named early cases: our start minus the annotator's")
    print(f"   {'clip / family':38s} {'base':>7s}" + "".join(f" {t[:11]:>12s}" for t in a.arms))
    keys = [k for k in base_on if any(n in k[1] for n in NAMED)]
    arm_on = {t: onsets(t, gold) for t in a.arms if root_of(t).exists()}
    for k in sorted(keys, key=lambda k: base_on[k][0]):
        cells = "".join(f" {arm_on[t][k][0]:+12.2f}" if t in arm_on and k in arm_on[t] else f" {'-':>12s}"
                        for t in a.arms)
        print(f"   {(k[0][:24] + ' / ' + k[1])[:38]:38s} {base_on[k][0]:+7.2f}{cells}")


if __name__ == "__main__":
    main()
