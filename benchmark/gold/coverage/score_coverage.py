"""Scorer v2: the onset scorer (score_per_sound.score_clip, unchanged) plus how long each needed sound is covered.

Hits, misses and wrong pictures are exactly score_per_sound's. Added per needed sound (importance 2-3):
  coverage = share of the sound's labelled time under a same-family picture that STARTED in this sound's onset window
             (-0.5 .. +1.0 s). A miss has 0. A picture that stays up through repeats adds coverage, never a second hit.
  cost_cov = (4 * sum over needed sounds (1 - coverage) + 2 * wrong pictures) / clips      (lower is better)
Also: old onset cost, hits, wrong, hit coverage (mean over hits), needed-time coverage (covered s / needed s),
wrong-picture seconds per clip, stale seconds per clip (a matched picture still up > 1 s after the end of the
sound(s) it matched).

    python benchmark/gold/coverage/score_coverage.py [pics.json ...] [--ends none|1]
pics.json: {"clips": {stem: {"pics_none": [[label, start, end], ...], "pics_1": [...]}}} (dump_pictures.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
STALE_AFTER = 1.0


def stems(name):
    return [x.strip() for x in (_ROOT / "benchmark" / "gold" / f"{name}_stems.txt").read_text(encoding="utf-8").splitlines() if x.strip()]


def _union(ivs):
    tot, cur = 0.0, None
    for a, b in sorted(ivs):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def scored(g):
    return g["needed"] and g["importance"] >= S.MIN_IMPORTANCE


def score_clip_v2(gold, pics, early=S.EARLY, late=S.LATE):
    """score_per_sound.score_clip + coverage, wrong seconds, stale seconds"""
    out = S.score_clip(gold, pics, early, late)
    g = sorted(gold, key=lambda x: x["start"])
    pics = sorted(pics, key=lambda p: p[1])
    # replay the matching (same order and rule as score_clip) to know which picture matched which sounds
    taken = [False] * len(g)
    matched = {}                       # picture index -> covered gold indices
    wrong_s = 0.0
    for k, (lab, a, b) in enumerate(pics):
        cands = [i for i, x in enumerate(g) if S.same_family(lab, x["label"]) and S.in_window(a, x["start"], early, late)]
        if cands:
            free = [i for i in cands if not taken[i]]
            if not free:
                continue                                   # duplicate: neither credit nor cost
            i = min(free, key=lambda i: abs(a - g[i]["start"]))
            covered = [i] + [j for j in cands if not taken[j] and S.same_family(g[j]["label"], g[i]["label"])]
            for j in covered:
                taken[j] = True
            hit_here = any(scored(g[j]) for j in covered)
            if any(not g[j]["needed"] for j in covered) and not hit_here:
                wrong_s += b - a                           # visible / obvious source
            else:
                matched[k] = covered
        else:
            wrong_s += b - a                               # cross-trigger or phantom
    cov, hit_cov, sec_need, sec_cov = [], [], 0.0, 0.0
    for i, x in enumerate(g):
        if not scored(x) or x["end"] <= x["start"]:
            continue
        # pictures of this family that started in THIS sound's onset window (a hit picture, or a duplicate in the window)
        ivs = [(max(a, x["start"]), min(b, x["end"])) for lab, a, b in pics
               if S.same_family(lab, x["label"]) and S.in_window(a, x["start"], early, late) and b > x["start"] and a < x["end"]]
        hit = any(i in cv for cv in matched.values())
        c = _union(ivs) if hit else 0.0
        cov.append(c / (x["end"] - x["start"]))
        if hit:
            hit_cov.append(cov[-1])
        sec_need += x["end"] - x["start"]; sec_cov += c
    stale = 0.0
    for k, covered in matched.items():
        end = max(g[j]["end"] for j in covered)
        stale += max(0.0, pics[k][2] - (end + STALE_AFTER))
    out.update(cov2=cov, hit_cov=hit_cov,
               sec_need=sec_need, sec_cov=sec_cov, wrong_s=wrong_s, stale_s=stale)
    return out


def aggregate(rows):
    n = max(1, len(rows))
    wrong = sum(r["visible"] + r["cross"] + r["phantom"] for r in rows)
    cov = [c for r in rows for c in r["cov2"]]
    hits = sum(r["hit"] for r in rows)
    hc = [c for r in rows for c in r["hit_cov"]]
    return {"clips": len(rows), "hits": hits, "needed": hits + sum(r["miss"] for r in rows), "wrong": wrong,
            "onset_cost": S.viewer_cost(rows),
            "cost_cov": (4.0 * sum(1.0 - c for c in cov) + 2.0 * wrong) / n,
            "hit_cov": float(np.mean(hc)) if hc else None,
            "needed_time_cov": sum(r["sec_cov"] for r in rows) / max(1e-9, sum(r["sec_need"] for r in rows)),
            "wrong_s_per_clip": sum(r["wrong_s"] for r in rows) / n,
            "stale_s_per_clip": sum(r["stale_s"] for r in rows) / n,
            "n_cov": len(cov)}


def fmt(a):
    hc = "-" if a["hit_cov"] is None else f"{a['hit_cov']:.2f}"
    return (f"{a['hits']:3d}/{a['needed']:<3d} wrong {a['wrong']:3d} | onset {a['onset_cost']:.3f} cost_cov {a['cost_cov']:.3f} | "
            f"hit-cov {hc} need-time-cov {a['needed_time_cov']:.2f} | wrong-s {a['wrong_s_per_clip']:.2f} stale-s {a['stale_s_per_clip']:.2f}")


def score_file(path, key, gold, splits):
    d = json.loads(Path(path).read_text(encoding="utf-8"))["clips"]
    res = {}
    for split, sts in splits.items():
        rows = []
        for st in sts:
            p = d.get(st, {}).get(key)
            if p is None:
                raise SystemExit(f"{path}: no pictures for {st}")
            rows.append(score_clip_v2(gold[st], [tuple(x) for x in p]))
        res[split] = aggregate(rows)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pics", nargs="*", default=[str(Path(__file__).with_name("pics_frozen.json"))])
    ap.add_argument("--ends", default="none", choices=["none", "1"], help="pics_none (scoring harness) or pics_1 (shipped)")
    ap.add_argument("--splits", nargs="+", default=["DEV", "TEST"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gold = S.load_gold([GOLD])
    splits = {s: stems(s.lower()) for s in a.splits}
    res = {}
    for split, sts in splits.items():
        res[f"silence|{split}"] = aggregate([score_clip_v2(gold[st], []) for st in sts])
        print(f"[{split:4s}] show-nothing      {fmt(res[f'silence|{split}'])}")
    for p in a.pics:
        r = score_file(p, "pics_" + a.ends, gold, splits)
        for split, agg in r.items():
            res[f"{Path(p).stem}|{split}"] = agg
            print(f"[{split:4s}] {Path(p).stem:17s} {fmt(agg)}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("->", a.out)


if __name__ == "__main__":
    main()
