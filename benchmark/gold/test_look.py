"""THE SINGLE TEST LOOK (2026-09-23). Run once, save everything, then read.

Every decision rule here was fixed in docs/prereg_v4.md BEFORE this ran:

  go/no-go (a) viewer cost per clip below v4b6's on the same clips
           (b) at least 15 hits (v4b6 has 17; the declared cap is a loss of at most 2)
           (c) the sign of the DEV cost improvement reproduced

The script computes in one pass everything a reader could ask for, so that no follow-up question can
force a second look at TEST: the go/no-go, the paired comparison against the blind baseline before
and after, the beta landmarks with intervals, the per-category table (TEST is no-ambient heavy,
which was disclosed before the look), and the identity of every sound the change loses.

    python benchmark/gold/test_look.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD, EARLY, LATE

W_MISS = 4.0
FINAL = "test_final_v30"
BASE = "v4b6"


def cost_arr(rows, b):
    return np.array([W_MISS * r["miss"] + b * (r["visible"] + r["cross"] + r["phantom"]) for r in rows], dtype=float)


def crossing(a, b, hi=20.0):
    """beta at which a becomes more expensive than b; None if a never starts cheaper"""
    lo = 0.0
    if cost_arr(a, lo).mean() >= cost_arr(b, lo).mean():
        return None
    for _ in range(60):
        m = (lo + hi) / 2
        if cost_arr(a, m).mean() < cost_arr(b, m).mean():
            lo = m
        else:
            hi = m
    return lo


def cross_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ix = rng.integers(0, len(a), len(a))
        c = crossing([a[i] for i in ix], [b[i] for i in ix])
        out.append(c if c is not None else 0.0)
    return float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def complete(root, stems):
    """a render is finished only when every augmented spec has an image file that exists"""
    shown = have = 0
    for st in stems:
        f = root / st / "augmentations.json"
        if not f.exists():
            continue
        for x in json.loads(f.read_text(encoding="utf-8")):
            if x.get("augment"):
                shown += 1
                ip = x.get("image_path")
                if ip and Path(ip).exists():
                    have += 1
    return shown, have


def hit_in(ps, s):
    return any(S.same_family(l, s["label"]) and S.in_window(x, s["start"], EARLY, LATE) for l, x, _ in ps)


def main():
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs["test_bench"]) & set(subs["bench"]))
    w = _ROOT / "data" / "work"
    out = {"clips": len(stems)}

    print("== 0. completeness guard (both arms, before any number is read)")
    for tag, sysn in ((FINAL, "proposed"), (FINAL, "blind_a2i"), (BASE, "proposed"), (BASE, "blind_a2i")):
        shown, have = complete(w / f"protocol_{sysn}_{tag}", stems)
        ok = have >= shown - 2
        print(f"   {sysn:10s} {tag:14s} {have}/{shown} pictures exist  {'OK' if ok else '*** UNFINISHED ***'}")
        if not ok:
            sys.exit("refusing to read TEST on an unfinished render")

    def rows(tag, sysn, sts=None):
        sts = sts or stems
        return [S.score_clip(gold[s], S.load_pictures(w / f"protocol_{sysn}_{tag}", s, sysn) or []) for s in sts]

    R = {"ours": rows(FINAL, "proposed"), "blind": rows(FINAL, "blind_a2i"),
         "v4b6": rows(BASE, "proposed"), "v4b6_blind": rows(BASE, "blind_a2i"),
         "silence": [S.score_clip(gold[s], []) for s in stems]}

    print("\n== 1. go/no-go, in the order declared")
    hits_new = sum(r["hit"] for r in R["ours"])
    hits_old = sum(r["hit"] for r in R["v4b6"])
    c_new, c_old = S.viewer_cost(R["ours"]), S.viewer_cost(R["v4b6"])
    a_ok, b_ok = c_new < c_old, hits_new >= 15
    c_ok = c_new < c_old
    print(f"   (a) cost below v4b6     : {c_new:.2f} vs {c_old:.2f}       {'PASS' if a_ok else 'FAIL'}")
    print(f"   (b) at least 15 hits    : {hits_new} (v4b6 has {hits_old})   {'PASS' if b_ok else 'FAIL'}")
    print(f"   (c) DEV sign reproduced : DEV 4.12->3.10, TEST {'fell' if c_ok else 'rose'}   {'PASS' if c_ok else 'FAIL'}")
    verdict = "ADOPTED" if (a_ok and b_ok and c_ok) else "NOT ADOPTED"
    print(f"   VERDICT: {verdict}")
    out["go_no_go"] = {"cost_new": c_new, "cost_v4b6": c_old, "hits_new": hits_new,
                       "hits_v4b6": hits_old, "a": a_ok, "b": b_ok, "c": c_ok, "verdict": verdict}

    print("\n== 2. TEST, %d clips" % len(stems))
    print(f"   {'system':14s} {'F1':>6s} {'P':>6s} {'R':>6s} {'FA/clip':>8s} {'cost':>6s} {'hits':>5s} {'miss':>5s}")
    for k in ("v4b6", "v4b6_blind", "ours", "blind", "silence"):
        a = S.aggregate(R[k])
        print(f"   {k:14s} {a['F1'] or 0:6.3f} {a['P'] or 0:6.3f} {a['R'] or 0:6.3f} "
              f"{S.fa_per_clip(R[k]):8.2f} {S.viewer_cost(R[k]):6.2f} "
              f"{sum(r['hit'] for r in R[k]):5d} {sum(r['miss'] for r in R[k]):5d}")
        out[k] = {"F1": a["F1"], "P": a["P"], "R": a["R"], "fa": S.fa_per_clip(R[k]),
                  "cost": S.viewer_cost(R[k]), "hits": sum(r["hit"] for r in R[k]),
                  "miss": sum(r["miss"] for r in R[k])}

    print("\n== 3. ours vs blind, paired clip bootstrap (2000 draws, seed 0); * = significant")
    for nm, a, b in (("before (v4b6)", R["v4b6"], R["v4b6_blind"]),
                     ("after (final cell)", R["ours"], R["blind"])):
        parts = [f"   {nm:20s}"]
        for k in ("F1", "P"):
            d, lo, hi, _ = S.paired_ci(a, b, key=k)
            parts.append(f"d{k} {d:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
        rng = np.random.default_rng(0)
        dd = cost_arr(b, 2.0) - cost_arr(a, 2.0)
        boot = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        parts.append(f"dcost {dd.mean():+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 else ' '}")
        print("  ".join(parts))

    print("\n== 4. beta landmarks on TEST")
    g = crossing(R["blind"], R["ours"])
    print(f"   {'gate starts paying (blind overtakes ours)':44s} " + (f"beta = {g:.2f}" if g else "blind never cheaper"))
    for nm, a in (("ours = silence", R["ours"]), ("blind = silence", R["blind"])):
        c = crossing(a, R["silence"])
        if c is None:
            print(f"   {nm:44s} never cheaper than silence in [0, 20]")
        else:
            _, lo, hi = cross_ci(a, R["silence"])
            print(f"   {nm:44s} beta = {c:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
            out["crossing_" + nm.split()[0]] = {"beta": c, "lo": lo, "hi": hi}

    print("\n== 5. per category (TEST is no-ambient heavy; disclosed before this look)")
    print(f"   {'category':12s} {'clips':>5s} {'ours':>7s} {'blind':>7s} {'silence':>8s} {'ours P':>7s} {'ours R':>7s}")
    percat = {}
    for cat in ("unseen", "mixed", "seen", "no_ambient"):
        sts = [s for s in stems if S.category(gold[s]) == cat]
        if not sts:
            continue
        o, bl = rows(FINAL, "proposed", sts), rows(FINAL, "blind_a2i", sts)
        sil = [S.score_clip(gold[s], []) for s in sts]
        a = S.aggregate(o)
        print(f"   {cat:12s} {len(sts):5d} {S.viewer_cost(o):7.2f} {S.viewer_cost(bl):7.2f} "
              f"{S.viewer_cost(sil):8.2f} {a['P'] or 0:7.3f} {a['R'] or 0:7.3f}")
        percat[cat] = {"clips": len(sts), "ours": S.viewer_cost(o), "blind": S.viewer_cost(bl),
                       "silence": S.viewer_cost(sil), "P": a["P"], "R": a["R"]}
    out["per_category"] = percat
    un = [s for s in stems if S.category(gold[s]) == "unseen"]
    o = rows(FINAL, "proposed", un)
    sil = [S.score_clip(gold[s], []) for s in un]
    c = crossing(o, sil)
    if c:
        _, lo, hi = cross_ci(o, sil)
        print(f"   unseen-only crossing: beta = {c:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  ({len(un)} clips)")
        out["crossing_unseen"] = {"beta": c, "lo": lo, "hi": hi, "clips": len(un)}
    else:
        print(f"   unseen-only: never cheaper than silence ({len(un)} clips)")

    print("\n== 6. every needed sound v4b6 showed and the final cell loses")
    lost = []
    for st in stems:
        pn = S.load_pictures(w / f"protocol_proposed_{FINAL}", st, "proposed") or []
        po = S.load_pictures(w / f"protocol_proposed_{BASE}", st, "proposed") or []
        for s in gold[st]:
            if s["needed"] and s["importance"] >= 2 and hit_in(po, s) and not hit_in(pn, s):
                lost.append((st, s["label"], s["start"], s["importance"]))
    for st, lab, t, imp in lost:
        print(f"   {st[:34]:34s} {lab[:24]:24s} at {t:5.2f}s  (importance {imp})")
    print(f"   total lost: {len(lost)}")
    out["lost"] = [{"clip": a, "label": b, "start": c, "importance": d} for a, b, c, d in lost]

    p = _ROOT / "benchmark" / "gold" / "test_look.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n->", p)


if __name__ == "__main__":
    main()
