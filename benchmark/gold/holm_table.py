"""The results hierarchy of docs/panel_2026-09-26_plan.md §A, for one table (one tag, one subset).

Primary: per-sound dF1 ours - blind (outside every family). Family 1 (Holm): the seven secondary rows the
scorer prints for ours - blind. Family 2 (Holm): ours - silence on F1 (all clips), viewer cost (all clips) and
viewer cost on the unseen stratum ("one pre-declared subgroup (amendment 5) of a post-hoc metric (amendment 9)").
Two-sided bootstrap p = min(1, 2 min(P(d >= 0), P(d <= 0))) over the same 2000 paired draws as
`score_per_sound.paired_ci` (seed 0); each recomputed (d, lo, hi) is asserted equal to paired_ci's before a p is
used. Differences are ours - other, as the scorer prints them: a negative cost / FA difference is a gain.

    python benchmark/gold/holm_table.py --tag dev_monocap_v31 --subset dev
    python benchmark/gold/holm_table.py --tag test_final_v33 --subset test --expect 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_per_sound as sps  # noqa: E402

FAMILY1 = [("F0.5", "dF0.5"), ("wF1", "dwF1"), ("P", "dP"), ("R", "dR"), ("fa_per_clip", "dFA/clip"),
           ("viewer_cost", "d cost/clip"), ("clean_acc", "d clean-acc")]


def paired_draws(rows_a, rows_b, key, n=2000, seed=0):
    """the draws behind score_per_sound.paired_ci (same generator, same order), returned in full"""
    rng = np.random.default_rng(seed)
    m = len(rows_a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        a = sps.aggregate([rows_a[i] for i in idx])[key]; b = sps.aggregate([rows_b[i] for i in idx])[key]
        diffs.append((a or 0.0) - (b or 0.0))
    return np.asarray(diffs)


def row(rows_a, rows_b, key):
    d, lo, hi, _ = sps.paired_ci(rows_a, rows_b, key=key)
    diffs = paired_draws(rows_a, rows_b, key)
    d2 = sps.aggregate(rows_a)[key] - sps.aggregate(rows_b)[key]
    lo2, hi2 = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    assert (d, lo, hi) == (d2, lo2, hi2), f"{key}: recomputed draws differ from paired_ci"
    p = min(1.0, 2 * min(float(np.mean(diffs >= 0)), float(np.mean(diffs <= 0))))
    return {"d": d, "ci": [lo, hi], "p": p}


def holm(rows, alpha=0.05):
    """step-down Holm over rows (dicts with 'p'); adds p_holm and survives"""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, running, stop = len(rows), 0.0, False
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * rows[i]["p"]))
        rows[i]["p_holm"] = running
        stop = stop or rows[i]["p"] > alpha / (m - k)
        rows[i]["survives"] = not stop
    return rows


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--annotations", nargs="+", default=["benchmark/gold/annotations/gold_AG.json"])
    ap.add_argument("--blind", default="blind_a2i")
    ap.add_argument("--work", default=None)
    ap.add_argument("--expect", type=int, default=0, help="completeness guard: clips every system must have")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gold = sps.load_gold(a.annotations)
    subs = sps.subsets_of(gold)
    stems_all = sorted(subs[a.subset])
    work = Path(a.work) if a.work else sps._ROOT / "data" / "work"
    per = {}
    for system in ("proposed", a.blind, "silence"):
        root = work / f"protocol_{system}_{a.tag}"
        rows = {}
        for st in stems_all:
            pics = [] if system == "silence" else sps.load_pictures(root, st, system)
            if pics is not None:
                rows[st] = sps.score_clip(gold[st], pics)
        per[system] = rows
        print(f"[guard] {system:10s} {len(rows)} of {len(stems_all)} clips rendered")
        if a.expect and len(rows) != a.expect:
            sys.exit(f"[guard] {system}: {len(rows)} clips, expected {a.expect} -- nothing scored")
    stems = [st for st in stems_all if all(st in per[s] for s in per)]
    A = [per["proposed"][st] for st in stems]; B = [per[a.blind][st] for st in stems]
    S = [per["silence"][st] for st in stems]
    out = {"tag": a.tag, "subset": a.subset, "clips": len(stems)}

    prim = row(A, B, "F1")
    out["primary"] = prim
    agg = {s: sps.aggregate(r) for s, r in (("proposed", A), (a.blind, B), ("silence", S))}
    print(f"\n{a.tag} / {a.subset}: {len(stems)} clips, {agg['proposed']['needed']} needed sounds")
    for s, g in agg.items():
        print(f"  {s:10s} F1 {g['F1']:.3f}  P {g['P']:.3f}  R {g['R']:.3f}  FA/clip {g['fa_per_clip']:.2f}  cost {g['viewer_cost']:.2f}  hits {g['hits']} miss {g['misses']}")
    print(f"\nPRIMARY  dF1 ours - {a.blind} = {prim['d']:+.3f} [{prim['ci'][0]:+.3f}, {prim['ci'][1]:+.3f}]  p {fmt_p(prim['p'])}  (outside the families)")

    fam1 = []
    for key, name in FAMILY1:
        try:
            r = row(A, B, key)
        except TypeError:
            continue
        r["row"] = name; fam1.append(r)
    holm(fam1)
    print(f"\nFAMILY 1  ours - {a.blind}, Holm over {len(fam1)} (negative cost/FA = gain; a surviving dR < 0 is a surviving LOSS)")
    for r in fam1:
        print(f"  {r['row']:12s} {r['d']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  p {fmt_p(r['p'])}  Holm {fmt_p(r['p_holm'])}  {'SURVIVES' if r['survives'] else '-'}")
    out["family1"] = fam1

    unseen = [st for st in stems if st in subs.get("cat_unseen", set())]
    fam2 = [dict(row(A, S, "F1"), row="dF1 vs silence (all; = ours' F1, silence has none)"),
            dict(row(A, S, "viewer_cost"), row="d cost vs silence (all)")]
    if unseen:
        fam2.append(dict(row([per["proposed"][s] for s in unseen], [per["silence"][s] for s in unseen], "viewer_cost"),
                         row=f"d cost vs silence (unseen, {len(unseen)} clips; pre-declared subgroup of a post-hoc metric)"))
    holm(fam2)
    print("\nFAMILY 2  ours - silence, Holm over", len(fam2))
    for r in fam2:
        print(f"  {r['d']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  p {fmt_p(r['p'])}  Holm {fmt_p(r['p_holm'])}  {'SURVIVES' if r['survives'] else '-'}   {r['row']}")
    out["family2"] = fam2

    print("\nCATEGORY ROWS (CIs only, no stars, in no family)")
    cats = {}
    for cat in ("cat_unseen", "cat_mixed", "cat_seen", "cat_no_ambient"):
        cs = [st for st in stems if st in subs.get(cat, set())]
        if not cs:
            continue
        g = {s: sps.aggregate([per[s][st] for st in cs]) for s in per}
        d = row([per["proposed"][st] for st in cs], [per[a.blind][st] for st in cs], "viewer_cost")
        cats[cat] = {"clips": len(cs), "cost": {s: g[s]["viewer_cost"] for s in g}, "d_cost_vs_blind": d["d"], "ci": d["ci"]}
        print(f"  {cat:15s} {len(cs):3d} clips  cost ours {g['proposed']['viewer_cost']:.2f}  blind {g[a.blind]['viewer_cost']:.2f}  silence {g['silence']['viewer_cost']:.2f}  "
              f"d cost vs blind {d['d']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]")
    out["categories"] = cats
    dest = Path(a.out) if a.out else sps._ROOT / "benchmark" / "gold" / f"holm_{a.tag}_{a.subset}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", dest)


if __name__ == "__main__":
    main()
