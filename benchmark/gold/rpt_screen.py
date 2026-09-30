"""Round 31 RPT-S screen (docs/prereg_round13_detector_push.md "Round 31 RPT-S"), CPU only, on the saved SHIP4+BTP (= SHIP5)
proposed pictures of merged DEV. A placed picture whose family already has an earlier placed picture in the clip is dropped
unless the family's FlexSED score (max over its queries) is < BAR for a contiguous >= SIL s somewhere in [previous same-family
picture start, this start]. Secondary: the silence must sit in [start - SIL, start). Rescored with score_per_sound;
nothing in src/ or config.py is edited.

    python benchmark/gold/rpt_screen.py [--bar 0.5] [--sil 1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import btp_screen as B
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from src.labels import canonical

B.ARM = "SHIP4+BTP"
OUT = _ROOT / "benchmark" / "gold" / "rpt_screen.json"
BASE = {"hits": 25, "wrong": 33, "cost": 2.620}


def flex_family(stem):
    p = B.FLEX_DIR / f"{stem}.npz"
    if not p.exists():
        return None
    fw, ft, labs = DCC.load_fr(p)
    fw, ft = np.asarray(fw), np.asarray(ft)
    out = {}
    for c, lab in enumerate(labs):
        f = canonical(lab)
        out[f] = np.maximum(out[f], fw[:, c]) if f in out else fw[:, c].copy()
    return out, ft


def silent_stretch(col, ft, a, b, bar):
    """longest contiguous stretch (s) inside [a, b] with col < bar"""
    dt = float(ft[1] - ft[0]) if len(ft) > 1 else 0.04
    m = (ft >= a) & (ft < b)
    if not m.any():
        return 0.0
    off = col[m] < bar
    best = run = 0
    for x in off:
        run = run + 1 if x else 0
        best = max(best, run)
    return best * dt


def apply(pics, fx, bar, sil, mode):
    keep, dropped = [], []
    prev = {}
    for lab, a, b, resc in sorted(pics, key=lambda p: p[1]):
        f = canonical(lab)
        if f in prev and fx is not None and f in fx[0]:
            col, ft = fx[0][f], fx[1]
            lo, hi = (prev[f], a) if mode == "between" else (max(0.0, a - sil), a)
            s = silent_stretch(col, ft, lo, hi, bar)
            if s < sil - 1e-6:
                dropped.append([lab, round(a, 2), round(b, 2), resc, round(s, 2)])
                continue
        prev.setdefault(f, a)
        keep.append((lab, a, b))
    return keep, dropped


def verdict(X, lost_parts, lost_n):
    gain = X["hits"] - BASE["hits"]
    old = (X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0)
           and X["cost"] < BASE["cost"] and not lost_parts)
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * lost_n and lost_n <= 3
    return old, few


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=float, default=0.5)
    ap.add_argument("--sil", type=float, default=1.0)
    a = ap.parse_args()
    P = B.parts()
    res = {"bar": a.bar, "sil": a.sil}
    base = {pt: [] for pt in ("dev", "dev2")}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    res["base"] = Bm
    print(f"BASE {B.ARM}: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    for mode in ("between", "before"):
        rows = {pt: [] for pt in ("dev", "dev2")}
        ch, lost = [], []
        for pt, st, g, pics in P:
            fx = flex_family(st)
            new, dropped = apply(pics, fx, a.bar, a.sil, mode)
            r0 = S.score_clip(g, [p[:3] for p in pics])
            r1 = S.score_clip(g, new)
            rows[pt].append(r1)
            if dropped:
                oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
                ch.append({"part": pt, "clip": st, "dropped": dropped, "before": oc(r0), "after": oc(r1)})
            if r1["hit"] < r0["hit"]:
                lost.append((pt, st, r0["hit"] - r1["hit"]))
        X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
        ln = sum(x[2] for x in lost)
        old, few = verdict(X["merged"], lost, ln)
        go = old or few
        res[mode] = {"rows": X, "changed": ch, "hits_lost": lost, "old_rule": old, "fewer_pictures": few, "GO": go}
        print(f"RPT-S[{mode}]: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  "
              f"hits lost {lost}  old {old} fewer {few} -> {'GO' if go else 'STOP'}")
        for c in ch:
            print(f"   {c['part']:4s} {c['clip']}: {c['dropped']}  {c['before']} -> {c['after']}")
    OUT.with_name(f"rpt_screen_bar{a.bar}_sil{a.sil}.json").write_text(json.dumps(res, indent=1, default=float),
                                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
