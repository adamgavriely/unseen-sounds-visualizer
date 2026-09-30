"""Round 35 DBR screen (DASM-covered repeat drop; docs/prereg_round13_detector_push.md "Round 35 DBR"), CPU only, on the
saved SHIP7 (= SHIP6+FLAP) proposed pictures of merged DEV, scored on the corrected DEV gold (visibility re-check). A placed
NON-rescued picture whose family already has an earlier placed picture in the clip is dropped when the family's DASM frame
score (max over its cache columns) is >= 0.575 (LISTENER_DASM_BAR) in every frame of the gap [earlier end, this start]
(gaps <= LISTEN_RUN_GAP = 0.24 s merged, the pipeline's _runs). Rescored with score_per_sound; nothing in src/ or config.py
is edited. Pass = vs SHIP7 on the corrected gold (28/58, 24, 2.366), which the screen must reproduce first.

    TG_ARMS=SHIP6+FLAP python benchmark/gold/dbr_screen.py        # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP6+FLAP")
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from benchmark.gold.dbx_screen import family_runs, BAR          # 0.575, _runs over LISTEN_RUN_GAP, identical to DBX
from src.labels import canonical

B.ARM = "SHIP6+FLAP"
BASE = {"hits": 28, "wrong": 24, "cost": 2.366}                 # SHIP7 on the corrected DEV gold (merged)
FLAT = 0.05
OUT = _ROOT / "benchmark" / "gold" / "dbr_screen.json"


def covered(runs, lo, hi):
    """one merged DASM run (>= BAR, gaps <= 0.24 s) spans [lo, hi]"""
    return any(s - 1e-6 <= lo and r + 1e-6 >= hi for s, r in runs)


def family_flat(fr):
    """{family: max - min of the family's DASM score over the whole clip}"""
    if fr is None:
        return {}
    fw, ft, labs = fr
    fam = {}
    for c, l in enumerate(labs):
        f = canonical(l)
        fam[f] = np.maximum(fam[f], fw[:, c]) if f in fam else fw[:, c].astype(np.float64)
    return {f: float(v.max() - v.min()) for f, v in fam.items()}


def passes(X, lost_parts, lost_n):
    gain = X["hits"] - BASE["hits"]
    old = (X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"]
           and not lost_parts)
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * lost_n and lost_n <= 3
    return old, few


def main():
    P = B.parts()
    n_resc = sum(r for *_x, pics in P for *_y, r in pics)
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {n_resc} rescued; DASM bar {BAR}")
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP7 (corrected gold): merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    BASE["cost"] = Bm["merged"]["cost"]
    dasm = {}
    for pt, st, g, pics in P:
        dx = CG.PARTS[pt]["dasm"] / f"{st}.npz"; dasm[(pt, st)] = DCC.load_fr(dx) if dx.exists() else None
    print(f"  clips without DASM cache: {sum(v is None for v in dasm.values())}")

    rows = {"dev": [], "dev2": []}; dropped = []; lost = []
    silent = {"first of family": 0, "rescued": 0, "no DASM evidence": 0, "overlap": 0, "DASM gap not covered": 0}
    flat_placed, flat_any = [], []
    for pt, st, g, pics in P:
        fr = dasm[(pt, st)]
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        keep = []
        for l, a, b, resc in pics:
            fam = canonical(l)
            earlier = [b2 for l2, a2, b2, _r in pics if canonical(l2) == fam and a2 < a - 1e-6]     # original placed set
            if not earlier:
                why = "first of family"
            elif resc:
                why = "rescued"
            else:
                prev_end = max(earlier)
                runs = family_runs(fr, fam)
                if runs is None:
                    why = "no DASM evidence"
                elif prev_end >= a - 1e-6:
                    why = "overlap"
                elif covered(runs, prev_end, a):
                    why = "dropped"
                else:
                    why = "DASM gap not covered"
            if why == "dropped":
                dropped.append({"part": pt, "clip": st, "label": l, "family": fam, "start": round(a, 2), "end": round(b, 2),
                                "prev_end": round(prev_end, 2), "before": before[(l, round(a, 3))]})
            else:
                silent[why] += 1
                keep.append((l, a, b))
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
        rows[pt].append(r1)
        oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
        for d in dropped:
            if d["part"] == pt and d["clip"] == st and "clip_after" not in d:
                d["clip_before"], d["clip_after"] = oc(r0), oc(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
        ff = family_flat(fr)
        fams = {canonical(l) for l, *_ in pics}
        fp = sorted(f for f in fams if f in ff and ff[f] < FLAT)
        fa = sorted(f for f, v in ff.items() if v < FLAT)
        if fp:
            flat_placed.append({"part": pt, "clip": st, "families": fp})
        if fa:
            flat_any.append({"part": pt, "clip": st, "n": len(fa), "of": len(ff)})
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    ln = sum(x[2] for x in lost)
    old, few = passes(X["merged"], lost, ln)
    verdict = "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
    print(f"DBR: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  dropped {len(dropped)}  "
          f"untouched {silent}  hits lost {lost} -> {verdict}")
    for d in dropped:
        print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['start']}-{d['end']} (earlier ends {d['prev_end']}): "
              f"{d['before']} dropped   clip {d['clip_before']} -> {d['clip_after']}")
    print(f"flat DASM family (max-min < {FLAT}): {len(flat_placed)} clips on a placed-picture family, "
          f"{len(flat_any)} clips on any family")
    for f in flat_placed:
        print(f"   {f['part']:4s} {f['clip']}: {f['families']}")
    res = {"base": Bm, "bar": BAR, "rows": X, "dropped": dropped, "hits_lost": lost, "untouched": silent, "old_rule": old,
           "fewer_pictures": few, "verdict": verdict, "flat_placed": flat_placed, "flat_any": flat_any}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
