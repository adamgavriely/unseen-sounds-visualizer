"""Round 40e EXPECT-A4 (docs/prereg_round13_detector_push.md "Round 40e EXPECT-A4"): the 23 pictures Round 40d added, each kept
only if its family's DASM score (max over same-canonical columns) >= LISTENER_DASM_BAR 0.575 within onset +- 0.5 s; no DASM column
for the family -> not kept (reported). Secondary (report only): the shipped _dasm_keeps window, span +- 0.5 s. CPU only.

    TG_ARMS=SHIP8 python benchmark/gold/expect_a4_screen.py        # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
import config
from benchmark.gold import expect_screen as E
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import gbtp_screen as G
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical

SRC = _ROOT / "benchmark" / "gold" / "expect_a3_screen.json"
OUT = _ROOT / "benchmark" / "gold" / "expect_a4_screen.json"
BAR, HALF, PIC_LEN = float(config.LISTENER_DASM_BAR), 0.5, 2.0
assert BAR == 0.575, BAR


def dasm_max(fr, fam, lo, hi):
    """(max of the family's DASM score over frames lo <= t <= hi, or None without a column / frame)"""
    if fr is None:
        return None
    fw, ft, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols:
        return None
    m = (ft >= lo - 1e-9) & (ft <= hi + 1e-9)
    if not m.any():
        return None
    return float(fw[m][:, cols].max())


def main():
    added = json.loads(SRC.read_text(encoding="utf-8"))["added"]
    P = E.parts()
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (E.BASE["hits"], E.BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - E.BASE["cost"]) < 0.001, Bm["merged"]
    n_clips = len(P)
    rows = {"dev": [], "dev2": []}; rows2 = {"dev": [], "dev2": []}; rep = []; lost = []
    tally = {"kept": 0, "below": 0, "no_column": 0, "kept_span": 0}
    for pt, st, g, pics in P:
        fr = DCC.load_fr(CG.PARTS[pt]["dasm"] / f"{st}.npz") if (CG.PARTS[pt]["dasm"] / f"{st}.npz").exists() else None
        old = [p[:3] for p in pics]; new = list(old); new2 = list(old)
        for d in [a for a in added if a["clip"] == st and a["part"] == pt]:
            fam, a = d["family"], float(d["start"])
            v = dasm_max(fr, fam, a - HALF, a + HALF)
            v2 = dasm_max(fr, fam, a - HALF, a + PIC_LEN + HALF)
            if v is None:
                out = "no DASM column"; tally["no_column"] += 1
            elif v >= BAR:
                out = "kept"; tally["kept"] += 1; new.append((fam, a, a + PIC_LEN))
            else:
                out = "below bar"; tally["below"] += 1
            if v2 is not None and v2 >= BAR:
                tally["kept_span"] += 1; new2.append((fam, a, a + PIC_LEN))
            rep.append({"part": pt, "clip": st, "family": fam, "start": a, "class_40d": d["class"], "dasm_onset": None if v is None else round(v, 3),
                        "dasm_span": None if v2 is None else round(v2, 3), "outcome": out})
        r0 = S.score_clip(g, old); r1 = S.score_clip(g, new); r2 = S.score_clip(g, new2)
        rows[pt].append(r1); rows2[pt].append(r2)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    X = {k: B.summ(rows["dev"] + rows["dev2"]) if k == "merged" else B.summ(rows[k]) for k in ("merged", "dev", "dev2")}
    X2 = {k: B.summ(rows2["dev"] + rows2["dev2"]) if k == "merged" else B.summ(rows2[k]) for k in ("merged", "dev", "dev2")}
    gain = X["merged"]["hits"] - E.BASE["hits"]
    c1w = {w: G.cost_w(X["merged"], n_clips, w) for w in (1, 2)}
    c0w = {w: G.cost_w(Bm["merged"], n_clips, w) for w in (1, 2)}
    main_go = X["merged"]["hits"] >= E.BASE["hits"] and not lost and c1w[2] < c0w[2]
    fa_ok = all(X[pt][k] <= Bm[pt][k] for pt in ("dev", "dev2") for k in ("cross", "phantom"))
    more_go = gain > 0 and fa_ok and c1w[1] < c0w[1]
    print(f"EXPECT-A4: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  {tally}  hits lost {lost}")
    print(f"  main rule: {'GO' if main_go else 'STOP'}   more-hits rule: {'PASS' if more_go else 'FAIL'}   "
          f"cost w=1 {c0w[1]:.3f} -> {c1w[1]:.3f}, w=2 {c0w[2]:.3f} -> {c1w[2]:.3f}")
    print(f"  secondary (span +- 0.5 window): merged {B.fmt(X2['merged'])}")
    for r in rep:
        print(f"   {r['part']:4s} {r['clip']} {r['family']} @{r['start']} [{r['class_40d']}] dasm {r['dasm_onset']} (span {r['dasm_span']}) -> {r['outcome']}")
    OUT.write_text(json.dumps({"base": Bm, "rows": X, "rows_span_window": X2, "pictures": rep, "tally": tally, "hits_lost": lost,
                               "cost": {str(w): {"base": round(c0w[w], 3), "new": round(c1w[w], 3)} for w in (1, 2)},
                               "main_GO": main_go, "more_hits_PASS": more_go}, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
