"""Round 38 E5 GBTP screen (generalised band-twin pull; docs/prereg_round13_detector_push.md "Round 38 E5 GBTP"), CPU only,
on the saved SHIP8 proposed pictures of merged DEV, scored on gold_AG.json. For every placed NON-rescued picture, the EARLIEST
onset s within the 3.0 s before its start a (a - 3.0 <= s < a) of a same-family run in any of the three frame-score caches
(FlexSED >= 0.5, BEATs >= 0.35, DASM >= 0.575; runs = the pipeline's _runs over LISTEN_RUN_GAP) that continues up to the
picture's start (e >= a - 0.24) becomes the new start. Never forward; end unchanged; rescued pictures untouched. Nothing in
src/ or config.py is edited. Base (SHIP8) must reproduce 28/58, 21 (6/13/2), 2.282 first.

    TG_ARMS=SHIP8 python benchmark/gold/gbtp_screen.py        # from ~/MscProj_tg
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
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical
from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP

B.ARM = "SHIP8"
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}                 # SHIP8 on the final merged DEV (gold_AG.json)
LOOK, GAP = 3.0, LISTEN_RUN_GAP
BARS = {"flex": float(R.arm_cfg("SHIP8")["BAND_TWIN_PULL"]), "beats": float(config.DISPLAY_THRESHOLD),
        "dasm": float(config.LISTENER_DASM_BAR)}
assert BARS == {"flex": 0.5, "beats": 0.35, "dasm": 0.575}, BARS
assert abs(GAP - 0.24) < 1e-9, GAP
OUT = _ROOT / "benchmark" / "gold" / "gbtp_screen.json"


def load(p):
    if not p.exists():
        return None
    fr = DCC.load_fr(p)
    assert fr[0].shape[0] == len(fr[1]) and fr[0].shape[1] == len(fr[2]), (p, fr[0].shape)
    return fr


def caches(pt, st):
    return {"flex": load(DCC.FLEX_DIR / f"{st}.npz"), "beats": load(CG.PARTS[pt]["beats"] / f"{st}.npz"),
            "dasm": load(CG.PARTS[pt]["dasm"] / f"{st}.npz")}


def family_runs(fr, fam, bar):
    """[(s, e)] of the family's evidence (max over same-family columns) >= bar, gaps <= LISTEN_RUN_GAP merged; None = no evidence"""
    if fr is None:
        return None
    fw, ft, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols:
        return None
    ev = fw[:, cols].max(axis=1).astype(np.float64)
    rr, dt = _runs(ev, ft, bar, GAP)
    return [(float(ft[i]), float(ft[j - 1]) + dt) for i, j in rr]


def earliest(a, cs, fam):
    """(new start, {model: onset}) — the smallest qualifying onset over the three models"""
    found = {}
    for m, fr in cs.items():
        runs = family_runs(fr, fam, BARS[m])
        if not runs:
            continue
        ok = [s for s, e in runs if a - LOOK - 1e-6 <= s < a - 1e-6 and e + 1e-6 >= a - GAP]
        if ok:
            found[m] = min(ok)
    return (min(found.values()) if found else a), found


def cost_w(x, n, w):
    return (4 * (x["n"] - x["hits"]) + w * x["visible"] + 2 * x["cross"] + 2 * x["phantom"]) / n


def main():
    P = B.parts()
    n_resc = sum(r for *_x, pics in P for *_y, r in pics)
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {n_resc} rescued; bars {BARS}, look-back {LOOK} s")
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    n_clips = len(P)
    assert abs(cost_w(Bm["merged"], n_clips, 2) - Bm["merged"]["cost"]) < 1e-9

    rows = {"dev": [], "dev2": []}; moved = []; lost = []
    missing = {m: 0 for m in BARS}; by_model = {m: 0 for m in BARS}; n_cand = 0
    for pt, st, g, pics in P:
        cs = caches(pt, st)
        for m in BARS:
            missing[m] += cs[m] is None
        old = [p[:3] for p in pics]
        new = []
        mv = []
        for l, a, b, resc in pics:
            if resc:
                new.append((l, a, b)); continue
            n_cand += 1
            a2, found = earliest(a, cs, canonical(l))
            assert a2 <= a + 1e-9
            new.append((l, a2, b))
            if a2 < a - 1e-6:
                src = min(found, key=found.get)
                by_model[src] += 1
                mv.append((l, a, a2, found, src))
        r0 = S.score_clip(g, old); r1 = S.score_clip(g, new)
        rows[pt].append(r1)
        if mv:
            c0 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, old)}
            c1 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            assert len(c0) == len(old) and len(c1) == len(new), (st, "classify key collision")
            oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
            for l, a, a2, found, src in mv:
                moved.append({"part": pt, "clip": st, "label": l, "family": canonical(l), "old": round(a, 2), "new": round(a2, 2),
                              "onsets": {m: round(v, 2) for m, v in found.items()}, "earliest_from": src,
                              "pic_before": c0[(l, round(a, 3))], "pic_after": c1[(l, round(a2, 3))],
                              "clip_before": oc(r0), "clip_after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    go = X["merged"]["hits"] >= BASE["hits"] and not lost and X["merged"]["cost"] < Bm["merged"]["cost"]
    print(f"GBTP: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  moved {len(moved)} of {n_cand} "
          f"non-rescued  (earliest from {by_model})  caches missing {missing}  hits lost {lost} -> {'GO' if go else 'STOP'}")
    for d in moved:
        print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['old']} -> {d['new']} ({d['earliest_from']}, onsets {d['onsets']}): "
              f"{d['pic_before']} -> {d['pic_after']}   clip {d['clip_before']} -> {d['clip_after']}")
    sweep = {str(w): {"base": round(cost_w(Bm["merged"], n_clips, w), 3), "gbtp": round(cost_w(X["merged"], n_clips, w), 3)}
             for w in (1, 2)}
    print("cost sweep (merged, w = visible weight):", sweep)
    res = {"base": Bm, "bars": BARS, "look_back": LOOK, "gap": GAP, "rows": X, "moved": moved, "hits_lost": lost,
           "caches_missing": missing, "earliest_from": by_model, "n_nonrescued": n_cand, "cost_sweep": sweep, "GO": go}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
