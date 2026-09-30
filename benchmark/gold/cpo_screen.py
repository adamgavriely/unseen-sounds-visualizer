"""Round 33 CPO screen (change-point onset gate; docs/prereg_round13_detector_push.md "Round 33 CPO"), CPU only, on the
saved SHIP7 (= SHIP6+FLAP) proposed pictures of merged DEV. A placed picture is kept iff its family's FlexSED delta score
(ideal step filter of length TAU, zero-padded) has a local maximum >= TH at some frame in [start - 1.0, start + 0.5 + TAU/2];
no FlexSED column for the family -> kept. Nothing re-timed or relabeled. Rescored with score_per_sound; nothing in src/ or
config.py is edited. Primary cell (0.48, 0.15); the 3 x 3 grid is reported and selected among ONLY by split-half CV
(cv_select.py procedure). Pass = the combined rule vs SHIP7 (25/55, 27, 2.451). BEATs-frame variant: record only.

    python benchmark/gold/cpo_screen.py        # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP7")
from benchmark.gold import btp_screen as B
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical

B.ARM = "SHIP7"
TAU, TH = 0.48, 0.15                                  # primary cell (Ebbers 2024 grid middle / lowest abs threshold)
GRID = [(t, h) for t in (0.32, 0.48, 0.64) for h in (0.15, 0.2, 0.3)]
BASE = {"hits": 25, "wrong": 27, "cost": 2.451}
WORK = _ROOT / "data" / "work"
BEATS = {"dev": WORK / "j2_dev_beats", "dev2": WORK / "j2_dev2_beats"}
OUT = _ROOT / "benchmark" / "gold" / "cpo_screen.json"


def family_ev(fr, fam):
    fw, ft, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols:
        return None, ft
    return fw[:, cols].max(axis=1).astype(np.float64), ft


def deltas(ev, ft, tau):
    """step-filter delta, zero-padded at both ends: mean of the next h frames minus mean of the previous h frames"""
    dt = float(ft[1] - ft[0]) if len(ft) > 1 else 0.04
    h = max(1, int(round(tau / 2 / dt)))
    pad = np.concatenate([np.zeros(h), ev, np.zeros(h)])
    cs = np.concatenate([[0.0], np.cumsum(pad)])
    i = np.arange(len(ev)) + h
    post = (cs[i + h] - cs[i]) / h
    pre = (cs[i] - cs[i - h]) / h
    return post - pre


def has_onset(ev, ft, a, tau, th):
    d = deltas(ev, ft, tau)
    lo, hi = a - S.LATE, a + S.EARLY + tau / 2
    n = len(d)
    for i in range(n):
        if not (lo <= ft[i] <= hi) or d[i] < th:
            continue
        left = d[i - 1] if i > 0 else -np.inf
        right = d[i + 1] if i < n - 1 else -np.inf
        if d[i] >= left and d[i] > right:           # local maximum (plateaus count on their first frame)
            return True
    return False


def passes(X, lost_parts):
    gain = X["hits"] - BASE["hits"]; lost = BASE["hits"] - X["hits"]
    old = X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"] and not lost_parts
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * max(lost, 0) and X["hits"] >= BASE["hits"] - 3
    return old, few


def run_cell(P, frames, tau, th):
    rows = {"dev": [], "dev2": []}; dropped = []; lost_parts = []; costs = []
    for pt, st, g, pics in P:
        fr = frames[(pt, st)]
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        keep, gone = [], []
        for l, a, b, _r in pics:
            ev, ft = family_ev(fr, canonical(l)) if fr is not None else (None, None)
            ok = True if ev is None else has_onset(ev, ft, a, tau, th)
            (keep if ok else gone).append((l, a, b))
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
        rows[pt].append(r1); costs.append(DCC.clip_cost(r1))
        for l, a, b in gone:
            dropped.append({"part": pt, "clip": st, "label": l, "start": round(a, 2), "end": round(b, 2), "was": before[(l, round(a, 3))]})
        if r1["hit"] < r0["hit"]:
            lost_parts.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    return X, dropped, lost_parts, np.array(costs)


def cv(C, part, cells, ship):
    """cv_select.py procedure: split-half (10 seeds x 2 halves) and 5-fold (10 seeds), stratified by part"""
    n = len(part); res = {}
    for name, K in (("half", 2), ("fold5", 5)):
        proc, fixed, picks = [], [], {}
        for seed in range(10):
            rng = np.random.default_rng(seed)
            fold = np.empty(n, int)
            for p in (0, 1):
                idx = np.where(part == p)[0]
                rng.shuffle(idx)
                fold[idx] = np.arange(len(idx)) % K
            for k in range(K):
                tr, te = fold != k, fold == k
                best = min(cells, key=lambda c: (round(C[c][tr].mean(), 9), c != ship))
                picks[best] = picks.get(best, 0) + 1
                proc.append(C[best][te].mean()); fixed.append(C[ship][te].mean())
        res[name] = {"procedure": float(np.mean(proc)), "fixed_ship": float(np.mean(fixed)), "picks": picks}
    res["full_argmin"] = min(cells, key=lambda c: (round(C[c].mean(), 9), c != ship))
    return res


def main():
    P = B.parts()
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures")
    base = {"dev": [], "dev2": []}; base_cost = []
    for pt, st, g, pics in P:
        r0 = S.score_clip(g, [p[:3] for p in pics]); base[pt].append(r0); base_cost.append(DCC.clip_cost(r0))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP7: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    BASE["cost"] = Bm["merged"]["cost"]
    flex = {}; beats = {}
    for pt, st, g, pics in P:
        fx = B.FLEX_DIR / f"{st}.npz"; flex[(pt, st)] = DCC.load_fr(fx) if fx.exists() else None
        bx = BEATS[pt] / f"{st}.npz"; beats[(pt, st)] = DCC.load_fr(bx) if bx.exists() else None
    print(f"  clips without FlexSED cache: {sum(v is None for v in flex.values())}, without BEATs: {sum(v is None for v in beats.values())}")
    res = {"base": Bm, "cells": {}}
    C = {"SHIP7": np.array(base_cost)}
    part = np.array([0 if pt == "dev" else 1 for pt, *_ in P])
    for tau, th in GRID:
        X, dropped, lost, costs = run_cell(P, flex, tau, th)
        name = f"CPO({tau},{th})"; C[name] = costs
        old, few = passes(X["merged"], lost)
        verdict = "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
        was = {}
        for d in dropped:
            was[d["was"]] = was.get(d["was"], 0) + 1
        tag = "PRIMARY " if (tau, th) == (TAU, TH) else "        "
        print(f"{tag}{name}: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  dropped {len(dropped)} {was}  "
              f"hits lost {lost} -> {verdict}")
        if (tau, th) == (TAU, TH):
            for d in dropped:
                print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['start']}-{d['end']} was {d['was']}")
        res["cells"][name] = {"tau": tau, "th": th, "rows": X, "dropped": dropped, "hits_lost": lost, "old_rule": old,
                              "fewer_pictures": few, "verdict": verdict, "primary": (tau, th) == (TAU, TH)}
    res["cv"] = cv(C, part, list(C), "SHIP7")
    print(f"CV: half {res['cv']['half']} | fold5 {res['cv']['fold5']} | full argmin {res['cv']['full_argmin']}")
    # record only: the same gate on BEATs tagger frames (primary cell)
    X, dropped, lost, _c = run_cell(P, beats, TAU, TH)
    old, few = passes(X["merged"], lost)
    was = {}
    for d in dropped:
        was[d["was"]] = was.get(d["was"], 0) + 1
    print(f"RECORD CPO-beats({TAU},{TH}): merged {B.fmt(X['merged'])}  dropped {len(dropped)} {was}  hits lost {lost} -> "
          f"{'GO (old rule)' if old else 'GO (fewer-pictures clause)' if few else 'STOP'}")
    res["beats_record"] = {"rows": X, "dropped": dropped, "hits_lost": lost, "old_rule": old, "fewer_pictures": few}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
