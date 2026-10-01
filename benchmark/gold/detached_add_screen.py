"""Round 39 DETACHED-ADD screen (docs/prereg_round13_detector_push.md "Round 39 DETACHED-ADD"), CPU only, on the saved SHIP8
proposed pictures of merged DEV, scored on gold_AG.json. For every family a clip already draws with a NON-rescued picture, find
runs of that family (max over same-family columns) in any of the three frame caches at the WEAK bars (BEATs >= AED_THRESHOLD
0.175, FlexSED >= FLEXSED_VETO 0.3, DASM >= LISTENER_DASM_BAR 0.575; _runs over LISTEN_RUN_GAP; length >= 0.24 s) whose onset is
> 3.0 s from every same-family picture start AND detached from every same-family picture by >= MERGE_GAP (1.5 s) of "below bar
in all three models"; add ONE picture per (clip, family) at the earliest such onset, 2 s long. Nothing in src/ or config.py is
edited. Base (SHIP8) must reproduce 28/58, 21 (6/13/2), 2.282 first.

    TG_ARMS=SHIP8 python benchmark/gold/detached_add_screen.py        # from ~/MscProj_tg
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
from benchmark.gold import gbtp_screen as G
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical
from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP

B.ARM = "SHIP8"
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}
FAR, MIN_RUN, PIC_LEN, GRID = 3.0, LISTEN_RUN_GAP, 2.0, 0.04
_CFG = R.arm_cfg("SHIP8")
BARS = {"flex": float(_CFG["FLEXSED_VETO"]), "beats": float(config.AED_THRESHOLD), "dasm": float(config.LISTENER_DASM_BAR)}
DETACH = 1.5                     # the shipped MERGE_GAP (config.py ship setter, line ~502); arm_cfg scores pictures with 2.0
assert BARS == {"flex": 0.3, "beats": 0.175, "dasm": 0.575}, BARS
assert DETACH == 1.5 and abs(LISTEN_RUN_GAP - 0.24) < 1e-9 and abs(config.DISPLAY_THRESHOLD / 2 - BARS["beats"]) < 1e-9
OUT = _ROOT / "benchmark" / "gold" / "detached_add_screen.json"


def evidence(fr, fam):
    """(times, family evidence = max over same-family columns) or None"""
    if fr is None:
        return None
    fw, ft, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols:
        return None
    return np.asarray(ft, dtype=np.float64), fw[:, cols].max(axis=1).astype(np.float64)


def runs_of(ev, bar):
    if ev is None:
        return []
    ft, e = ev
    rr, dt = _runs(e, ft, bar, LISTEN_RUN_GAP)
    return [(float(ft[i]), float(ft[j - 1]) + dt) for i, j in rr]


def detached(run, pics, evs):
    s, e = run
    for a, b in pics:
        if e <= a:                                   # run precedes the picture
            if not any_window(evs, e, a):
                return False
        elif s >= b:                                 # run follows the picture
            if not any_window(evs, b, s):
                return False
        else:
            return False                             # overlaps the picture
    return True


def any_window(evs, t0, t1):
    """is there a stretch >= DETACH inside [t0, t1] where every model is below bar? scan the grid for the longest quiet stretch"""
    if t1 - t0 < DETACH - 1e-6:
        return False
    ts = np.arange(t0, t1 + 1e-9, GRID)
    loud = np.zeros(len(ts), dtype=bool)
    for m, ev in evs.items():
        if ev is None:
            continue
        ft, e = ev
        idx = np.clip(np.searchsorted(ft, ts), 0, len(ft) - 1)
        idx2 = np.clip(idx - 1, 0, len(ft) - 1)
        near = np.where(np.abs(ft[idx] - ts) <= np.abs(ft[idx2] - ts), idx, idx2)
        loud |= e[near] >= BARS[m]
    best = cur = 0
    for v in loud:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return (best - 1) * GRID >= DETACH - 1e-6


def candidate(cs, fam, fam_pics):
    """(onset, {model: onset}) of the earliest qualifying run over the three models, or (None, {})"""
    evs = {m: evidence(fr, fam) for m, fr in cs.items()}
    found = {}
    for m in BARS:
        for s, e in runs_of(evs[m], BARS[m]):
            if e - s < MIN_RUN - 1e-6:
                continue
            if any(abs(s - a) <= FAR for a, b in fam_pics):
                continue
            if not detached((s, e), fam_pics, evs):
                continue
            if m not in found or s < found[m]:
                found[m] = s
    return (min(found.values()) if found else None), found


def main():
    P = B.parts()
    n_resc = sum(r for *_x, pics in P for *_y, r in pics)
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {n_resc} rescued; bars {BARS}, far {FAR} s, detach {DETACH} s")
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    n_clips = len(P)

    rows = {"dev": [], "dev2": []}; added = []; lost = []; dups = {"base": 0, "new": 0}
    missing = {m: 0 for m in BARS}; by_model = {m: 0 for m in BARS}; n_fam = 0
    for pt, st, g, pics in P:
        cs = G.caches(pt, st)
        for m in BARS:
            missing[m] += cs[m] is None
        old = [p[:3] for p in pics]
        new = list(old)
        fams = {}
        for l, a, b, resc in pics:
            if not resc:
                fams.setdefault(canonical(l), l)
        n_fam += len(fams)
        adds = []
        for fam, lab in fams.items():
            fam_pics = [(a, b) for l, a, b, _ in pics if canonical(l) == fam]
            s, found = candidate(cs, fam, fam_pics)
            if s is None:
                continue
            src = min(found, key=found.get)
            by_model[src] += 1
            new.append((lab, s, s + PIC_LEN))
            adds.append((lab, fam, s, found, src))
        r0 = S.score_clip(g, old); r1 = S.score_clip(g, new)
        rows[pt].append(r1); dups["base"] += r0["dup"]; dups["new"] += r1["dup"]
        if adds:
            c1 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
            for lab, fam, s, found, src in adds:
                added.append({"part": pt, "clip": st, "label": lab, "family": fam, "start": round(s, 2), "end": round(s + PIC_LEN, 2),
                              "onsets": {m: round(v, 2) for m, v in found.items()}, "earliest_from": src,
                              "class": c1[(lab, round(s, 3))], "clip_before": oc(r0), "clip_after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    gain = X["merged"]["hits"] - BASE["hits"]
    c1w = {w: G.cost_w(X["merged"], n_clips, w) for w in (1, 2)}
    c0w = {w: G.cost_w(Bm["merged"], n_clips, w) for w in (1, 2)}
    main_go = X["merged"]["hits"] >= 29 and X["merged"]["wrong"] <= BASE["wrong"] + 2 * gain and c1w[2] < c0w[2] and not lost
    fa_ok = all(X[pt][k] <= Bm[pt][k] for pt in ("dev", "dev2") for k in ("cross", "phantom"))
    more_go = gain > 0 and fa_ok and c1w[1] < c0w[1]
    print(f"DETACHED-ADD: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  added {len(added)} over "
          f"{n_fam} (clip, family) pairs (earliest from {by_model})  caches missing {missing}  dups {dups}  hits lost {lost}")
    print(f"  main rule: {'GO' if main_go else 'STOP'}   more-hits rule: {'PASS' if more_go else 'FAIL'}   "
          f"cost w=1 {c0w[1]:.3f} -> {c1w[1]:.3f}, w=2 {c0w[2]:.3f} -> {c1w[2]:.3f}")
    for d in added:
        print(f"   {d['part']:4s} {d['clip']} {d['label']} +{d['start']}-{d['end']} ({d['earliest_from']}, onsets {d['onsets']}): "
              f"{d['class']}   clip {d['clip_before']} -> {d['clip_after']}")
    res = {"base": Bm, "bars": BARS, "far": FAR, "detach": DETACH, "rows": X, "added": added, "hits_lost": lost, "dups": dups,
           "caches_missing": missing, "earliest_from": by_model, "n_pairs": n_fam,
           "cost": {str(w): {"base": round(c0w[w], 3), "new": round(c1w[w], 3)} for w in (1, 2)},
           "main_GO": main_go, "more_hits_PASS": more_go}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
