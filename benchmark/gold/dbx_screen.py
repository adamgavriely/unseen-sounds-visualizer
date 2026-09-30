"""Round 35 DBX screen (DASM back-extension of late picture starts; docs/prereg_round13_detector_push.md "Round 35 DBX"),
CPU only, on the saved SHIP7 (= SHIP6+FLAP) proposed pictures of merged DEV. A placed NON-rescued picture whose family's DASM
frame score (max over the family's cache columns) is >= 0.575 (LISTENER_DASM_BAR) at its start, in a run (gaps <= LISTEN_RUN_GAP
merged, the pipeline's _runs) that begins >= 1.0 s (MAX_AFTER_END) earlier, gets its start moved back to that run's start.
DBX-S (report only): also untouched when the moved picture would overlap an earlier placed picture of the same family.
Rescored with score_per_sound; nothing in src/ or config.py is edited. Pass = the combined rule vs SHIP7 (25/55, 27, 2.451).

    TG_ARMS=SHIP6+FLAP python benchmark/gold/dbx_screen.py        # from ~/MscProj_tg
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
from src.labels import canonical
from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP

B.ARM = "SHIP6+FLAP"
BAR, MIN_MOVE = 0.575, 1.0                                  # LISTENER_DASM_BAR (F8's bar), MAX_AFTER_END (shipped 1.0)
BASE = {"hits": 25, "wrong": 27, "cost": 2.451}
OUT = _ROOT / "benchmark" / "gold" / "dbx_screen.json"


def family_runs(fr, fam):
    """(runs [(s, r)], evidence present?) for the family's DASM columns"""
    if fr is None:
        return None
    fw, ft, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols:
        return None
    ev = fw[:, cols].max(axis=1).astype(np.float64)
    rr, dt = _runs(ev, ft, BAR, LISTEN_RUN_GAP)
    return [(float(ft[i]), float(ft[j - 1] + dt)) for i, j in rr]


def new_start(a, runs):
    if runs is None:
        return a, "no DASM evidence"
    cov = [(s, r) for s, r in runs if s - 1e-6 <= a <= r + 1e-6]
    if not cov:
        return a, "DASM < bar at start"
    s = min(cov)[0]
    if a - s < MIN_MOVE:
        return a, f"run starts {a - s:.2f} s earlier (< {MIN_MOVE})"
    return s, "moved"


def passes(X, lost_parts):
    gain = X["hits"] - BASE["hits"]; lost = BASE["hits"] - X["hits"]
    old = X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"] and not lost_parts
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * max(lost, 0) and X["hits"] >= BASE["hits"] - 3
    return old, few


def run_variant(P, dasm, strict):
    rows = {"dev": [], "dev2": []}; moved = []; lost_parts = []; silent = {"no DASM evidence": 0, "DASM < bar at start": 0, "short": 0,
                                                                            "overlap (DBX-S)": 0}
    for pt, st, g, pics in P:
        fr = dasm[(pt, st)]
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        new = []
        for l, a, b, resc in pics:
            fam = canonical(l)
            a2, why = (a, "rescued") if resc else new_start(a, family_runs(fr, fam))
            if why == "moved" and strict:
                earlier = [(l2, a3, b3) for l2, a3, b3, _r in pics if canonical(l2) == fam and a3 < a - 1e-6 and b3 > a2 + 1e-6]
                if earlier:
                    a2, why = a, "overlap (DBX-S)"
            if why not in ("moved", "rescued"):
                silent["short" if why.startswith("run starts") else why] += 1
            new.append((l, a2, b))
            if why == "moved":
                moved.append({"part": pt, "clip": st, "label": l, "family": fam, "old_start": round(a, 2), "new_start": round(a2, 2),
                              "end": round(b, 2), "before": before[(l, round(a, 3))]})
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, new)
        rows[pt].append(r1)
        mine = [m for m in moved if m["part"] == pt and m["clip"] == st and "after" not in m]
        if mine:
            after = classify(g, new)
            oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
            for m in mine:
                m["after"] = next(k for l, a, b, k, _ in after if l == m["label"] and abs(a - m["new_start"]) < 0.006)
                m["clip_before"], m["clip_after"] = oc(r0), oc(r1)
        if r1["hit"] < r0["hit"]:
            lost_parts.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    return X, moved, lost_parts, silent


def main():
    P = B.parts()
    n_resc = sum(r for *_x, pics in P for *_y, r in pics)
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {n_resc} rescued; DASM bar {BAR}, min move {MIN_MOVE}")
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP7: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    BASE["cost"] = Bm["merged"]["cost"]
    dasm = {}
    for pt, st, g, pics in P:
        dx = CG.PARTS[pt]["dasm"] / f"{st}.npz"; dasm[(pt, st)] = DCC.load_fr(dx) if dx.exists() else None
    print(f"  clips without DASM cache: {sum(v is None for v in dasm.values())}")
    res = {"base": Bm, "bar": BAR, "min_move": MIN_MOVE}
    for name, strict in (("DBX", False), ("DBX-S", True)):
        X, moved, lost, silent = run_variant(P, dasm, strict)
        old, few = passes(X["merged"], lost)
        verdict = "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
        print(f"{name}: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  moved {len(moved)}  "
              f"untouched {silent}  hits lost {lost} -> {verdict}")
        for m in moved:
            print(f"   {m['part']:4s} {m['clip']} {m['label']} {m['old_start']} -> {m['new_start']} (end {m['end']}): "
                  f"{m['before']} -> {m['after']}   clip {m['clip_before']} -> {m['clip_after']}")
        res[name] = {"rows": X, "moved": moved, "hits_lost": lost, "untouched": silent, "old_rule": old, "fewer_pictures": few,
                     "verdict": verdict, "decisive": not strict}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
