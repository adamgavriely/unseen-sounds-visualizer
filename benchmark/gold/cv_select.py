"""Round 25 (docs/prereg_round13_detector_push.md): 5-fold cross-validated selection over the TIER_SPLIT x LISTENER_LO grid on
merged DEV. Per-clip costs of every cell are recomputed from the saved pictures (DEV 49 in data/work/r13, tagger DEV part in
tagger_prep's folder). Folds stratified by part, seeds 0-9; per fold the cell with the lowest mean training cost (ties -> the
shipped cell) is scored on the held-out clips. Reports the procedure's mean held-out cost vs the fixed shipped cell.

    python benchmark/gold/cv_select.py        # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as DCC

SHIP = "SHIP2+KV4"                                   # = SHIP3, cell (0.6, 0.5)
CELLS = [SHIP, "SHIP3+CV54", "SHIP3+CV64", "SHIP3+CV74", "SHIP3+CV75", "SHIP3+CV76"]
WORK = _ROOT / "data" / "work"


def costs():
    from benchmark.gold import round13_dev as R
    g1, st1 = DCC.dev_stems()
    rows = {c: [] for c in CELLS}
    part = []
    for c in CELLS:
        with R.flags({k: R.arm_cfg(c)[k] for k in R.DISPLAY_KEYS}):
            P = {st: S.load_pictures(WORK / "r13" / f"{c}_proposed", st, "proposed") or [] for st in st1}
        rows[c] += [S.score_clip(g1[st], P[st]) for st in st1]
    part += [0] * len(st1)
    from benchmark.gold import tagger_prep as T
    _D, R2, st2 = T.configure("dev2")
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    keep = set(st2)
    d["clips"] = [c for c in d["clips"] if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = WORK / "cv_gold_tmp.json"
    DCC.dump(tmp, d)
    g2 = T._REAL_LOAD_GOLD([tmp])
    tmp.unlink()
    st2 = [s for s in st2 if s in g2]
    for c in CELLS:
        with R2.flags({k: R2.arm_cfg(c)[k] for k in R2.DISPLAY_KEYS}):
            P = {st: S.load_pictures(T.out("dev2") / f"{c}_proposed", st, "proposed") or [] for st in st2}
        rows[c] += [S.score_clip(g2[st], P[st]) for st in st2]
    part += [1] * len(st2)
    return rows, np.array(part)


def main():
    rows, part = costs()
    C = {c: np.array([DCC.clip_cost(r) for r in rows[c]]) for c in CELLS}
    full = {c: DCC.metrics(rows[c]) for c in CELLS}
    for c in CELLS:
        x = full[c]
        print(f"{c:12s} hits {x['hits']} wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['viewer_cost']:.3f}")
    n = len(part)
    proc, fixed, picks = [], [], {}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        fold = np.empty(n, int)
        for p in (0, 1):
            idx = np.where(part == p)[0]
            rng.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % 5
        for k in range(5):
            tr, te = fold != k, fold == k
            best = min(CELLS, key=lambda c: (round(C[c][tr].mean(), 9), c != SHIP))
            picks[best] = picks.get(best, 0) + 1
            proc.append(C[best][te].mean())
            fixed.append(C[SHIP][te].mean())
    argmin = min(CELLS, key=lambda c: (round(C[c].mean(), 9), c != SHIP))
    out = {"cells": {c: {k: full[c][k] for k in ("hits", "wrong", "visible", "cross", "phantom", "viewer_cost")} for c in CELLS},
           "cv_procedure": float(np.mean(proc)), "cv_fixed_ship": float(np.mean(fixed)), "fold_picks": picks,
           "full_argmin": argmin}
    print(f"CV held-out cost: procedure {np.mean(proc):.3f} vs fixed SHIP3 {np.mean(fixed):.3f}; fold picks {picks}; full argmin {argmin}")
    (_ROOT / "benchmark" / "gold" / "cv_select.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
