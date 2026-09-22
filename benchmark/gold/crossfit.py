"""5-fold cross-fit over the gold clips for a shared-stage knob (amendment 5): the config is
picked on four folds by the BLIND system's own F1 (gate-blind criterion) and scored on the fifth;
the pooled out-of-fold F1 is compared with the declared config's F1 on the same clips. Reads the
per-clip rows of benchmark/gold/detector_dry.py (detector_dry_rows.json).

    python benchmark/gold/crossfit.py [--declared beats@0.35] [--configs beats@0.25 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

ROWS = _ROOT / "benchmark" / "gold" / "detector_dry_rows.json"
GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--declared", default="beats@0.35")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    cfgs = a.configs or sorted(rows)
    gold = S.load_gold([GOLD])
    stems = sorted(st for st in gold if all(st in rows[c] for c in cfgs))
    cats = {st: S.category(gold[st]) for st in stems}
    rng = np.random.default_rng(a.seed)
    folds = [[] for _ in range(a.folds)]
    for cat in sorted(set(cats.values())):            # stratified by clip category
        members = [st for st in stems if cats[st] == cat]
        rng.shuffle(members)
        for i, st in enumerate(members):
            folds[i % a.folds].append(st)
    f1 = lambda c, ss: S.aggregate([rows[c][s] for s in ss])["F1"]
    oof, picks = [], []
    for k in range(a.folds):
        train = [st for j, f in enumerate(folds) if j != k for st in f]
        best = max(cfgs, key=lambda c: f1(c, train))
        picks.append(best)
        oof.extend((st, best) for st in folds[k])
    pooled_tuned = S.aggregate([rows[c][st] for st, c in oof])
    pooled_decl = S.aggregate([rows[a.declared][st] for st in stems])
    # paired bootstrap of the difference (tuned out-of-fold - declared)
    ra = [rows[c][st] for st, c in oof]; rb = [rows[a.declared][st] for st, _ in oof]
    d, lo, hi, p = S.paired_ci(ra, rb)
    print(f"clips {len(stems)} folds {a.folds} configs {len(cfgs)}")
    print("per-fold picks:", picks)
    print(f"declared {a.declared}: F1 {pooled_decl['F1']:.3f} (P {pooled_decl['P']:.2f} R {pooled_decl['R']:.2f})")
    print(f"cross-fitted tuned:  F1 {pooled_tuned['F1']:.3f} (P {pooled_tuned['P']:.2f} R {pooled_tuned['R']:.2f})")
    print(f"dF1 tuned - declared = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P(d>0)={p:.3f}")
    full = max(cfgs, key=lambda c: f1(c, stems))
    print(f"config chosen on all clips (deployed candidate): {full} F1 {f1(full, stems):.3f}")
    out = {"clips": len(stems), "picks": picks, "declared": a.declared, "F1_declared": pooled_decl["F1"],
           "F1_crossfit": pooled_tuned["F1"], "dF1": d, "ci": [lo, hi], "p_gt0": p, "full_pick": full}
    (_ROOT / "benchmark" / "gold" / "crossfit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
