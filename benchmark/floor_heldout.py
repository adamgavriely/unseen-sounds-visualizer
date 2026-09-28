"""Confidence floor 0.40 (chosen on DEV) on the held-out 415 AudioSet-Strong clips (docs/prereg_v4.md, 2026-09-28).

    python benchmark/floor_heldout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D

FLOOR = 0.40


def run(set_name):
    R.use_set(set_name)
    cl = D.usable()
    base_rows, base = D.run_cell(cl, {})
    rows = [D.clip_cost(c, [e for e in D.stack(c["id"]) if e.confidence >= FLOOR]) for c in cl]
    minutes = sum(c["duration"] for c in cl) / 60.0
    import numpy as np
    cell = {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
            "fp_per_min": sum(r["fp"] for r in rows) / minutes,
            "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / max(1, sum(r["n_conseq"] for r in rows))}
    d = D.boot([a["C_overlap"] - b["C_overlap"] for a, b in zip(rows, base_rows)])
    return {"clips": len(cl), "shipped": base, "floor": cell, "dC_overlap": d, "pass": d[2] < 0}


def main():
    res = {"calib_280": run("calib"), "heldout": run("heldout")}
    (_ROOT / "benchmark" / "floor_heldout.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"{k}: {v['clips']} clips; C-overlap shipped {v['shipped']['C_overlap']:.3f} -> floor {v['floor']['C_overlap']:.3f}; "
              f"recall {v['shipped']['recall_overlap']:.1%} -> {v['floor']['recall_overlap']:.1%}; false/min "
              f"{v['shipped']['fp_per_min']:.2f} -> {v['floor']['fp_per_min']:.2f}; dC {v['dC_overlap'][0]:+.3f} "
              f"[{v['dC_overlap'][1]:+.3f}, {v['dC_overlap'][2]:+.3f}] pass {v['pass']}")


if __name__ == "__main__":
    main()
