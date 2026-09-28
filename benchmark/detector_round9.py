"""Detector round 9 (docs/prereg_round9_contrast.md, 2026-09-28): local-contrast veto on BEATs-only spans.

  J1  a shown BEATs-only span (BEATs origin, no FlexSED twin) is dropped if mean in-span family BEATs score minus the mean
      over the +-3 s flanks < 0.1 (no flank frames -> kept)
  J2  J1 + round 8's I7 (FlexSED-only spans, margin 0.2 on FlexSED's scale)

Reuses detector_round8 (stack, gate 0, scoring, bootstrap, Holm). Reads the caches only.

    python benchmark/detector_round9.py fit       # gate 0 on the 280, J1 and J2, the picks
    python benchmark/detector_round9.py heldout   # the picks on the 415 (+ Holm)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import detector_round8 as M

OUT = _ROOT / "benchmark" / "detector_round9.json"
J_MARGIN, FLANK = 0.1, 3.0
CELLS = ["J1", "J2"]


def j1(x, events, info, log):
    fw, ts, labs = x.b
    keep = []
    for e in events:
        if id(e) not in info["flex"] and id(e) not in info["twinned"]:
            cols = M.cols_for(labs, M.key(e))
            if cols:
                s = fw[:, cols].max(axis=1)
                ins = (ts >= e.start) & (ts < e.end)
                fl = ((ts >= e.start - FLANK) & (ts < e.start)) | ((ts >= e.end) & (ts < e.end + FLANK))
                log["beats_only"] += 1
                if not fl.any():
                    log["no_flank"] += 1
                elif ins.any() and s[ins].mean() - s[fl].mean() < J_MARGIN:
                    log["dropped"] += 1
                    continue
        keep.append(e)
    return keep


def run(cell, xs, log):
    out = []
    for x in xs:
        ev, info = M.stack8(x, {"i7": cell == "J2"}, tag=True)
        out.append(j1(x, ev, info, log))
    return out


def one_set(set_name, cells, log, key):
    cl, xs, base_evs, brows, brows_d, Sb = M.gate_set(set_name, log)
    base = (base_evs, brows, brows_d, M.hbd_flags(xs, base_evs))
    R = log.setdefault(key, {})
    R["baseline"] = Sb
    for cell in cells:
        jl = {"beats_only": 0, "no_flank": 0, "dropped": 0}
        evs = run(cell, xs, jl)
        rows, _rd, S = M.score(xs, evs, base)
        S["j1_log"] = jl
        if key == "heldout":
            S["pass"] = bool(S["dC_overlap"][2] < 0)
            d = [a["C_overlap"] - b["C_overlap"] for a, b in zip(rows, brows)]
            S["strata"] = {s: {"n": len(ix), "dC_overlap": M.boot8([d[i] for i in ix]) if ix else None}
                           for s, ix in ((s, [i for i, x in enumerate(xs) if x.c.get("stratum") == s]) for s in ("complex", "random"))}
        R[cell] = S
        M.show(cell, S); print(f"   J1 log {jl}" + (f" -> {'PASS' if S['pass'] else 'fail'}" if key == "heldout" else ""), flush=True)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return R, Sb


def fit(log):
    R, Sb = one_set("calib", CELLS, log, "fit")
    log["picks"] = [k for k in CELLS if R[k]["C_overlap"] < Sb["C_overlap"] and R[k]["C_onset"] < Sb["C_onset"]]
    print(f"[picks] to the 415: {log['picks']}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(log):
    if not log.get("picks"):
        log["heldout"] = "no cell picked on the 280"; print(log["heldout"])
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8"); return
    H, _Sb = one_set("heldout", log["picks"], log, "heldout")
    H["holm"] = M.holm({k: H[k]["dC_overlap"][3] for k in log["picks"]})
    print(f"[holm] {H['holm']}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fit", "heldout"))
    a = ap.parse_args()
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    {"fit": fit, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
