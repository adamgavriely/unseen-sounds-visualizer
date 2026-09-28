"""Detector round 3 (docs/prereg_v4.md, 2026-09-28): a FlexSED bar that depends on speech/music at the candidate's time, and a
better second-opinion veto. Six fixed cells, no grid; pick on the 280 (fit set) by cost C, then one held-out test (415).

  C0 shipped: FlexSED bar 0.8, PANNs veto (clip peak >= 0.05)
  C1 bar 0.6, PANNs relative veto (clip peak >= that class's 95th percentile of frame scores over the 280)
  C2 bar 0.6, PretrainedSED local veto (score within +-1 s >= that class's 95th percentile over the 280)
  C3 bar 0.5, PretrainedSED local veto
  C4 bar 0.6 in quiet frames / 0.8 under speech or music, shipped PANNs veto
  C5 bar 0.5 in quiet / 0.7 under speech or music, PretrainedSED local veto
"Under speech or music" = BEATs Speech or Music >= 0.3 in any frame of the candidate span. Vetoes apply to FlexSED-only
spans, as shipped. Cost C, pick rule and held-out test exactly as amendment 24 (lowest C-overlap that also lowers C-onset
below C0's; held-out passes iff the paired bootstrap upper 95 % CI of dC-overlap < 0).

    python benchmark/detector_round3.py fit
    python benchmark/detector_round3.py heldout --cell C2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from src.labels import canonical
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "detector_round3.json"
CELLS = {"C0": (0.8, 0.8, "panns"), "C1": (0.6, 0.6, "panns_rel"), "C2": (0.6, 0.6, "psed"), "C3": (0.5, 0.5, "psed"),
         "C4": (0.6, 0.8, "panns"), "C5": (0.5, 0.7, "psed")}
P95 = {}


def psed_dir():
    return R.E.WIN / "psed"


def class_p95(kind):
    """95th percentile of each class's frame scores over the 280 (the fit set), keyed by label"""
    if kind in P95:
        return P95[kind]
    R.use_set("calib")
    root = R.PANNS if kind == "panns" else psed_dir()
    stacks, labs = [], None
    for c in D.usable():
        p = root / f"{c['id']}.npz"
        if p.exists():
            fw, _t, labs = R.load(p)
            stacks.append(fw)
    allf = np.concatenate(stacks)
    P95[kind] = {l: max(float(np.percentile(allf[:, i], 95)), 1e-3) for i, l in enumerate(labs)}
    return P95[kind]


def speechy(b, e, thr=0.3):
    fw, ts, labs = b
    m = (ts >= e.start) & (ts <= e.end)
    cols = [i for i, l in enumerate(labs) if l in ("Speech", "Music")]
    return bool(m.any() and cols and float(fw[m][:, cols].max()) >= thr)


def stack3(cid, q_bar, m_bar, veto, p95p, p95s):
    b, f, p, _pe = D.caches(cid)
    AED, HYS, DISP = D.AED, D.HYS, D.DISP
    events = _extract_events(b[0], b[1], b[2], AED, None, config.AED_MIN_DUR, low=AED * HYS)
    lo = min(q_bar, m_bar)
    fev = [e for e in _extract_events(f[0], f[1], f[2], lo, None, config.AED_MIN_DUR, low=lo * HYS)
           if e.confidence >= (m_bar if speechy(b, e) else q_bar)]
    key = lambda e: canonical(e.label)
    fresh = []
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
        else:
            fresh.append(e)
    flex_ids = {id(e) for e in fresh}
    events = events + fresh
    fpk = D.clip_peak(f)
    events = [e for e in events if fpk.get(key(e), 1.0) >= 0.3]
    if veto == "panns":
        ppk = D.clip_peak(p)
        events = [e for e in events if id(e) not in flex_ids or ppk.get(key(e), 1.0) >= 0.05]
    elif veto == "panns_rel":
        fw, _t, labs = p
        pk = fw.max(axis=0)
        ok = {}
        for i, l in enumerate(labs):
            ok[canonical(l)] = ok.get(canonical(l), False) or float(pk[i]) >= p95p[l]
        events = [e for e in events if id(e) not in flex_ids or ok.get(key(e), True)]
    else:
        sp = R.load(psed_dir() / f"{cid}.npz")
        fw, ts, labs = sp
        def heard(e):
            m = (ts >= e.start - 1.0) & (ts <= e.end + 1.0)
            cols = [i for i, l in enumerate(labs) if canonical(l) == key(e)]
            if not cols:
                return True                       # the veto has no such class: it cannot object (as the shipped veto)
            return bool(m.any() and any(float(fw[m][:, i].max()) >= p95s[labs[i]] for i in cols))
        events = [e for e in events if id(e) not in flex_ids or heard(e)]
    return [e for e in events if e.confidence >= DISP]


def run(cl, cell, p95p, p95s):
    q, m, v = CELLS[cell]
    rows = [D.clip_cost(c, stack3(c["id"], q, m, v, p95p, p95s)) for c in cl]
    minutes = sum(c["duration"] for c in cl) / 60.0
    return rows, {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
                  "fp_per_min": sum(r["fp"] for r in rows) / minutes,
                  "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / max(1, sum(r["n_conseq"] for r in rows))}


def usable_psed():
    return [c for c in D.usable() if (psed_dir() / f"{c['id']}.npz").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fit", "heldout"))
    ap.add_argument("--cell")
    a = ap.parse_args()
    p95p, p95s = class_p95("panns"), class_p95("psed")
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if a.step == "fit":
        R.use_set("calib")
        cl = usable_psed()
        res = {k: run(cl, k, p95p, p95s)[1] for k in CELLS}
        base = res["C0"]
        ok = {k: v for k, v in res.items() if k != "C0" and v["C_overlap"] < base["C_overlap"] and v["C_onset"] < base["C_onset"]}
        pick = min(ok, key=lambda k: ok[k]["C_overlap"]) if ok else None
        log.update({"clips": len(cl), "fit": res, "pick": pick})
        for k, v in res.items():
            print(f"{k}: C-overlap {v['C_overlap']:.3f} C-onset {v['C_onset']:.3f} recall {v['recall_overlap']:.1%} false/min {v['fp_per_min']:.2f}")
        print("PICK:", pick)
    else:
        assert a.cell and a.cell == log.get("pick"), "only the pick goes to held-out"
        R.use_set("heldout")
        cl = usable_psed()
        b_rows, b = run(cl, "C0", p95p, p95s)
        c_rows, c = run(cl, a.cell, p95p, p95s)
        d = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(c_rows, b_rows)])
        log["heldout"] = {"cell": a.cell, "clips": len(cl), "C0": b, "cell_summary": c, "dC_overlap": d, "pass": d[2] < 0}
        print(f"held-out {len(cl)} clips: {a.cell} dC-overlap {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] pass {d[2] < 0}; "
              f"recall {b['recall_overlap']:.1%} -> {c['recall_overlap']:.1%}; false/min {b['fp_per_min']:.2f} -> {c['fp_per_min']:.2f}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
