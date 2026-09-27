"""Amendment 24 (docs/prereg_v4.md): detector round 2 -- PE-A-Frame screen, cells on the 280 (fit set) scored by the cost
C, one pick, and the held-out test of that pick. Stage-4 logic from cached frame scores, as detector_round_stage0.py.

    python benchmark/detector_round2.py screen              # PE percentile grid + E-delta pool screen (280)
    python benchmark/detector_round2.py fit                 # all a-priori cells on the 280, C per cell, the pick
    python benchmark/detector_round2.py heldout --cell NAME # the pick vs shipped on the held-out set, paired bootstrap

C per clip = 4 x consequential events with no same-family span overlapping (>= 0.5 s or half the event) + 2 x false spans
(C-overlap, primary); C-onset uses a span starting in [onset - 0.5, onset + 1.0] s instead of the overlap.
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
from src.labels import canonical, is_salient_nonspeech, is_music
from src.stage4_audio_event_detection import _extract_events

DISP, AED, HYS, WIN = 0.35, 0.175, 1.0, 1.0
OUT = _ROOT / "benchmark" / "detector_round2.json"


# ----------------------------------------------------------------------------- caches
def caches(cid):
    b = R.load(R.E.WIN / "beats" / f"{cid}.npz")
    f = R.load(R.FLEX / f"{cid}.npz")
    p = R.load(R.PANNS / f"{cid}.npz")
    pe_p = R.PEF / f"{cid}.npz"
    pe = R.load(pe_p) if pe_p.exists() else None
    return b, f, p, pe


def peak_near(fr, e, win=WIN):
    """highest same-family frame score within [start - win, end + win]; 0 when the family is not queried"""
    fw, ts, labs = fr
    m = (ts >= e.start - win) & (ts <= e.end + win)
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(e.label)]
    if not cols or not m.any():
        return 0.0
    return float(fw[m][:, cols].max())


def clip_peak(fr):
    fw, _t, labs = fr
    out = {}
    for i, lab in enumerate(labs):
        k = canonical(lab); out[k] = max(out.get(k, 0.0), float(fw[:, i].max()))
    return out


# ----------------------------------------------------------------------------- the stack
def stack(cid, fbar=0.8, t2=None, t3=None, pveto=("panns", 0.05), weak="absorb", upe=None, keep_raw=False):
    """t2 = (beats_min, panns_min) on FlexSED-only spans; t3 = callable(event, caches) -> promote?; pveto = ('panns', tau)
    or ('pe', theta) on FlexSED-only spans; upe = theta_u: PE-A-Frame spans added when FlexSED >= 0.3 agrees within 1 s."""
    b, f, p, pe = caches(cid)
    events = _extract_events(b[0], b[1], b[2], AED, None, config.AED_MIN_DUR, low=AED * HYS)
    fev = _extract_events(f[0], f[1], f[2], fbar, None, config.AED_MIN_DUR, low=fbar * HYS)
    key = lambda e: canonical(e.label)
    fresh = []
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if weak == "ignore":
            tw = [x for x in tw if x.confidence >= DISP]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
        else:
            fresh.append(e)
    if t2 and fresh:
        fresh = [e for e in fresh if peak_near(b, e) >= t2[0] or peak_near(p, e) >= t2[1]]
    flex_ids = {id(e) for e in fresh}
    events = events + fresh
    fpk = clip_peak(f)
    events = [e for e in events if fpk.get(key(e), 1.0) >= 0.3]                       # FlexSED veto (clip level)
    if pveto[0] == "panns":
        ppk = clip_peak(p)
        events = [e for e in events if id(e) not in flex_ids or ppk.get(key(e), 1.0) >= pveto[1]]
    else:                                                                               # PE-A-Frame replaces the PANNs veto
        events = [e for e in events if id(e) not in flex_ids or peak_near(pe, e) >= pveto[1]]
    if t3:
        for e in events:
            if id(e) not in flex_ids and e.confidence < DISP and t3(e, b, f, p, pe):
                e.confidence = DISP
    out = [e for e in events if e.confidence >= DISP]
    if upe is not None and pe is not None:
        pev = _extract_events(pe[0], pe[1], pe[2], upe, None, config.AED_MIN_DUR, low=upe * HYS)
        for e in pev:
            if any(key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end for x in out):
                continue
            if peak_near(f, e) >= 0.3:
                e.confidence = max(e.confidence, DISP); out.append(e)
    return out


def t3_or(fmin, pmin):
    return lambda e, b, f, p, pe: peak_near(f, e) >= fmin or peak_near(p, e) >= pmin


def t3_and(fmin, pmin):
    return lambda e, b, f, p, pe: peak_near(f, e) >= fmin and peak_near(p, e) >= pmin


def t3_pe(theta):
    return lambda e, b, f, p, pe: pe is not None and peak_near(pe, e) >= theta


def cells(theta=None, theta_u=None):
    c = {
        "shipped":  dict(),
        "F":        dict(weak="ignore"),
        "E":        dict(fbar=0.5, t2=(0.1, 0.05), t3=t3_or(0.3, 0.05)),
        "E-F 0.3":  dict(fbar=0.5, t2=(0.1, 0.05), t3=t3_or(0.3, 1.01)),
        "E-F 0.5":  dict(fbar=0.5, t2=(0.1, 0.05), t3=t3_or(0.5, 1.01)),
        "E-AND":    dict(fbar=0.5, t2=(0.1, 0.05), t3=t3_and(0.3, 0.05)),
    }
    if theta is not None:
        c["D+PE"] = dict(fbar=0.5, t2=(0.1, 0.05), pveto=("pe", theta))
        c["E+PE"] = dict(fbar=0.5, t2=(0.1, 0.05), pveto=("pe", theta), t3=t3_pe(theta))
    if theta_u is not None:
        c["U+PE"] = dict(upe=theta_u)
    return c


# ----------------------------------------------------------------------------- scoring
def _ev(ev):
    return [e for e in ev if is_salient_nonspeech(e.label) and not is_music(e.label)]


def clip_cost(c, ev):
    ev = _ev(ev)
    gold = [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"]) and g["consequential"]]
    miss_ov = sum(not any(R.E._same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev) for g in gold)
    miss_on = sum(not any(R.E._same(e.label, g["label"]) and g["start"] - R.EARLY <= e.start <= g["start"] + R.LATE for e in ev) for g in gold)
    fp = sum(not any(R.E._same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"]) for e in ev)
    return {"C_overlap": 4 * miss_ov + 2 * fp, "C_onset": 4 * miss_on + 2 * fp, "fp": fp, "miss_overlap": miss_ov,
            "miss_onset": miss_on, "n_conseq": len(gold), "masked": [g["masked"] for g in gold]}


def usable():
    """clips with all four caches (BEATs, FlexSED, PANNs, PE-A-Frame), so every cell is scored on the same clips"""
    need = lambda c: all((d / f"{c['id']}.npz").exists() for d in (R.E.WIN / "beats", R.FLEX, R.PANNS, R.PEF))
    return [c for c in R.E.clips() if need(c)]


def run_cell(cl, spec):
    rows = [clip_cost(c, stack(c["id"], **spec)) for c in cl]
    minutes = sum(c["duration"] for c in cl) / 60.0
    return rows, {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
                  "fp_per_min": sum(r["fp"] for r in rows) / minutes,
                  "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / max(1, sum(r["n_conseq"] for r in rows)),
                  "recall_onset": 1 - sum(r["miss_onset"] for r in rows) / max(1, sum(r["n_conseq"] for r in rows))}


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------- steps
def screen(cl, log):
    pe_ok = [c for c in cl if (R.PEF / f"{c['id']}.npz").exists()]
    vals = np.concatenate([R.load(R.PEF / f"{c['id']}.npz")[0].ravel() for c in pe_ok])
    grid = [float(np.percentile(vals, q)) for q in (50, 80, 90, 95, 99)]
    log["pe_grid"] = {"percentiles": [50, 80, 90, 95, 99], "theta": grid, "clips": len(pe_ok)}
    print(f"[screen] PE-A-Frame percentile grid over {len(pe_ok)} clips: {[round(g, 4) for g in grid]}")
    pool = []                                           # E-delta: spans E raises that shipped does not show
    E_spec = cells()["E"]
    for c in pe_ok:
        sh = stack(c["id"]); ee = stack(c["id"], **E_spec)
        b, f, p, pe = caches(c["id"])
        for e in _ev(ee):
            if any(canonical(x.label) == canonical(e.label) and min(x.end, e.end) - max(x.start, e.start) > 0 for x in sh):
                continue
            true = any(R.E._same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"])
            pool.append((true, peak_near(pe, e), peak_near(p, e)))
    y = np.array([t for t, _, _ in pool]); spe = np.array([s for _, s, _ in pool]); spa = np.array([s for _, _, s in pool])
    log["pool"] = {"n": int(len(y)), "true": int(y.sum()), "false": int((~y).sum())}
    print(f"[screen] E-delta pool: {len(y)} spans, {int(y.sum())} true, {int((~y).sum())} false (written before scoring)")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")

    def auroc(s, yy):
        pos, neg = s[yy], s[~yy]
        if not len(pos) or not len(neg):
            return float("nan")
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y)); bs.append((auroc(spe[i], y[i]), auroc(spa[i], y[i])))
    bs = np.array(bs)
    log["auroc"] = {"pe": auroc(spe, y), "pe_ci": [float(np.nanpercentile(bs[:, 0], 2.5)), float(np.nanpercentile(bs[:, 0], 97.5))],
                    "panns": auroc(spa, y), "panns_ci": [float(np.nanpercentile(bs[:, 1], 2.5)), float(np.nanpercentile(bs[:, 1], 97.5))]}
    ok_theta = None
    for th in sorted(grid, reverse=True):
        kt = int((spe[y] >= th).sum()); kf = int((spe[~y] >= th).sum())
        log.setdefault("keep", []).append({"theta": th, "kept_true": kt, "kept_false": kf})
    cand = [k for k in log["keep"] if k["kept_true"] >= 0.8 * y.sum()]
    if cand:
        k = max(cand, key=lambda k: k["theta"])         # the operating point at the 80 % recall floor (clarification 1)
        ratio = k["kept_false"] / max(1, k["kept_true"])
        log["screen_point"] = {**k, "ratio": ratio}
        ok_theta = k["theta"] if ratio <= 0.84 else None
    log["screen_pass"] = ok_theta is not None
    print(f"[screen] AUROC PE {log['auroc']['pe']:.3f} {log['auroc']['pe_ci']}, PANNs {log['auroc']['panns']:.3f}; "
          f"point {log.get('screen_point')}; pass {log['screen_pass']}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return grid, log["screen_pass"]


def fit(cl, log):
    grid = log["pe_grid"]["theta"]
    res = {}
    base_rows, base = run_cell(cl, {})
    res["shipped"] = base
    for name, spec in cells().items():
        if name != "shipped":
            res[name] = run_cell(cl, spec)[1]
    if log.get("screen_pass"):
        for which in ("D+PE", "E+PE"):
            best = min(((th, run_cell(cl, cells(theta=th)[which])[1]) for th in grid), key=lambda x: x[1]["C_overlap"])
            res[which] = {**best[1], "theta": best[0]}
    best_u = min(((th, run_cell(cl, cells(theta_u=th)["U+PE"])[1]) for th in grid), key=lambda x: x[1]["C_overlap"])
    res["U+PE"] = {**best_u[1], "theta_u": best_u[0]}
    for k, v in res.items():
        v["label"] = "strict pass" if v["fp_per_min"] <= base["fp_per_min"] else "trade pass"
    ok = {k: v for k, v in res.items() if k != "shipped" and v["C_onset"] < base["C_onset"] and v["C_overlap"] < base["C_overlap"]}
    pick = min(ok, key=lambda k: ok[k]["C_overlap"]) if ok else None
    log["fit"] = res; log["pick"] = pick
    print(f"{'cell':10s} {'C-overlap':>9} {'C-onset':>8} {'recall':>7} {'onset-rec':>9} {'false/min':>9}")
    for k, v in res.items():
        print(f"{k:10s} {v['C_overlap']:9.3f} {v['C_onset']:8.3f} {v['recall_overlap']:7.1%} {v['recall_onset']:9.1%} {v['fp_per_min']:9.2f}  {v['label']}")
    print(f"PICK (lowest C-overlap that also lowers C-onset below shipped): {pick}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(cell, log):
    R.use_set("heldout")
    cl = usable()
    fitres = log["fit"][cell]
    spec = cells(theta=fitres.get("theta"), theta_u=fitres.get("theta_u"))[cell]
    b_rows, b = run_cell(cl, {}); c_rows, c = run_cell(cl, spec)
    d_ov = [x["C_overlap"] - y["C_overlap"] for x, y in zip(c_rows, b_rows)]
    d_on = [x["C_onset"] - y["C_onset"] for x, y in zip(c_rows, b_rows)]
    strata = {}
    for s in ("complex", "random"):
        idx = [i for i, cc in enumerate(cl) if cc.get("stratum") == s]
        strata[s] = {"n": len(idx), "dC_overlap": boot([d_ov[i] for i in idx]) if idx else None,
                     "dfp": float(np.mean([c_rows[i]["fp"] - b_rows[i]["fp"] for i in idx])) if idx else None}
    m, lo, hi = boot(d_ov)
    log["heldout"] = {"cell": cell, "clips": len(cl), "shipped": b, "cell_summary": c, "dC_overlap": [m, lo, hi],
                      "dC_onset": boot(d_on), "strata": strata, "pass": hi < 0}
    print(f"[heldout] {len(cl)} clips; dC-overlap {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if hi < 0 else 'fail'}; strata {strata}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("screen", "fit", "heldout"))
    ap.add_argument("--cell", default=None)
    a = ap.parse_args()
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    R.use_set("calib")
    cl = usable()
    if a.step == "screen":
        screen(cl, log)
    elif a.step == "fit":
        fit(cl, log)
    else:
        assert a.cell and a.cell == log.get("pick"), "only the committed pick goes to held-out"
        heldout(a.cell, log)


if __name__ == "__main__":
    main()
