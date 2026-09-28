"""Detector round 4 (docs/prereg_v4.md, 2026-09-28): two fixed questions from the saved caches (no model run).

  Test 1  end-trim: a BEATs-origin shown span ends at the last FlexSED frame inside it whose same-family score is
          >= 0.5 x FlexSED's in-span peak (never extend, cut <= 1.5 s, never before start + 0.5 s).
  Test 2  BEATs self-veto: a FlexSED-only span is kept iff BEATs' clip-max for its family >= b, in place of the PANNs
          veto (PANNs clip-max >= 0.05); b fixed on the 280 so the kept share equals PANNs'.

    python benchmark/detector_round4.py fit       # the 280 (fit set): sanity gate, Test 1 rule, b, Test 2 numbers
    python benchmark/detector_round4.py heldout   # the 415: Test 1 only if a candidate on the 280, Test 2 always

Cost C, clip sets and bootstrap as amendment 24 (benchmark/detector_round2.py). No display floor (display-level).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from src.labels import canonical, is_salient_nonspeech, is_music
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "detector_round4.json"
FBAR, FVETO, PTAU = 0.8, 0.3, 0.05
TRIM_FRAC, TRIM_MAX, TRIM_MIN_LEN = 0.5, 1.5, 0.5
EXPECT = {"calib": {"C_overlap": 3.071, "C_onset": 3.886, "recall_overlap": 0.504, "fp_per_min": 4.46},
          "heldout": {"C_overlap": 1.923, "recall_overlap": 0.491, "fp_per_min": 3.25}}
key = lambda e: canonical(e.label)


# ----------------------------------------------------------------------------- the tagged stack
def load3(cid):
    return (R.load(R.E.WIN / "beats" / f"{cid}.npz"), R.load(R.FLEX / f"{cid}.npz"), R.load(R.PANNS / f"{cid}.npz"))


def stack_tagged(cid, veto=("panns", PTAU), caches=None):
    """detector_round2.stack() with default arguments, plus origin tags and the veto pool.
    Returns (shown spans, set of ids of FlexSED-only spans, pool = [(span, panns clip-max, beats clip-max)])."""
    b, f, p = caches or load3(cid)
    events = _extract_events(b[0], b[1], b[2], D.AED, None, config.AED_MIN_DUR, low=D.AED * D.HYS)
    fev = _extract_events(f[0], f[1], f[2], FBAR, None, config.AED_MIN_DUR, low=FBAR * D.HYS)
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
    events = [e for e in events if fpk.get(key(e), 1.0) >= FVETO]                     # FlexSED veto (clip level)
    ppk, bpk = D.clip_peak(p), D.clip_peak(b)
    pool = [(e, ppk.get(key(e), 1.0), bpk.get(key(e), 1.0)) for e in events if id(e) in flex_ids]
    if veto[0] == "panns":
        drop = {id(e) for e, pv, _bv in pool if pv < veto[1]}
    else:
        drop = {id(e) for e, _pv, bv in pool if bv < veto[1]}
    events = [e for e in events if id(e) not in drop]
    out = [e for e in events if e.confidence >= D.DISP]
    return out, flex_ids, pool


def same_spans(a, b):
    sig = lambda ev: sorted((e.label, round(e.start, 6), round(e.end, 6), round(e.confidence, 6)) for e in ev)
    return sig(a) == sig(b)


# ----------------------------------------------------------------------------- Test 1: the trim
def trim(ev, flex_ids, f, log):
    fw, ts, labs = f
    dt = 1.0 / 25.0
    out = []
    for e in ev:
        if id(e) in flex_ids:
            out.append(e); continue
        cols = [i for i, l in enumerate(labs) if canonical(l) == key(e)]
        log["beats_spans"] += 1
        m = (ts >= e.start) & (ts < e.end)
        if not cols or not m.any() or e.start + TRIM_MIN_LEN >= e.end:
            out.append(e); continue
        s = fw[m][:, cols].max(axis=1)
        P = float(s.max())
        if P <= 0:
            out.append(e); continue
        t_last = float(ts[m][np.where(s >= TRIM_FRAC * P)[0][-1]])
        new_end = min(e.end, max(t_last + dt, e.end - TRIM_MAX, e.start + TRIM_MIN_LEN))
        if new_end < e.end - 1e-9:
            log["trimmed"] += 1; log["cut_s"].append(e.end - new_end); log["low_peak"] += P < 0.3
        out.append(dataclasses.replace(e, end=new_end))
    return out


def end_errors(c, ev):
    """{gold index: |end error|} for consequential salient gold events matched by a span (largest overlap, tie -> earlier)"""
    ev = D._ev(ev)
    res = {}
    for gi, g in enumerate(c["events"]):
        if not (is_salient_nonspeech(g["label"]) and not is_music(g["label"]) and g["consequential"]):
            continue
        m = [e for e in ev if R.E._same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"])]
        if m:
            best = min(m, key=lambda e: (-(min(e.end, g["end"]) - max(e.start, g["start"])), e.start))
            res[gi] = abs(best.end - g["end"])
    return res


# ----------------------------------------------------------------------------- summaries
def summary(cl, rows):
    minutes = sum(c["duration"] for c in cl) / 60.0
    n = max(1, sum(r["n_conseq"] for r in rows))
    return {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
            "fp": int(sum(r["fp"] for r in rows)), "fp_per_min": sum(r["fp"] for r in rows) / minutes,
            "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / n,
            "recall_onset": 1 - sum(r["miss_onset"] for r in rows) / n, "n_conseq": int(n)}


def med_delta(pairs_per_clip, idx=None):
    pp = [x for i in (range(len(pairs_per_clip)) if idx is None else idx) for x in pairs_per_clip[i]]
    if not pp:
        return float("nan")
    a = np.array(pp)
    return float(np.median(a[:, 0]) - np.median(a[:, 1]))


def boot_med(pairs_per_clip, n=2000, seed=0):
    rng = np.random.default_rng(seed); N = len(pairs_per_clip)
    m = np.array([med_delta(pairs_per_clip, rng.integers(0, N, N)) for _ in range(n)])
    return med_delta(pairs_per_clip), float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))


def check_shipped(set_name, cl, rows, S):
    exp = EXPECT[set_name]
    tol = {"C_overlap": 0.0005, "C_onset": 0.0005, "recall_overlap": 0.0005, "fp_per_min": 0.005}
    ok = all(abs(S[k] - v) <= tol[k] for k, v in exp.items())
    print(f"[sanity {set_name}] {len(cl)} clips; shipped C-overlap {S['C_overlap']:.3f} C-onset {S['C_onset']:.3f} "
          f"recall {S['recall_overlap']:.1%} false/min {S['fp_per_min']:.2f} -> expected {exp}: {'OK' if ok else 'MISMATCH'}")
    return ok


# ----------------------------------------------------------------------------- one set
def base(set_name):
    R.use_set(set_name)
    cl = D.usable()
    per = []
    for c in cl:
        cc = load3(c["id"])
        ev, fids, pool = stack_tagged(c["id"], caches=cc)
        assert same_spans(ev, D.stack(c["id"])), f"tagged stack differs from detector_round2.stack on {c['id']}"
        per.append((c, cc, ev, fids, pool))
    rows = [D.clip_cost(c, ev) for c, _cc, ev, _f, _p in per]
    S = summary(cl, rows)
    return cl, per, rows, S


def test1(cl, per, rows_ship, S_ship):
    log = {"beats_spans": 0, "trimmed": 0, "low_peak": 0, "cut_s": []}
    rows_t, pairs, n_ship, n_trim, e_ship, e_trim = [], [], 0, 0, [], []
    for c, cc, ev, fids, _pool in per:
        tv = trim(ev, fids, cc[1], log)
        rows_t.append(D.clip_cost(c, tv))
        a, b = end_errors(c, ev), end_errors(c, tv)
        n_ship += len(a); n_trim += len(b); e_ship += list(a.values()); e_trim += list(b.values())
        pairs.append([(a[k], b[k]) for k in a if k in b])
    S_t = summary(cl, rows_t)
    d, lo, hi = boot_med(pairs)
    dC = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rows_t, rows_ship)])
    dCon = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows_t, rows_ship)])
    npair = sum(len(p) for p in pairs)
    allp = np.array([x for p in pairs for x in p]) if npair else np.zeros((0, 2))
    res = {"shipped": S_ship, "trimmed": S_t,
           "end_err_paired": {"n": npair, "median_shipped": float(np.median(allp[:, 0])) if npair else None,
                              "median_trimmed": float(np.median(allp[:, 1])) if npair else None, "delta": [d, lo, hi]},
           "end_err_per_arm": {"shipped": {"n": n_ship, "median": float(np.median(e_ship)) if e_ship else None},
                               "trimmed": {"n": n_trim, "median": float(np.median(e_trim)) if e_trim else None}},
           "dC_overlap": dC, "dC_onset": dCon,
           "trim_log": {"beats_origin_shown_spans": log["beats_spans"], "trimmed": log["trimmed"],
                        "trimmed_with_peak_below_0.3": int(log["low_peak"]),
                        "median_cut_s": float(np.median(log["cut_s"])) if log["cut_s"] else None,
                        "mean_cut_s": float(np.mean(log["cut_s"])) if log["cut_s"] else None}}
    print(f"[test1] end error (paired, n={npair}): shipped {res['end_err_paired']['median_shipped']:.3f} s -> trimmed "
          f"{res['end_err_paired']['median_trimmed']:.3f} s; delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; per arm n {n_ship}/{n_trim}")
    print(f"[test1] recall {S_ship['recall_overlap']:.1%} -> {S_t['recall_overlap']:.1%}; onset-recall {S_ship['recall_onset']:.1%} -> "
          f"{S_t['recall_onset']:.1%}; false spans {S_ship['fp']} -> {S_t['fp']} ({S_ship['fp_per_min']:.2f} -> {S_t['fp_per_min']:.2f}/min); "
          f"C-overlap {S_ship['C_overlap']:.3f} -> {S_t['C_overlap']:.3f} dC {dC}; C-onset {S_ship['C_onset']:.3f} -> {S_t['C_onset']:.3f}")
    print(f"[test1] trims {res['trim_log']}")
    return res


def pool_arrays(per):
    pv = np.array([x[1] for *_a, pool in per for x in pool], float)
    bv = np.array([x[2] for *_a, pool in per for x in pool], float)
    return pv, bv


def test2(cl, per, rows_ship, S_ship, b):
    rows_s = []
    for c, cc, _ev, _f, _p in per:
        ev, _fids, _pool = stack_tagged(c["id"], veto=("beats", b), caches=cc)
        rows_s.append(D.clip_cost(c, ev))
    S_s = summary(cl, rows_s)
    pv, bv = pool_arrays(per)
    kp, kb = pv >= PTAU, bv >= b
    agree = {"both_keep": int((kp & kb).sum()), "both_drop": int((~kp & ~kb).sum()), "panns_only_keep": int((kp & ~kb).sum()),
             "beats_only_keep": int((~kp & kb).sum()), "rate": float((kp == kb).mean()) if len(kp) else None,
             "pool": int(len(kp)), "kept_share_panns": float(kp.mean()) if len(kp) else None,
             "kept_share_beats": float(kb.mean()) if len(kb) else None}
    dC = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rows_s, rows_ship)])
    dCon = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows_s, rows_ship)])
    res = {"b": b, "shipped_panns": S_ship, "self_veto": S_s, "dC_overlap": dC, "dC_onset": dCon, "agreement": agree}
    print(f"[test2] b={b:.4f}; C-overlap {S_ship['C_overlap']:.3f} -> {S_s['C_overlap']:.3f} dC {dC}; C-onset {S_ship['C_onset']:.3f} -> "
          f"{S_s['C_onset']:.3f}; recall {S_ship['recall_overlap']:.1%} -> {S_s['recall_overlap']:.1%}; onset-recall "
          f"{S_ship['recall_onset']:.1%} -> {S_s['recall_onset']:.1%}; false/min {S_ship['fp_per_min']:.2f} -> {S_s['fp_per_min']:.2f}")
    print(f"[test2] agreement {agree}")
    return res


# ----------------------------------------------------------------------------- steps
def fit(log):
    cl, per, rows, S = base("calib")
    ok = check_shipped("calib", cl, rows, S)
    log["calib_sanity"] = {"clips": len(cl), "shipped": S, "ok": ok}
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if not ok:
        sys.exit("sanity gate failed: stop")
    t1 = test1(cl, per, rows, S)
    d = t1["end_err_paired"]["delta"][0]
    t1["candidate"] = bool(d >= 0.3 and t1["trimmed"]["recall_overlap"] >= S["recall_overlap"] - 0.005 and t1["trimmed"]["fp"] <= S["fp"])
    print(f"[test1] 280 adopt-candidate: {t1['candidate']}")
    log["test1_calib"] = t1
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    pv, bv = pool_arrays(per)
    k = int((pv >= PTAU).sum())
    b = float(np.sort(bv)[::-1][k - 1]) if k > 0 else float("inf")
    log["test2_b"] = {"pool_calib": int(len(pv)), "panns_kept": k, "panns_kept_share": round(k / max(1, len(pv)), 4), "b": b,
                      "beats_kept_share": round(float((bv >= b).mean()), 4)}
    print(f"[test2] 280 pool {len(pv)} FlexSED-only spans; PANNs keeps {k}; b = {b:.4f}; BEATs kept share {log['test2_b']['beats_kept_share']:.4f}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    log["test2_calib"] = test2(cl, per, rows, S, b)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(log):
    assert "test2_b" in log, "run fit first"
    cl, per, rows, S = base("heldout")
    ok = check_shipped("heldout", cl, rows, S)
    log["heldout_sanity"] = {"clips": len(cl), "shipped": S, "ok": ok}
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if not ok:
        sys.exit("sanity gate failed: stop")
    if log["test1_calib"]["candidate"]:
        t1 = test1(cl, per, rows, S)
        d, lo, _hi = t1["end_err_paired"]["delta"]
        t1["adopt"] = bool(d >= 0.3 and lo > 0 and t1["dC_overlap"][2] <= 0)
        print(f"[test1] 415 adopt: {t1['adopt']}")
        log["test1_heldout"] = t1
    else:
        log["test1_heldout"] = "not scored: no adopt-candidate on the 280 (as pre-registered)"
        print("[test1] not a candidate on the 280 -> the 415 is not scored for Test 1")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    pv, _bv = pool_arrays(per)
    print(f"[test2] 415 pool {len(pv)} FlexSED-only spans")
    t2 = test2(cl, per, rows, S, log["test2_b"]["b"])
    t2["drop_panns"] = bool(t2["dC_overlap"][2] < 0.1 and t2["self_veto"]["recall_overlap"] >= S["recall_overlap"] - 0.005)
    print(f"[test2] 415 drop PANNs: {t2['drop_panns']}")
    log["test2_heldout"] = t2
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fit", "heldout"))
    a = ap.parse_args()
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if a.step == "fit":
        log = {}
        fit(log)
    else:
        heldout(log)


if __name__ == "__main__":
    main()
