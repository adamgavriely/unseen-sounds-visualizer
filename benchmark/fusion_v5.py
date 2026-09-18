"""The last detector attempt (docs/prereg_detector_v5.md): PSED proposes at a loose bar; long
candidates are confirmed by a long-window model (BEATs 2-s windows or SSLAM 10-s windows);
short candidates are scored by PSED alone at its calibrated bar. All choices on the
AudioSet-Strong calibration set; slice B once.

    python -m benchmark.fusion_v5 --step0                 # AUROC of the long-window score on PSED's long boxes + unreachable misses
    python -m benchmark.fusion_v5 --select                # the three variants at 2.6 FP/min, split-half, -> benchmark/fusion_v5_setting.json
    python -m benchmark.fusion_v5 --sliceb                # one run of the frozen winner on slice B

Caches read: benchmark/audioset_calib_windows/{psed,beats,sslam}, benchmark/audioset_windows/{psed,beats,sslam}.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_detector_eval as E
from src.labels import is_salient_nonspeech, is_music, canonical, is_descendant

LOOSE = 0.05                       # PSED's loose proposal bar, fixed a priori (the lowest grid value)
WINDOW = {"beats": 2.0, "sslam": 10.0}   # long-window models: window length in seconds (times = window centres)
FP_TARGET = 2.6                    # selection rate on the calibration set (PSED's own bar)
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
L_GRID = [1.0, 2.0, 3.0, 4.0]
OUT = _ROOT / "benchmark" / "fusion_v5_setting.json"
MIN_DUR = 0.5


# ----------------------------------------------------------------------------- data
def load_set(which: str, models):
    """[(clip dict, {model: (fw, times, labels)})] for clips cached in every model"""
    E.use_set(which)
    out = []
    for c in E.clips():
        d = {}
        for m in models:
            p = E.WIN / m / f"{c['id']}.npz"
            if not p.exists():
                d = None; break
            d[m] = E._load(p)
        if d:
            out.append((c, d))
    return out


def psed_bar() -> float:
    f = _ROOT / "benchmark" / "detector_calib.json"
    return float(json.loads(f.read_text(encoding="utf-8"))["bars"]["psed"]) if f.exists() else 0.15


def boxes(fw, times, labels, bar):
    """PSED spans at `bar` (hysteresis bar/2, >= MIN_DUR) as (label, start, end, max score)"""
    from src.stage4_audio_event_detection import _extract_events
    return [(e.label, e.start, e.end, e.confidence) for e in _extract_events(fw, times, labels, bar, None, MIN_DUR, low=bar * 0.5)
            if is_salient_nonspeech(e.label) and not is_music(e.label)]


def long_score(fw, times, labels, label, a, b, window=2.0):
    """mean score of the same family (generic: whatever family the candidate claims) in the
    long-window model, over the windows whose span intersects [a, b]"""
    idx = [i for i, l in enumerate(labels) if l == label or canonical(l) == canonical(label) or is_descendant(l, label) or is_descendant(label, l)]
    if not idx:
        return 0.0
    m = (times + window / 2 >= a) & (times - window / 2 <= b)
    if not m.any():
        m = np.argmin(np.abs(times - (a + b) / 2)); return float(fw[m, idx].max())
    return float(fw[m][:, idx].max(axis=1).mean())


def gold_events(c):
    return [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"])]


def is_true(box, gold):
    """a candidate is true if it overlaps a same-family gold sound by >= 0.5 s (or half the
    sound) -- the same overlap rule as recall, so long boxes get no free pass"""
    lab, a, b, _ = box
    return any(E._same(lab, g["label"]) and E._overlap_ok(a, b, g["start"], g["end"]) for g in gold)


# ----------------------------------------------------------------------------- step 0
def step0(long_models=("beats", "sslam")):
    from sklearn.metrics import roc_auc_score
    avail = [m for m in long_models if (E.WIN / m).exists()] if E.use_set("calib") is None else long_models
    E.use_set("calib")
    avail = [m for m in long_models if (E.WIN / m).exists()]
    data = load_set("calib", ["psed"] + avail)
    rows = []       # (length, true?, {m: score})
    unreachable = 0; masked_total = 0; reachable = 0
    for c, d in data:
        fw, t, lab = d["psed"]
        bx = boxes(fw, t, lab, LOOSE)
        gold = gold_events(c)
        for box in bx:
            rows.append((box[2] - box[1], is_true(box, gold), {m: long_score(*d[m], box[0], box[1], box[2], WINDOW[m]) for m in avail}))
        for g in gold:
            if g["consequential"] and g["masked"]:
                masked_total += 1
                hit = any(E._same(bb[0], g["label"]) and E._overlap_ok(bb[1], bb[2], g["start"], g["end"]) for bb in bx)
                reachable += hit; unreachable += (not hit)
    print(f"[step0] {len(data)} calibration clips, {len(rows)} loose PSED boxes, {sum(r[1] for r in rows)} true")
    print(f"[step0] masked consequential sounds: {masked_total}; reachable by a loose box: {reachable} ({reachable / max(1, masked_total):.0%}); unreachable: {unreachable}")
    res = {"clips": len(data), "boxes": len(rows), "true_boxes": int(sum(r[1] for r in rows)), "masked_total": masked_total,
           "reachable": reachable, "unreachable": unreachable, "auroc": {}}
    for m in avail:
        for L in [0.0] + L_GRID:
            sub = [r for r in rows if r[0] >= L]
            y = [int(r[1]) for r in sub]; s = [r[2][m] for r in sub]
            auc = roc_auc_score(y, s) if 0 < sum(y) < len(y) else float("nan")
            res["auroc"][f"{m}@L>={L}"] = {"n": len(sub), "true": int(sum(y)), "auroc": auc}
            print(f"[step0] {m:6s} boxes >= {L:.0f} s: n={len(sub):5d} true={sum(y):4d}  AUROC {auc:.3f}")
    (_ROOT / "benchmark" / "fusion_v5_step0.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- variants
def run_variant(data, variant, long_model, theta, L, short_bar):
    """detections per clip under a variant; returns (masked recall, all recall, FP/min)"""
    hits_m = tot_m = hits_all = tot_all = fp = 0; minutes = 0.0
    for c, d in data:
        fw, t, lab = d["psed"]
        gold = gold_events(c)
        minutes += c["duration"] / 60.0
        dets = []
        if variant == "gate":                 # V1/V2: loose PSED, keep a box iff long score >= theta (short boxes: PSED bar)
            for box in boxes(fw, t, lab, LOOSE):
                L_ = box[2] - box[1]
                if L_ < L:
                    if box[3] >= short_bar: dets.append(box)
                elif long_score(*d[long_model], box[0], box[1], box[2], WINDOW[long_model]) >= theta:
                    dets.append(box)
        elif variant == "route":              # V3: box confidence = PSED max (short) or long-window mean (long); one threshold
            for box in boxes(fw, t, lab, LOOSE):
                L_ = box[2] - box[1]
                conf = box[3] if L_ < L else long_score(*d[long_model], box[0], box[1], box[2], WINDOW[long_model])
                bar = short_bar if L_ < L else theta
                if conf >= bar: dets.append(box)
        elif variant == "rescue":             # V1/V2 as intended: PSED's own detections at its bar are never lost;
            base = boxes(fw, t, lab, short_bar)                    # a long loose box is ADDED iff the long-window model confirms it
            dets = list(base)
            for box in boxes(fw, t, lab, LOOSE):
                if box[2] - box[1] < L or box[3] >= short_bar:
                    continue
                if any(b[0] == box[0] and min(b[2], box[2]) - max(b[1], box[1]) > 0 for b in base):
                    continue
                if long_score(*d[long_model], box[0], box[1], box[2], WINDOW[long_model]) >= theta:
                    dets.append(box)
        elif variant == "psed":               # control: PSED alone at `theta`
            dets = boxes(fw, t, lab, theta)
        for g in gold:
            m = any(E._same(x[0], g["label"]) and E._overlap_ok(x[1], x[2], g["start"], g["end"]) for x in dets)
            tot_all += 1; hits_all += m
            if g["consequential"] and g["masked"]:
                tot_m += 1; hits_m += m
        for x in dets:
            if not is_true(x, c["events"]):          # same overlap rule as recall
                fp += 1
    return hits_m / max(1, tot_m), hits_all / max(1, tot_all), fp / max(1e-6, minutes)


def best_at_rate(data, variant, long_model, L, short_bar):
    """the loosest theta whose FP/min <= FP_TARGET; returns (theta, masked, all, fp)"""
    best = None
    for th in GRID:
        r = run_variant(data, variant, long_model, th, L, short_bar)
        if r[2] <= FP_TARGET:
            best = (th,) + r; break
    return best or (GRID[-1],) + run_variant(data, variant, long_model, GRID[-1], L, short_bar)


def select():
    E.use_set("calib")
    longs = [m for m in ("beats", "sslam") if (E.WIN / m).exists()]
    data = load_set("calib", ["psed"] + longs)
    print(f"[select] {len(data)} calibration clips with every cache; masked consequential sounds: "
          f"{sum(1 for c, _ in data for g in gold_events(c) if g['consequential'] and g['masked'])}")
    rng = np.random.default_rng(0); idx = rng.permutation(len(data)); half = len(data) // 2
    A = [data[i] for i in idx[:half]]; B = [data[i] for i in idx[half:]]
    sb = psed_bar()
    variants = [("psed", None, 0.0, sb)] + [(v, m, L, sb) for v in ("gate", "route") for m in longs for L in L_GRID]         + [("rescue", m, L, bb) for m in longs for L in L_GRID for bb in (0.20, 0.25, 0.30, 0.35)]
    rows = []
    for v, m, L, bb in variants:
        th, rm, ra, fpr = best_at_rate(data, v, m or "beats", L, bb)
        thA, rmA, _, fA = best_at_rate(A, v, m or "beats", L, bb)
        rB = run_variant(B, v, m or "beats", thA, L, bb)          # A's setting read on B
        rows.append({"variant": v, "long": m, "L": L, "base_bar": bb, "theta": th, "masked": rm, "all": ra, "fp": fpr,
                     "half": {"theta_A": thA, "masked_A": rmA, "masked_B_at_A": rB[0], "all_B_at_A": rB[1], "fp_B_at_A": rB[2]}})
        print(f"[select] {v:6s} {str(m):6s} L={L:.0f} base={bb:.2f} theta={th:.2f}: masked {rm:.1%} all {ra:.1%} FP {fpr:.2f} | split A->B masked {rB[0]:.1%} all {rB[1]:.1%} FP {rB[2]:.2f}")
    control = rows[0]
    # a variant competes only if it reaches the false-alarm target on the full set AND on the
    # held-out half; it must beat the control on both; ties under 2 points go to the simpler rule
    order = {"psed": 0, "rescue": 1, "gate": 2, "route": 3}
    cands = [r for r in rows[1:] if r["fp"] <= FP_TARGET and r["half"]["fp_B_at_A"] <= FP_TARGET
             and r["half"]["masked_B_at_A"] > control["half"]["masked_B_at_A"] and r["masked"] > control["masked"]
             and r["all"] >= control["all"] - 0.01]          # must not lose recall on the hundreds of other events
    win = None
    if cands:
        top = max(cands, key=lambda r: r["masked"])
        near = [r for r in cands if top["masked"] - r["masked"] < 0.02]
        win = min(near, key=lambda r: (order[r["variant"]], r["L"]))
    unreachable = [f"{r['variant']}/{r['long']}/L{r['L']:.0f}/b{r['base_bar']:.2f}" for r in rows[1:] if r["fp"] > FP_TARGET or r["half"]["fp_B_at_A"] > FP_TARGET]
    if unreachable:
        print(f"[select] cannot reach {FP_TARGET}/min: {', '.join(unreachable)}")
    out = {"when": datetime.now().isoformat(timespec="minutes"), "fp_target": FP_TARGET, "loose": LOOSE, "short_bar": sb,
           "control": control, "rows": rows, "winner": win}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[select] control PSED: masked {control['masked']:.1%} at FP {control['fp']:.2f}")
    print(f"[select] winner: {win}" if win else "[select] no variant beats PSED on both the full set and the held-out half")
    print("->", OUT)


def sliceb():
    s = json.loads(OUT.read_text(encoding="utf-8"))
    win = s["winner"]
    E.use_set("sliceB")
    longs = [m for m in ("beats", "sslam") if (E.WIN / m).exists()]
    data = load_set("sliceB", ["psed"] + longs)
    print(f"[sliceB] {len(data)} clips with every cache; masked consequential sounds: "
          f"{sum(1 for c, _ in data for g in gold_events(c) if g['consequential'] and g['masked'])}")
    ctrl = run_variant(data, "psed", "beats", s["control"]["theta"], 0.0, s["short_bar"])
    print(f"[sliceB] PSED control @{s['control']['theta']:.2f}: masked {ctrl[0]:.1%} all {ctrl[1]:.1%} FP {ctrl[2]:.2f}")
    res = {"control": ctrl}
    if win:
        r = run_variant(data, win["variant"], win["long"] or "beats", win["theta"], win["L"], win.get("base_bar", s["short_bar"]))
        print(f"[sliceB] winner {win['variant']}/{win['long']} L={win['L']} theta={win['theta']:.2f}: masked {r[0]:.1%} all {r[1]:.1%} FP {r[2]:.2f}")
        res["winner"] = r; res["passed"] = bool(r[0] >= 0.70 and r[2] <= 2.6)
        print(f"[sliceB] pass rule (>= 70% at <= 2.6/min): {'PASSED' if res['passed'] else 'FAILED'}")
    s["sliceB"] = res
    OUT.write_text(json.dumps(s, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--sliceb", action="store_true")
    a = ap.parse_args()
    if a.step0: step0()
    if a.select: select()
    if a.sliceb: sliceb()


if __name__ == "__main__":
    main()
