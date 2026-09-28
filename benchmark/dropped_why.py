"""Why are "heard but dropped" sounds low? (2026-09-28, follow-up to benchmark/detector_audit.py). Descriptive, CPU only,
the 280 and the 415; nothing on DEV/TEST.

Groups of consequential gold events (shipped stack, detector_audit primary bins):
  hit   matched under C-overlap (the reference: is "masked" special to drops, or true of every event?)
  ii    missed, FlexSED 0.4-0.8 near it        iii  missed, BEATs 0.175-0.35 near it
  iv    missed, heard above the bar but removed by a veto
Band spans: FlexSED spans extracted at a bar of 0.4 (hysteresis 1.0) with peak in [0.4, 0.8), salient labels, run through
the shipped rules (twin rule, FlexSED clip veto 0.3, BEATs self-veto b 0.1218). Fate: shown (a bar of 0.4 would admit it)
| absorbed by a shown BEATs twin | absorbed by a weak or vetoed BEATs twin | clip veto | self-veto.
  band-true     overlaps (C-overlap rule) a consequential event the shipped stack misses
  band-phantom  overlaps no same-family gold event (a false span, clip_cost's rule)
  band-other    overlaps a same-family gold event that is not a missed consequential one

Features: BEATs Speech / Music max during the event or span (masked = >= 0.3) and gold "masked"; duration; loudness =
dB RMS over the event minus the clip's median 50-ms-frame dB (the whole mix, not the source alone); number of other gold
events overlapping it; family; FlexSED family peak within +-1 s, its time, inside the event or shifted; BEATs family max
within +-1 s.

Separation: A = dropped events (ii+iii+iv, gold times) vs band phantoms a bar of 0.4 would admit (the question as asked);
B = band spans on a missed event vs band phantoms, any fate (span level, so a rule could use it). Per feature: AUROC, and
the single threshold with the best Youden J on the 280, applied unchanged to the 415.

    python benchmark/dropped_why.py        # -> benchmark/dropped_why.json (+ tables on stdout)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from benchmark import detector_round4 as R4
from benchmark import detector_round5 as R5
from benchmark import detector_audit as A
from src.labels import canonical, is_salient_nonspeech, is_music, SPEECH_LABELS
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "dropped_why.json"
BAND_BAR = 0.4
SR, HOP = 16000, 800                       # 50-ms RMS frames
DROP = ("ii", "iii", "iv")
FATES = ["shown", "absorbed: shown BEATs twin", "absorbed: weak or vetoed BEATs twin", "BEATs self-veto", "FlexSED clip veto"]
key = lambda e: canonical(e.label)
sal = lambda l: is_salient_nonspeech(l) and not is_music(l)


def stack_bar(cc, fbar):
    """the shipped rules with a free FlexSED bar (hysteresis 1.0); returns [(FlexSED span, fate)]"""
    b, f, _p = cc
    events = _extract_events(b[0], b[1], b[2], D.AED, None, config.AED_MIN_DUR, low=D.AED * D.HYS)
    fev = _extract_events(f[0], f[1], f[2], fbar, None, config.AED_MIN_DUR, low=fbar * D.HYS)
    fresh, twins = [], {}
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
            twins[id(e)] = tw
        else:
            fresh.append(e)
    fids = {id(e) for e in fresh}
    fpk, bpk = D.clip_peak(f), D.clip_peak(b)
    ok = lambda e: fpk.get(key(e), 1.0) >= R4.FVETO and not (id(e) in fids and bpk.get(key(e), 1.0) < A.B)
    sid = {id(e) for e in events + fresh if ok(e) and e.confidence >= D.DISP}
    fate = []
    for e in fev:
        if id(e) in twins:
            fate.append((e, FATES[1] if any(id(x) in sid for x in twins[id(e)]) else FATES[2]))
        elif fpk.get(key(e), 1.0) < R4.FVETO:
            fate.append((e, FATES[4]))
        elif bpk.get(key(e), 1.0) < A.B:
            fate.append((e, FATES[3]))
        else:
            fate.append((e, FATES[0]))
    return fate


def audio_db(cid):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(R.E.VIDEOS / f"{cid}.mp4"), "-vn", "-ac", "1", "-ar",
                          str(SR), "-f", "s16le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0
    n = len(x) // HOP
    fr = x[:n * HOP].reshape(n, HOP)
    return x, float(np.median(10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)))


def loud(x, med_db, s, e):
    seg = x[int(max(0, s) * SR):int(max(s + 0.05, e) * SR)]
    return round(float(10 * np.log10((seg ** 2).mean() + 1e-12) - med_db), 2) if len(seg) else None


def flex_peak(fr, cols, s, e):
    fw, ts, _l = fr
    m = (ts >= s - A.NEAR) & (ts <= e + A.NEAR)
    if not cols or not m.any():
        return None, None, None
    v = fw[m][:, cols].max(axis=1)
    i = int(np.argmax(v)); t = float(ts[m][i])
    shift = 0.0 if s <= t <= e else (t - s if t < s else t - e)
    return round(float(v[i]), 3), round(t, 3), round(shift, 3)


def features(c, cc, x, med_db, BF, FF, label, s, e, gold_self=None):
    sp = A.peak(cc[0], [BF.ref.index("Speech")], s, e); mu = A.peak(cc[0], [BF.ref.index("Music")], s, e)
    fpv, fpt, fsh = flex_peak(cc[1], FF.fam(label), s, e)
    bf = A.peak(cc[0], BF.fam(label), s - A.NEAR, e + A.NEAR)
    others = [g for g in c["events"] if g is not gold_self and min(e, g["end"]) - max(s, g["start"]) > 0]
    return {"clip": c["id"], "label": label, "family": canonical(label), "start": round(s, 3), "end": round(e, 3),
            "dur": round(e - s, 3), "beats_speech": round(sp, 3), "beats_music": round(mu, 3),
            "masked_beats": bool(max(sp, mu) >= 0.3), "loud_rel_db": loud(x, med_db, s, e),
            "n_overlap_gold": len(others),
            "n_overlap_speech_music": sum(g["label"] in SPEECH_LABELS or is_music(g["label"]) for g in others),
            "flex_peak": fpv, "flex_peak_t": fpt, "flex_shift_s": fsh, "flex_inside": None if fsh is None else fsh == 0.0,
            "beats_family": None if bf is None else round(bf, 3), "stratum": c.get("stratum")}


def run_set(set_name, ref):
    cl, S, fps, misses, timing = A.audit(set_name, ref)
    mbin = {(m["clip"], m["label"], m["start"], m["end"]): m["bin"] for m in misses}
    BF, FF = A.Frames("BEATs"), A.Frames("FlexSED")
    events, band = [], []
    for c in cl:
        cc = R4.load3(c["id"])
        BF.check(cc[0][2]); FF.check(cc[1][2])
        x, med_db = audio_db(c["id"])
        shown, _i, _d = A.tagged(cc)
        ev = D._ev(shown)
        fate = [(e, r) for e, r in stack_bar(cc, BAND_BAR) if e.confidence < R4.FBAR and sal(e.label)]   # the band only
        missed_here = []
        for g in c["events"]:
            if not (sal(g["label"]) and g["consequential"]):
                continue
            hit = any(A.same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev)
            grp = "hit" if hit else mbin[(c["id"], g["label"], g["start"], g["end"])]
            if not hit:
                missed_here.append(g)
            f = features(c, cc, x, med_db, BF, FF, g["label"], g["start"], g["end"], gold_self=g)
            fm = [r for e, r in fate if A.same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"])]
            f.update({"group": grp, "masked_gold": g["masked"],
                      "band_fate": next((o for o in FATES if o in fm), "no 0.4-0.8 FlexSED span covers it")})
            events.append(f)
        for e, r in fate:
            true = any(A.same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for g in missed_here)
            false = not any(A.same(e.label, g["label"]) and A.ov(e, g) > 0 for g in c["events"])
            f = features(c, cc, x, med_db, BF, FF, e.label, e.start, e.end)
            f.update({"group": "band-true" if true else ("band-phantom" if false else "band-other"), "fate": r,
                      "flex_peak": round(e.confidence, 3), "flex_inside": True, "flex_shift_s": 0.0})
            band.append(f)
    return {"clips": len(cl), "events": events, "band": band}


# ----------------------------------------------------------------------------- summaries
NUM = ["dur", "loud_rel_db", "n_overlap_gold", "beats_speech", "beats_music", "flex_peak", "beats_family"]
SEP = ["beats_family", "loud_rel_db", "dur", "flex_peak", "beats_speech", "beats_music", "n_overlap_gold"]


def quart(xs):
    xs = [v for v in xs if v is not None]
    return None if not xs else [round(float(np.percentile(xs, q)), 3) for q in (25, 50, 75)]


def describe(rows):
    n = len(rows)
    d = {"n": n, "clips": len({r["clip"] for r in rows}),
         "masked_beats": [sum(r["masked_beats"] for r in rows), A.pct(sum(r["masked_beats"] for r in rows), n)]}
    if rows and "masked_gold" in rows[0]:
        d["masked_gold"] = [sum(r["masked_gold"] for r in rows), A.pct(sum(r["masked_gold"] for r in rows), n)]
    for k in NUM:
        d[k] = quart([r[k] for r in rows])
    ins = [r for r in rows if r["flex_inside"] is not None]
    d["flex_peak_inside"] = [sum(r["flex_inside"] for r in ins), A.pct(sum(r["flex_inside"] for r in ins), len(ins))]
    d["flex_shift_when_outside"] = quart([r["flex_shift_s"] for r in ins if not r["flex_inside"]])
    d["top_families"] = [[k, v] for k, v in Counter(r["family"] for r in rows).most_common(10)]
    return d


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def val(r, k):
    v = r[k]
    return -1.0 if v is None else float(v)


def separation(tp, tn, hp, hn):
    """true vs phantom rows: per feature AUROC on each set; threshold (keep >= t or keep <= t) with the best Youden J
    on the 280, applied unchanged to the 415"""
    out = {"n": {"280": [len(tp), len(tn)], "415": [len(hp), len(hn)]}}
    feats = {}
    for k in SEP:
        a = auroc([val(r, k) for r in tp], [val(r, k) for r in tn])
        if a is None:
            continue
        sign = 1 if a >= 0.5 else -1                       # keep the side where true rows sit
        best = None
        for t in sorted({val(r, k) for r in tp + tn}):
            keep = lambda r: sign * val(r, k) >= sign * t
            kt, kn = sum(map(keep, tp)), sum(map(keep, tn))
            j = kt / len(tp) - kn / len(tn)
            if best is None or j > best[0]:
                best = (j, t, kt, kn)
        j, t, kt, kn = best
        keep = lambda r: sign * val(r, k) >= sign * t
        a2 = auroc([val(r, k) for r in hp], [val(r, k) for r in hn])
        feats[k] = {"auroc_280": round(a, 3), "auroc_415": None if a2 is None else round(a2, 3),
                    "rule": f"keep {k} {'>=' if sign > 0 else '<='} {t:g}", "youden_280": round(j, 3),
                    "280": {"true_kept": f"{kt}/{len(tp)}", "phantom_kept": f"{kn}/{len(tn)}"},
                    "415": {"true_kept": f"{sum(map(keep, hp))}/{len(hp)}", "phantom_kept": f"{sum(map(keep, hn))}/{len(hn)}"}}
    out["features"] = feats
    out["top3"] = sorted(feats, key=lambda k: -abs(feats[k]["auroc_280"] - 0.5))[:3]
    return out


def main():
    r5 = json.loads(R5.OUT.read_text(encoding="utf-8"))
    res = {"note": "descriptive; CPU only; nothing on DEV/TEST", "definitions": __doc__}
    raw = {}
    for s in ("calib", "heldout"):
        raw[s] = run_set(s, r5[f"gate1_{s}"]["baseline_shipped_b"])
        E, Bd = raw[s]["events"], raw[s]["band"]
        groups = {g: describe([r for r in E if r["group"] == g]) for g in ("hit",) + DROP}
        groups["dropped (ii+iii+iv)"] = describe([r for r in E if r["group"] in DROP])
        groups["all misses"] = describe([r for r in E if r["group"] != "hit"])
        adm = [r for r in Bd if r["fate"] == FATES[0]]
        groups["band phantom, admitted at 0.4"] = describe([r for r in adm if r["group"] == "band-phantom"])
        groups["band true, admitted at 0.4"] = describe([r for r in adm if r["group"] == "band-true"])
        groups["band phantom, any fate"] = describe([r for r in Bd if r["group"] == "band-phantom"])
        groups["band true, any fate"] = describe([r for r in Bd if r["group"] == "band-true"])
        res[s] = {"clips": raw[s]["clips"], "groups": groups,
                  "dropped_event_fate_at_0.4": {g: dict(Counter(r["band_fate"] for r in E if r["group"] == g)) for g in DROP},
                  "band_span_fates": {g: dict(Counter(r["fate"] for r in Bd if r["group"] == g))
                                      for g in ("band-true", "band-phantom", "band-other")},
                  "events": E, "band": Bd}
        print(f"[{s}] events {len(E)}, band spans {len(Bd)}", flush=True)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    sel = lambda s, f: [r for r in raw[s]["events"] + raw[s]["band"] if f(r)]
    isdrop = lambda r: r["group"] in DROP
    isadm = lambda r: r["group"] == "band-phantom" and r.get("fate") == FATES[0]
    ist = lambda r: r["group"] == "band-true"
    isp = lambda r: r["group"] == "band-phantom"
    res["separation_A"] = separation(sel("calib", isdrop), sel("calib", isadm), sel("heldout", isdrop), sel("heldout", isadm))
    res["separation_B"] = separation(sel("calib", ist), sel("calib", isp), sel("heldout", ist), sel("heldout", isp))
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for s in ("calib", "heldout"):
        print(f"\n===== {s} =====")
        for g, d in res[s]["groups"].items():
            print(g, json.dumps(d))
        print("dropped fate", json.dumps(res[s]["dropped_event_fate_at_0.4"]))
        print("band fates", json.dumps(res[s]["band_span_fates"]))
    for name in ("separation_A", "separation_B"):
        print(f"\n===== {name} n {res[name]['n']} =====")
        for k, v in res[name]["features"].items():
            print(k, json.dumps(v))
        print("top3", res[name]["top3"])


if __name__ == "__main__":
    main()
