"""Detector audit (2026-09-28): where do the SHIPPED stage-4 stack's false spans and misses come from?
Descriptive only, no selection, no model run: the saved BEATs / FlexSED caches on the AudioSet-Strong fit set (the 280,
R.use_set("calib")) and the held-out set (the 415, "heldout"). Nothing on DEV/TEST.

Shipped stack = detector_round5.run(slot 0 = BEATs, AED 0.175, display 0.35, self-veto b = 0.1218): BEATs spans + FlexSED
0.8 spans (twin rule keeps the earlier start) + FlexSED clip veto 0.3 + BEATs self-veto on FlexSED-only spans (PANNs off).
The script rebuilds it with origin tags, asserts span-for-span equality with detector_round5 and the gate-1 numbers in
detector_round5.json, then bins:

  1. false spans (clip_cost's fp):  a  overlaps a gold event of a RELATED family (sibling/cousin: deepest common ancestor
                                        in the AudioSet ontology is not one of the 7 roots; parent/child already match)
                                     b  same family as a gold event elsewhere in the clip, no overlap (timing / extent)
                                     c  no related gold event (phantom): origin, BEATs Speech/Music >= 0.3 in the span
     precedence a > b > c (the a-and-b count is reported).
  2. missed consequential events (miss_overlap), one primary bin, precedence
         iv (heard above the bar, removed by a veto) > v (a shown span of a related family overlaps it) >
         t (timing: a same-family span is shown but overlaps too little or sits within 1 s) >
         ii (FlexSED 0.4-0.8 near it) > iii (BEATs 0.175-0.35 near it) > i (no score >= 0.2 near it) > vi (other);
     every flag is also counted on its own (non-exclusive). "Near" = [start - 1 s, end + 1 s]; "family" = labels with
     E._same(label, gold label). Under speech/music = gold "masked" (>= half under gold Speech/Music), plus BEATs >= 0.3.
  3. timing: events hit under C-overlap but missed under C-onset: signed onset error of the max-overlap span
     (span start - gold start) and the step that set the onset (BEATs / FlexSED twin rule / FlexSED only; merge flag =
     the span starts on or before an earlier same-family gold event of the clip).

    python benchmark/detector_audit.py            # both sets -> benchmark/detector_audit.json (+ markdown tables on stdout)
"""
from __future__ import annotations

import json
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
from src.labels import canonical, ancestors, _common_parent, is_salient_nonspeech, is_music
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "detector_audit.json"
B = R5.B_SHIPPED
ROOTS = {"Human sounds", "Animal", "Music", "Sounds of things", "Natural sounds", "Source-ambiguous sounds",
         "Channel, environment and background"}
NEAR, HEARD = 1.0, 0.2
key = lambda e: canonical(e.label)
same = lambda a, b: R.E._same(a, b)
ov = lambda e, g: min(e.end, g["end"]) - max(e.start, g["start"])


# ----------------------------------------------------------------------------- ontology relation
_REL = {}


def related(a, g):
    """deepest non-root common ancestor of two labels that do NOT already match (E._same), else None"""
    k = (a, g)
    if k not in _REL:
        best, bd = None, -1
        if not same(a, g):
            for x, y in ((a, g), (canonical(a), canonical(g)), (a, canonical(g)), (canonical(a), g)):
                cp = _common_parent(x, y)
                if cp and cp not in ROOTS and len(ancestors(cp)) > bd:
                    best, bd = cp, len(ancestors(cp))
        _REL[k] = best
    return _REL[k]


# ----------------------------------------------------------------------------- the tagged shipped stack
def tagged(cc):
    """round 5's pre() + finish() at the shipped bars, with origin tags and every discard pool.
    Returns (shown, info{id: origin/start0/twins}, dropped [(span, reason)])."""
    b, f, _p = cc
    events = _extract_events(b[0], b[1], b[2], D.AED, None, config.AED_MIN_DUR, low=D.AED * D.HYS)
    info = {id(e): {"origin": "beats", "start0": e.start, "twins": []} for e in events}
    fev = _extract_events(f[0], f[1], f[2], R4.FBAR, None, config.AED_MIN_DUR, low=R4.FBAR * D.HYS)
    fresh = []
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start); info[id(x)]["twins"].append(e)
        else:
            fresh.append(e); info[id(e)] = {"origin": "flex", "start0": e.start, "twins": []}
    fids = {id(e) for e in fresh}
    fpk, bpk = D.clip_peak(f), D.clip_peak(b)
    shown, dropped = [], []
    for e in events + fresh:
        if fpk.get(key(e), 1.0) < R4.FVETO:
            dropped.append((e, "FlexSED clip veto 0.3"))
        elif id(e) in fids and bpk.get(key(e), 1.0) < B:
            dropped.append((e, "BEATs self-veto (b 0.1218)"))
        elif e.confidence < D.DISP:
            dropped.append((e, "weak BEATs twin absorbed a FlexSED >= 0.8 span" if info[id(e)]["twins"] else "BEATs span < 0.35"))
        else:
            shown.append(e)
    return shown, info, dropped


# ----------------------------------------------------------------------------- frame helpers
class Frames:
    """family columns of one detector's label list (cached per distinct list; `ref` = the current clip's list)"""
    def __init__(self, name):
        self.name, self.ref, self.cols, self.lists = name, None, {}, set()

    def check(self, labs):
        self.ref = tuple(labs); self.lists.add(self.ref)

    def fam(self, glabel):
        k = (self.ref, glabel)
        if k not in self.cols:
            self.cols[k] = [i for i, l in enumerate(self.ref) if same(l, glabel)]
        return self.cols[k]


def peak(fr, cols, t0, t1):
    """max score of `cols` over frames with t in [t0, t1]; None if the family is not in the detector's list;
    the nearest frame to the middle when no frame falls inside (short spans vs 0.25-s BEATs stamps)"""
    fw, ts, _l = fr
    if not cols:
        return None
    m = (ts >= t0) & (ts <= t1)
    if not m.any():
        m = np.zeros(len(ts), bool); m[int(np.argmin(np.abs(ts - 0.5 * (t0 + t1))))] = True
    return float(fw[m][:, cols].max())


def bucket(x, edges, names):
    if x is None:
        return "not asked"
    for e, n in zip(edges, names):
        if x < e:
            return n
    return names[-1]


def pct(n, d):
    return round(100.0 * n / d, 1) if d else 0.0


def table(counter, total, top=None):
    items = counter.most_common(top) if top else sorted(counter.items(), key=lambda kv: -kv[1])
    return [{"k": k, "n": n, "pct": pct(n, total)} for k, n in items]


def qstats(x):
    if not len(x):
        return None
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": round(float(np.median(x)), 3), "q1": round(float(np.percentile(x, 25)), 3),
            "q3": round(float(np.percentile(x, 75)), 3), "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}


# ----------------------------------------------------------------------------- one set
def audit(set_name, ref):
    R.use_set(set_name)
    cl = D.usable()
    BF, FF = Frames("BEATs"), Frames("FlexSED")
    rows, fps, misses, timing = [], [], [], []
    for c in cl:
        cc = R4.load3(c["id"])
        BF.check(cc[0][2]); FF.check(cc[1][2])
        shown, info, dropped = tagged(cc)
        assert R4.same_spans(shown, R5.finish(R5.pre(cc[0], cc[1], D.AED), B, D.DISP)), f"tagged != round5 on {c['id']}"
        rows.append(D.clip_cost(c, shown))
        ev = D._ev(shown)
        gold_all = c["events"]
        sp_cols = [BF.ref.index("Speech")], [BF.ref.index("Music")]
        # ---------------------------------------------------------------- 1. false spans
        for e in ev:
            if any(same(e.label, g["label"]) and ov(e, g) > 0 for g in gold_all):
                continue
            rel = [(g, related(e.label, g["label"])) for g in gold_all if ov(e, g) > 0]
            rel = [(g, r) for g, r in rel if r]
            elsewhere = [g for g in gold_all if same(e.label, g["label"])]
            gap = min((max(g["start"] - e.end, e.start - g["end"]) for g in elsewhere), default=None)
            inf = info[id(e)]
            origin = "FlexSED only" if inf["origin"] == "flex" else ("both" if inf["twins"] else "BEATs only")
            sp = peak(cc[0], sp_cols[0], e.start, e.end); mu = peak(cc[0], sp_cols[1], e.start, e.end)
            bin_ = "a" if rel else ("b" if elsewhere else "c")
            fps.append({"clip": c["id"], "label": e.label, "family": key(e), "start": round(e.start, 3), "end": round(e.end, 3),
                        "conf": round(e.confidence, 3), "bin": bin_, "a_and_b": bool(rel and elsewhere), "origin": origin,
                        "related_gold": [{"label": g["label"], "via": r} for g, r in rel],
                        "gap_to_same_family_s": None if gap is None else round(float(gap), 3),
                        "beats_speech": round(sp, 3), "beats_music": round(mu, 3),
                        "flex_in_span": None if inf["origin"] == "flex" else peak(cc[1], FF.fam(e.label), e.start, e.end),
                        "beats_clipmax": float(D.clip_peak(cc[0]).get(key(e), 0.0)) if inf["origin"] == "flex" else None,
                        "gold_under": sorted({g["label"] for g in gold_all if ov(e, g) > 0}),
                        "stratum": c.get("stratum")})
        # ---------------------------------------------------------------- 2. misses and 3. timing
        dpool = [(e, r) for e, r in dropped if is_salient_nonspeech(e.label) and not is_music(e.label)]
        for g in gold_all:
            if not (is_salient_nonspeech(g["label"]) and not is_music(g["label"]) and g["consequential"]):
                continue
            hits = [e for e in ev if same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"])]
            on_hit = any(same(e.label, g["label"]) and g["start"] - R.EARLY <= e.start <= g["start"] + R.LATE for e in ev)
            speech_b = peak(cc[0], sp_cols[0], g["start"], g["end"]); music_b = peak(cc[0], sp_cols[1], g["start"], g["end"])
            if hits:
                if on_hit:
                    continue
                best = min(hits, key=lambda e: (-ov(e, g), e.start))
                inf = info[id(best)]
                src = "FlexSED only" if inf["origin"] == "flex" else (
                    "FlexSED twin rule" if best.start < inf["start0"] - 1e-9 else "BEATs")
                merged = any(g2 is not g and same(g2["label"], g["label"]) and g2["start"] < g["start"] - 1e-9
                             and best.start <= g2["end"] for g2 in gold_all)
                timing.append({"clip": c["id"], "label": g["label"], "g_start": g["start"], "g_end": g["end"],
                               "span": [round(best.start, 3), round(best.end, 3)], "span_label": best.label,
                               "err": round(best.start - g["start"], 3), "source": src, "merge": bool(merged),
                               "beats_start0": round(inf["start0"], 3) if inf["origin"] == "beats" else None,
                               "masked": g["masked"]})
                continue
            fb = peak(cc[0], BF.fam(g["label"]), g["start"] - NEAR, g["end"] + NEAR)
            ff = peak(cc[1], FF.fam(g["label"]), g["start"] - NEAR, g["end"] + NEAR)
            vb = fb or 0.0; vf = ff or 0.0
            vsp = [(e, r) for e, r in dpool if r != "BEATs span < 0.35" and (
                   (same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]))
                   or (r.startswith("weak BEATs twin") and any(same(t.label, g["label"]) and
                       R.E._overlap_ok(t.start, t.end, g["start"], g["end"]) for t in info[id(e)]["twins"])))]
            vsp += [(e, "twin rule: FlexSED span absorbed into a shown BEATs twin that misses the event") for e in ev
                    if info[id(e)]["twins"] and any(same(t.label, g["label"]) and
                    R.E._overlap_ok(t.start, t.end, g["start"], g["end"]) for t in info[id(e)]["twins"])]
            veto = [r for _e, r in vsp]
            veto_spans = [{"label": e.label, "origin": info[id(e)]["origin"], "conf": round(e.confidence, 3), "reason": r,
                           "flex_clipmax_family": round(float(D.clip_peak(cc[1]).get(key(e), -1.0)), 3),
                           "twins": [t.label for t in info[id(e)]["twins"]]} for e, r in vsp]
            rel = [(e, related(e.label, g["label"])) for e in ev if ov(e, g) > 0]
            rel = [(e, r) for e, r in rel if r]
            tim = [e for e in ev if same(e.label, g["label"]) and max(g["start"] - e.end, e.start - g["end"]) <= NEAR]
            flags = {"iv": bool(veto), "v": bool(rel), "t": bool(tim), "ii": 0.4 <= vf < R4.FBAR, "iii": D.AED <= vb < D.DISP,
                     "i": max(vb, vf) < HEARD, "above_bar": vf >= R4.FBAR or vb >= D.DISP}
            prim = next((k for k in ("iv", "v", "t", "ii", "iii", "i") if flags[k]), "vi")
            sub = None
            if prim == "iv":
                sub = sorted(set(veto))[0] if len(set(veto)) == 1 else " + ".join(sorted(set(veto)))
            elif prim == "i":
                sub = "FlexSED not asked for the family" if ff is None else "both asked, both silent"
            elif prim == "vi":
                sub = ("above the bar, no matching span" if flags["above_bar"] else
                       "FlexSED 0.2-0.4 only" if 0.2 <= vf < 0.4 else "other")
            misses.append({"clip": c["id"], "label": g["label"], "family": canonical(g["label"]), "start": g["start"],
                           "end": g["end"], "bin": prim, "sub": sub, "flags": flags,
                           "beats_near": None if fb is None else round(fb, 3), "flex_near": None if ff is None else round(ff, 3),
                           "related_shown": [{"label": e.label, "via": r} for e, r in rel], "veto_spans": veto_spans,
                           "onset_hit": bool(on_hit), "masked": g["masked"], "beats_speech_or_music_03": bool(max(speech_b, music_b) >= 0.3),
                           "stratum": c.get("stratum")})
    # ---------------------------------------------------------------- baseline check
    S = R4.summary(cl, rows)
    for k in ("C_overlap", "C_onset", "recall_overlap", "fp_per_min"):
        assert abs(S[k] - ref[k]) <= 1e-9, f"{set_name}: {k} {S[k]} != round-5 gate-1 {ref[k]}"
    assert len(fps) == S["fp"], (len(fps), S["fp"])
    print(f"[{set_name}] label lists: BEATs {len(BF.lists)} distinct, FlexSED {len(FF.lists)} distinct "
          f"({sorted(len(x) for x in FF.lists)[:5]} labels)", flush=True)
    assert len(misses) == sum(r["miss_overlap"] for r in rows), (len(misses), sum(r["miss_overlap"] for r in rows))
    n_on = sum(r["miss_onset"] for r in rows)
    assert n_on == len(timing) + sum(not x["onset_hit"] for x in misses), (n_on, len(timing))
    return cl, S, fps, misses, timing


# ----------------------------------------------------------------------------- summaries
def summarise(cl, S, fps, misses, timing):
    nf, nm = len(fps), len(misses)
    out = {"clips": len(cl), "baseline": {k: S[k] for k in ("C_overlap", "C_onset", "recall_overlap", "recall_onset",
                                                            "fp", "fp_per_min", "n_conseq")}}
    # 1. false spans
    fb = Counter(x["bin"] for x in fps)
    fam = {b: Counter(x["family"] for x in fps if x["bin"] == b) for b in "abc"}
    pairs = Counter(f"{x['family']} over {x['related_gold'][0]['label']} (via {x['related_gold'][0]['via']})"
                    for x in fps if x["bin"] == "a")
    vias = Counter(x["related_gold"][0]["via"] for x in fps if x["bin"] == "a")
    gaps = [x["gap_to_same_family_s"] for x in fps if x["bin"] == "b"]
    c = [x for x in fps if x["bin"] == "c"]
    out["false"] = {
        "n": nf, "bins": {b: {"n": fb[b], "pct": pct(fb[b], nf)} for b in "abc"},
        "a_and_b": sum(x["a_and_b"] for x in fps),
        "top_families": {b: table(fam[b], fb[b], 10) for b in "abc"},
        "a_top_pairs": table(pairs, fb["a"], 10), "a_top_via": table(vias, fb["a"], 10),
        "b_gap_s": qstats(gaps),
        "b_gap_bins": table(Counter(bucket(g, [0.5, 1.0, 2.0, 99], ["< 0.5 s", "0.5-1 s", "1-2 s", ">= 2 s"]) for g in gaps), len(gaps)),
        "c_origin": table(Counter(x["origin"] for x in c), len(c)),
        "c_speech_music": {
            "speech_ge_0.3": {"n": sum(x["beats_speech"] >= 0.3 for x in c), "pct": pct(sum(x["beats_speech"] >= 0.3 for x in c), len(c))},
            "music_ge_0.3": {"n": sum(x["beats_music"] >= 0.3 for x in c), "pct": pct(sum(x["beats_music"] >= 0.3 for x in c), len(c))},
            "either_ge_0.3": {"n": sum(max(x["beats_speech"], x["beats_music"]) >= 0.3 for x in c),
                              "pct": pct(sum(max(x["beats_speech"], x["beats_music"]) >= 0.3 for x in c), len(c))}},
        "c_by_origin_speech_music": {o: {"n": sum(x["origin"] == o for x in c),
                                         "either_ge_0.3": sum(x["origin"] == o and max(x["beats_speech"], x["beats_music"]) >= 0.3 for x in c)}
                                     for o in ("BEATs only", "both", "FlexSED only")},
        "c_beats_conf": table(Counter(bucket(x["conf"], [0.5, 0.7, 9], ["0.35-0.5", "0.5-0.7", ">= 0.7"])
                                      for x in c if x["origin"] != "FlexSED only"), sum(x["origin"] != "FlexSED only" for x in c)),
        "c_flex_in_span_beats_origin": table(Counter(bucket(x["flex_in_span"], [0.3, 0.5, 0.8, 9], ["< 0.3", "0.3-0.5", "0.5-0.8", ">= 0.8"])
                                                     for x in c if x["origin"] != "FlexSED only"), sum(x["origin"] != "FlexSED only" for x in c)),
        "c_beats_clipmax_flex_only": table(Counter(bucket(x["beats_clipmax"], [0.1218, 0.2, 0.35, 9], ["< b", "b-0.2", "0.2-0.35", ">= 0.35"])
                                                   for x in c if x["origin"] == "FlexSED only"), sum(x["origin"] == "FlexSED only" for x in c)),
        "c_gold_under": table(Counter(l for x in c for l in (x["gold_under"] or ["(nothing labelled)"])), len(c), 10),
        "by_origin_all_bins": {b: table(Counter(x["origin"] for x in fps if x["bin"] == b), fb[b]) for b in "abc"},
    }
    # 2. misses
    mb = Counter(x["bin"] for x in misses)
    order = ["iv", "v", "t", "ii", "iii", "i", "vi"]
    out["miss"] = {
        "n": nm, "bins": {b: {"n": mb[b], "pct": pct(mb[b], nm), "masked": sum(x["masked"] for x in misses if x["bin"] == b),
                              "beats_sm_03": sum(x["beats_speech_or_music_03"] for x in misses if x["bin"] == b)} for b in order},
        "flags_nonexclusive": {k: {"n": sum(x["flags"][k] for x in misses), "pct": pct(sum(x["flags"][k] for x in misses), nm)}
                               for k in ("iv", "v", "t", "ii", "iii", "i", "above_bar")},
        "ii_and_iii": sum(x["flags"]["ii"] and x["flags"]["iii"] for x in misses),
        "sub": {b: table(Counter(x["sub"] for x in misses if x["bin"] == b), mb[b]) for b in ("iv", "i", "vi")},
        "top_families": {b: table(Counter(x["family"] for x in misses if x["bin"] == b), mb[b], 10) for b in order},
        "v_top_names": table(Counter(f"{x['family']} heard as {x['related_shown'][0]['label']} (via {x['related_shown'][0]['via']})"
                                     for x in misses if x["bin"] == "v"), mb["v"], 10),
        "masked": {"n": sum(x["masked"] for x in misses), "pct": pct(sum(x["masked"] for x in misses), nm)},
        "beats_sm_03": {"n": sum(x["beats_speech_or_music_03"] for x in misses), "pct": pct(sum(x["beats_speech_or_music_03"] for x in misses), nm)},
        "flex_not_asked": sum(x["flex_near"] is None for x in misses),
    }
    # 3. timing
    errs = [x["err"] for x in timing]
    side = lambda e: "early (< -0.5 s)" if e < -R.EARLY else "late (> +1.0 s)"
    out["timing"] = {
        "n": len(timing), "err_s": qstats(errs),
        "side": table(Counter(side(e) for e in errs), len(errs)),
        "late_bins": table(Counter(bucket(e, [1.5, 2.5, 99], ["1.0-1.5 s", "1.5-2.5 s", ">= 2.5 s"]) for e in errs if e > R.LATE),
                           sum(e > R.LATE for e in errs)),
        "early_bins": table(Counter(bucket(-e, [1.0, 2.0, 99], ["0.5-1 s early", "1-2 s early", ">= 2 s early"]) for e in errs if e < -R.EARLY),
                            sum(e < -R.EARLY for e in errs)),
        "source": {s: {"n": sum(x["source"] == s for x in timing), "pct": pct(sum(x["source"] == s for x in timing), len(timing)),
                       "early": sum(x["source"] == s and x["err"] < -R.EARLY for x in timing),
                       "late": sum(x["source"] == s and x["err"] > R.LATE for x in timing),
                       "err_s": qstats([x["err"] for x in timing if x["source"] == s])}
                   for s in ("BEATs", "FlexSED twin rule", "FlexSED only")},
        "merge": {"n": sum(x["merge"] for x in timing), "pct": pct(sum(x["merge"] for x in timing), len(timing)),
                  "early": sum(x["merge"] and x["err"] < -R.EARLY for x in timing)},
        "masked": sum(x["masked"] for x in timing),
        "top_families": table(Counter(canonical(x["label"]) for x in timing), len(timing), 10),
    }
    # the harness scores with config.LABEL_FILTER as imported (the default); the shipped profile draws only "depictable"
    # labels -- how many audited items the shipped filter would never show (descriptive; nothing re-scored)
    old = config.LABEL_FILTER
    try:
        config.LABEL_FILTER = "depictable"
        okd = lambda l: is_salient_nonspeech(l) and not is_music(l)
        nd = [x for x in fps if not okd(x["label"])]
        out["shipped_filter_check"] = {
            "harness_filter": old, "false_not_depictable": {"n": len(nd), "pct": pct(len(nd), nf),
                                                           "bins": dict(Counter(x["bin"] for x in nd)),
                                                           "top": table(Counter(x["family"] for x in nd), len(nd), 10)},
            "misses_not_depictable": table(Counter(x["label"] for x in misses if not okd(x["label"])), nm)}
    finally:
        config.LABEL_FILTER = old
    return out


def md(res):
    """side-by-side markdown tables (counts and %) for the doc"""
    A, H = res["calib"]["summary"], res["heldout"]["summary"]
    L = []
    cell = lambda d: f"{d['n']} ({d['pct']}%)"
    L.append("| | 280 (fit) | 415 (held-out) |\n|---|---|---|")
    for k in ("C_overlap", "C_onset", "recall_overlap", "fp", "fp_per_min", "n_conseq"):
        f = (lambda v: f"{v:.3f}") if isinstance(A["baseline"][k], float) else str
        L.append(f"| {k} | {f(A['baseline'][k])} | {f(H['baseline'][k])} |")
    L.append("\n**False spans**\n\n| bin | 280 | 415 |\n|---|---|---|")
    names = {"a": "a related family overlaps", "b": "same family elsewhere (timing)", "c": "phantom (no related gold)"}
    for b in "abc":
        L.append(f"| {names[b]} | {cell(A['false']['bins'][b])} | {cell(H['false']['bins'][b])} |")
    L.append(f"| total | {A['false']['n']} | {H['false']['n']} |")
    L.append(f"| (a and b both true) | {A['false']['a_and_b']} | {H['false']['a_and_b']} |")
    L.append("\n**Misses (consequential, C-overlap)**\n\n| bin | 280 | 415 |\n|---|---|---|")
    mn = {"iv": "iv heard, removed by a veto", "v": "v heard under a related name", "t": "t same-family span shown, too little overlap",
          "ii": "ii FlexSED 0.4-0.8", "iii": "iii BEATs 0.175-0.35", "i": "i unheard (< 0.2)", "vi": "vi other"}
    for b in mn:
        a, h = A["miss"]["bins"][b], H["miss"]["bins"][b]
        L.append(f"| {mn[b]} | {a['n']} ({a['pct']}%), masked {a['masked']} | {h['n']} ({h['pct']}%), masked {h['masked']} |")
    L.append(f"| total | {A['miss']['n']}, masked {A['miss']['masked']['n']} | {H['miss']['n']}, masked {H['miss']['masked']['n']} |")
    L.append("\n**Timing (overlap hit, onset miss)**\n\n| | 280 | 415 |\n|---|---|---|")
    L.append(f"| n | {A['timing']['n']} | {H['timing']['n']} |")
    ea, eh = A["timing"]["err_s"] or {}, H["timing"]["err_s"] or {}
    L.append(f"| onset error median [q1, q3] s | {ea.get('median')} [{ea.get('q1')}, {ea.get('q3')}] | {eh.get('median')} [{eh.get('q1')}, {eh.get('q3')}] |")
    for s in ("BEATs", "FlexSED twin rule", "FlexSED only"):
        a, h = A["timing"]["source"][s], H["timing"]["source"][s]
        L.append(f"| set by {s} | {a['n']} ({a['pct']}%; early {a['early']}, late {a['late']}) | {h['n']} ({h['pct']}%; early {h['early']}, late {h['late']}) |")
    L.append(f"| merge flag | {A['timing']['merge']['n']} | {H['timing']['merge']['n']} |")
    return "\n".join(L)


def main():
    r5 = json.loads(R5.OUT.read_text(encoding="utf-8"))
    res = {"note": "descriptive audit of the shipped stack (b 0.1218, PANNs off); no selection; nothing on DEV/TEST",
           "definitions": __doc__}
    for set_name in ("calib", "heldout"):
        ref = r5[f"gate1_{set_name}"]["baseline_shipped_b"]
        cl, S, fps, misses, timing = audit(set_name, ref)
        res[set_name] = {"summary": summarise(cl, S, fps, misses, timing), "false_spans": fps, "misses": misses, "timing": timing}
        b = res[set_name]["summary"]["baseline"]
        print(f"[{set_name}] {len(cl)} clips; baseline reproduced: C-ov {b['C_overlap']:.4f} C-on {b['C_onset']:.4f} "
              f"recall {b['recall_overlap']:.4f} fp {b['fp']} ({b['fp_per_min']:.3f}/min)", flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(md(res))
    for s in ("calib", "heldout"):
        print(f"\n===== {s} summary =====")
        print(json.dumps(res[s]["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
