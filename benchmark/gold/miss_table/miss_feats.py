"""feature dump for the B0 (scored, proposed) DEV misses and wrong pictures; loaders from dev_candidates_check / score_per_sound"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("P:/MscProj")
sys.path.insert(0, str(ROOT))
SCR = Path(__file__).resolve().parent / "cl"
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import score_per_sound as S
from src.labels import canonical, ancestors, is_salient_nonspeech
import config

C.BEATS_DIR = SCR / "j2_dev_beats"
C.scored_dir = lambda sysn: SCR / f"protocol_{sysn}_{C.TAG}"
PANNS_DIR = ROOT / "benchmark" / "gold" / "panns_fw"
DISP = C.F["DISP"]


def sal(l):
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        return is_salient_nonspeech(l)
    finally:
        config.LABEL_FILTER = old


def fam_cols(labs, glab):
    return [i for i, l in enumerate(labs) if S.same_family(l, glab)]


def best(fr, glab, a, b):
    fw, ts, labs = fr
    cols = fam_cols(labs, glab)
    m = (ts >= a) & (ts <= b)
    if not cols or not m.any():
        return 0.0, None
    sub = fw[m][:, cols]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    return float(sub[k]), labs[cols[k[1]]]


def runs(fr, glab, a, b, thr=0.4, exact=False):
    """longest contiguous run >= thr (max over family columns, or the one column glab if exact) that touches [a, b]"""
    fw, ts, labs = fr
    cols = [labs.index(glab)] if exact else fam_cols(labs, glab)
    if not cols:
        return 0.0, None
    s = fw[:, cols].max(axis=1) >= thr
    dt = ts[1] - ts[0]
    bestl, bestr, i, n = 0.0, None, 0, len(s)
    while i < n:
        if not s[i]:
            i += 1; continue
        j = i
        while j < n and s[j]:
            j += 1
        r0, r1 = float(ts[i]), float(ts[j - 1] + dt)
        if r1 >= a and r0 <= b and r1 - r0 > bestl:
            bestl, bestr = r1 - r0, [round(r0, 2), round(r1, 2)]
        i = j
    return round(bestl, 2), bestr


def parent(l):
    an = ancestors(l)
    return an[0] if an else None


def sibling(a, b):
    """near label: not the same family, but sharing a direct parent or one is the other's grandparent-sibling"""
    if S.same_family(a, b):
        return False
    top = {"Sounds of things", "Animal", "Human sounds", "Natural sounds", "Music", "Channel, environment and background",
           "Source-ambiguous sounds"}
    pa, pb = parent(a), parent(b)
    return bool(pa and pa == pb and pa not in top)


def main():
    gold, stems = C.dev_stems()
    s4all = json.loads((ROOT / "data/work/devcand/stage4.json").read_text(encoding="utf-8"))["arms"]
    s4, s4b1 = s4all["B0r|proposed"], s4all["B1|proposed"]
    out = {"misses": [], "wrong": [], "hits": []}
    for st in stems:
        sd = C.scored_dir("proposed") / st
        pics = S.load_pictures(C.scored_dir("proposed"), st, "proposed") or []
        rp = S.load_pictures(SCR / "devcand" / "B0r_proposed", st, "proposed") or []
        assert C.pics_sig(rp) == C.pics_sig(pics), st
        specs = json.loads((sd / "augmentations.json").read_text(encoding="utf-8"))
        trace = json.loads((sd / "onset_trace.json").read_text(encoding="utf-8"))
        votes = json.loads((sd / "gate_votes.json").read_text(encoding="utf-8")) if (sd / "gate_votes.json").exists() else []
        fr = {"B": C.load_fr(C.BEATS_DIR / f"{st}.npz"), "F": C.load_fr(C.FLEX_DIR / f"{st}.npz"),
              "P": C.load_fr(PANNS_DIR / f"{st}.npz")}
        rows4 = s4[st]
        flex04 = C.ext(fr["F"], 0.4)
        bc = C.band_cands({"beats": fr["B"], "flex": fr["F"]})
        pk, bk = C.clip_peak(fr["P"]), C.clip_peak(fr["B"])
        nh = C.needed_hit(gold[st], pics)
        for g, h in nh:
            on, en = g["start"], g["end"]
            wa, wb = on - S.EARLY, on + S.LATE
            na, nb = on - 1.0, on + 2.0
            r = {"clip": st, "label": g["label"], "onset": round(on, 2), "end": round(en, 2), "dur": round(en - on, 2),
                 "importance": g["importance"], "hit": bool(h)}
            for k, name in (("B", "beats"), ("F", "flex"), ("P", "panns")):
                v, l = best(fr[k], g["label"], wa, wb)
                r[f"{name}_win"], r[f"{name}_win_label"] = round(v, 3), l
                v, l = best(fr[k], g["label"], na, nb)
                r[f"{name}_near"], r[f"{name}_near_label"] = round(v, 3), l
                v, l = best(fr[k], g["label"], on - 1.0, en + 1.0)
                r[f"{name}_sound"] = round(v, 3)
            r["flex_run04"], r["flex_run04_span"] = runs(fr["F"], g["label"], wa, wb)
            r["flex_run04_near"], r["flex_run04_near_span"] = runs(fr["F"], g["label"], na, nb)
            r["flex_run08"], r["flex_run08_span"] = runs(fr["F"], g["label"], wa, wb, 0.8)
            if r["flex_win_label"]:
                r["flex_run04_bestlab"], r["flex_run04_bestlab_span"] = runs(fr["F"], r["flex_win_label"], wa, wb, 0.4, True)
            else:
                r["flex_run04_bestlab"], r["flex_run04_bestlab_span"] = 0.0, None
            if r["beats_win_label"]:
                r["beats_run0175_bestlab"], r["beats_run0175_span"] = runs(fr["B"], r["beats_win_label"], wa, wb, 0.175, True)
                r["beats_run035_bestlab"], r["beats_run035_span"] = runs(fr["B"], r["beats_win_label"], wa, wb, 0.35, True)
            r["band_cands_fam"] = [[e.label, round(e.start, 2), round(e.end, 2), round(e.confidence, 3)] for e in bc
                                   if S.same_family(e.label, g["label"]) and S.in_window(e.start, on, S.EARLY, S.LATE)]
            r["panns_clipmax_fam"] = round(max([v for k, v in pk.items() if S.same_family(k, g["label"])] or [0.0]), 3)
            r["beats_clipmax_fam"] = round(max([v for k, v in bk.items() if S.same_family(k, g["label"])] or [0.0]), 3)
            r["s4_B1_fam"] = [[x["label"], round(x["start"], 2), round(x["end"], 2), round(x["conf"], 3), x["origin"]]
                              for x in s4b1[st] if S.same_family(x["label"], g["label"]) and x["start"] <= en + 1.0 and x["end"] >= on - 1.0]
            r["s4_other_inwin_aed"] = [[x["label"], round(x["start"], 2), round(x["conf"], 3), x["origin"]] for x in rows4
                                       if not S.same_family(x["label"], g["label"]) and S.in_window(x["start"], on, S.EARLY, S.LATE)
                                       and sal(x["label"])]
            r["flex04_spans_fam"] = [[e.label, round(e.start, 2), round(e.end, 2), round(e.confidence, 3)] for e in flex04
                                     if S.same_family(e.label, g["label"]) and e.start <= en + 1.0 and e.end >= on - 1.0]
            # stage 4 (B0r = scored, after vetoes and onset refinement)
            fam4 = [x for x in rows4 if S.same_family(x["label"], g["label"])]
            r["s4_fam"] = [[x["label"], round(x["start"], 2), round(x["end"], 2), round(x["conf"], 3), x["origin"],
                            S.in_window(x["start"], on, S.EARLY, S.LATE)] for x in fam4
                           if x["start"] <= en + 1.0 and x["end"] >= on - 1.0]
            r["s4_other_inwin"] = [[x["label"], round(x["start"], 2), round(x["conf"], 3), x["origin"]] for x in rows4
                                   if not S.same_family(x["label"], g["label"]) and S.in_window(x["start"], on, S.EARLY, S.LATE)
                                   and x["conf"] >= DISP and sal(x["label"])]
            r["s4_sibling_inwin"] = [[x["label"], round(x["start"], 2), round(x["conf"], 3)] for x in rows4
                                     if sibling(x["label"], g["label"]) and S.in_window(x["start"], on, S.EARLY, S.LATE)]
            # trace: FlexSED raw 0.8 spans of the family (before the vetoes)
            r["trace_flexraw_fam"] = [[x["label"], round(x["start"], 2), round(x["end"], 2)] for x in trace
                                      if x["step"] == "flexsed_raw" and S.same_family(x["label"], g["label"])
                                      and x["start"] <= en + 1.0 and x["end"] >= on - 1.0]
            r["trace_veto_fam"] = [[x["label"], round(x["start"], 2), round(x["end"], 2)] for x in trace
                                   if x["step"] == "veto" and S.same_family(x["label"], g["label"])
                                   and x["start"] <= en + 1.0 and x["end"] >= on - 1.0]
            # stage 5 specs of the family and final pictures
            r["specs_fam"] = [[s["event_label"], round(s["start"], 2), round(s["end"], 2), round(s.get("confidence", 0), 3),
                               bool(s.get("augment")), s.get("reason", ""), [[round(a, 2), round(b, 2)] for a, b in (s.get("spans") or [])]]
                              for s in specs if S.same_family(s["event_label"], g["label"]) and s["start"] <= en + 1.0 and s["end"] >= on - 1.0]
            r["pics_fam"] = [[l, round(a, 2), round(b, 2)] for l, a, b in pics if S.same_family(l, g["label"]) and b >= on - 1.0 and a <= en + 1.0]
            r["pics_other_inwin"] = [[l, round(a, 2), round(b, 2)] for l, a, b in pics if not S.same_family(l, g["label"]) and S.in_window(a, on, S.EARLY, S.LATE)]
            r["gate_fam"] = [[v["label"], [round(x, 2) for x in v["stretch"]], v["seen"], v.get("named", "")] for v in votes
                             if S.same_family(v["label"], g["label"])]
            out["hits" if h else "misses"].append(r)
        # wrong pictures: class of each picture = the change of score_clip's counts when it is added (same order)
        ps = sorted(pics, key=lambda p: p[1])
        prev = S.score_clip(gold[st], [])
        for k, p in enumerate(ps):
            cur = S.score_clip(gold[st], ps[:k + 1])
            cls = [c for c in ("hit", "visible", "cross", "phantom", "dup", "dontcare", "collision") if cur[c] != prev[c]]
            prev = cur
            if not any(c in cls for c in ("visible", "cross", "phantom")):
                continue
            typ = [c for c in cls if c in ("visible", "cross", "phantom")][0]
            l, a, b = p
            src = [x for x in rows4 if S.same_family(x["label"], l) and min(b, x["end"]) - max(a, x["start"]) > -0.01]
            srcs = sorted(src, key=lambda x: abs(x["start"] - a))
            coll = [[g["label"], round(g["start"], 2), round(g["end"], 2), "needed" if g["needed"] else ("visible" if g["visible"] else "obvious")]
                    for g in gold[st] if S.in_window(a, g["start"], S.EARLY, S.LATE) or (g["start"] <= a <= g["end"])]
            vis_g = [[g["label"], round(g["start"], 2), "visible" if g["visible"] else "obvious"] for g in gold[st]
                     if S.same_family(l, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE) and not g["needed"]]
            spec = [s for s in specs if s.get("augment") and S.same_family(s["event_label"], l)]
            fb = best(fr["F"], l, a - 0.5, a + 1.0)
            bb = best(fr["B"], l, a - 0.5, a + 1.0)
            pb = best(fr["P"], l, a - 0.5, a + 1.0)
            out["wrong"].append({"clip": st, "label": l, "start": round(a, 2), "end": round(b, 2), "type": typ,
                                 "stage4_src": [[x["label"], round(x["start"], 2), round(x["end"], 2), round(x["conf"], 3), x["origin"]] for x in srcs[:3]],
                                 "spec_conf": [round(s.get("confidence", 0), 3) for s in spec],
                                 "spec_detail": [s.get("detail", "") for s in spec],
                                 "beats_at": [round(bb[0], 3), bb[1]], "flex_at": [round(fb[0], 3), fb[1]], "panns_at": [round(pb[0], 3), pb[1]],
                                 "gold_at_time": coll, "same_family_not_needed": vis_g})
    (SCR.parent / "feats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(len(out["hits"]), "hits", len(out["misses"]), "misses", len(out["wrong"]), "wrong",
          {t: sum(1 for w in out["wrong"] if w["type"] == t) for t in ("visible", "cross", "phantom")})


if __name__ == "__main__":
    main()
