"""Detector round 12 (docs/prereg_round12_v2.md, 2026-09-28): the corrected AudioSet cost "v2", every earlier cell
re-scored under it, four new cells (N1 weak-twin, N2 short-sound path, N3 combination, N4 DASM-filtered ontology vetoes).

v2 per clip: gold names by MID (setup_audit.remap_clip) -> consequential AND drawn by the shipped "depictable" filter AND not
music -> runs of one canonical family merged when the pause <= 2.0 s (config.MERGE_GAP, the gold rule) -> hit / false rules
with the video benchmark's window [onset - 0.5, onset + 1.0] (score_per_sound EARLY / LATE) -> 4 x missed + 2 x false.

    python benchmark/detector_round12.py step1              # gate 0 + ours vs "show nothing" under v2 (280); STOP rule
    python benchmark/detector_round12.py rescore            # every earlier cell under v2 (280)
    python benchmark/detector_round12.py new                # N1, N2, N3, N4 (+ references) under v2 (280)
    python benchmark/detector_round12.py picks              # <= 3 picks (both v2 costs below ours)
    python benchmark/detector_round12.py heldout            # the picks on the 415 + Holm
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import detector_round8 as M
from benchmark import setup_audit as SA
from src.labels import canonical, is_salient_nonspeech, is_music
from src.types import AudioEvent

R, D = M.R, M.D
OUT = _ROOT / "benchmark" / "detector_round12.json"
EARLY, LATE, GAP = 0.5, 1.0, 2.0
BAND, SHORT_PEAK, SHORT_MIN_FRAMES = 0.4, 0.5, 2
key, same = M.key, M.same
V2 = dict(remap=True, filt="depictable", merge=True, tol=True, fwin=True)
OLD = dict(remap=False, filt="lists", merge=False, tol=False, fwin=False)
REFERENCE = {"R0", "A0", "N1-all", "N2-all"}


def load_log():
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save_log(log):
    OUT.write_text(json.dumps(log, indent=1, default=float), encoding="utf-8")


# ============================================================================= the v2 cost
_MAP = None


def mid_map():
    global _MAP
    if _MAP is None:
        _MAP = SA.strong_to_beats()
    return _MAP


def ok_label(lab):
    return is_salient_nonspeech(lab) and not is_music(lab)


def merge_runs(scored, gap=GAP):
    fam = {}
    for g in scored:
        fam.setdefault(canonical(g["label"]), []).append(g)
    out = []
    for gs in fam.values():
        gs = sorted(gs, key=lambda g: (g["start"], g["end"]))
        cur = {"label": gs[0]["label"], "labels": {gs[0]["label"]}, "start": gs[0]["start"], "end": gs[0]["end"], "n": 1}
        for g in gs[1:]:
            if g["start"] - cur["end"] <= gap:
                cur["end"] = max(cur["end"], g["end"]); cur["labels"].add(g["label"]); cur["n"] += 1
            else:
                out.append(cur)
                cur = {"label": g["label"], "labels": {g["label"]}, "start": g["start"], "end": g["end"], "n": 1}
        out.append(cur)
    return sorted(out, key=lambda g: g["start"])


def gold2(c, remap=True, filt="depictable", merge=True, **_kw):
    """(scored events, events for the false-span test) of one clip under the given parts of v2"""
    cc = SA.remap_clip(c, mid_map()) if remap else c
    with M.label_filter(filt):
        flags = [bool(g["consequential"] and ok_label(g["label"])) for g in cc["events"]]
    scored = [g for g, f in zip(cc["events"], flags) if f]
    others = [dict(g, labels={g["label"]}) for g, f in zip(cc["events"], flags) if not f]
    if merge:
        sc = merge_runs(scored)
    else:
        sc = [dict(g, labels={g["label"]}, n=1) for g in scored]
    allg = sc + others if merge else [dict(g, labels={g["label"]}) for g in cc["events"]]
    return sc, allg


def cost2(g2, ev, filt="depictable", tol=True, fwin=True, **_kw):
    sc, allg = g2
    with M.label_filter(filt):
        ev = [e for e in ev if ok_label(e.label)]
    m = lambda e, g: any(same(e.label, l) for l in g["labels"])
    win = lambda e, g: g["start"] - EARLY <= e.start <= g["start"] + LATE
    ovl = lambda e, g: min(e.end, g["end"]) - max(e.start, g["start"]) > 0
    if tol:
        hit_ov = lambda e, g: ovl(e, g) or win(e, g)
    else:
        hit_ov = lambda e, g: R.E._overlap_ok(e.start, e.end, g["start"], g["end"])
    miss_ov = sum(not any(m(e, g) and hit_ov(e, g) for e in ev) for g in sc)
    miss_on = sum(not any(m(e, g) and win(e, g) for e in ev) for g in sc)
    true_ = (lambda e, g: ovl(e, g) or win(e, g)) if fwin else ovl
    fp = sum(not any(m(e, g) and true_(e, g) for g in allg) for e in ev)
    return {"C_overlap": 4 * miss_ov + 2 * fp, "C_onset": 4 * miss_on + 2 * fp, "fp": fp, "miss_overlap": miss_ov,
            "miss_onset": miss_on, "n_conseq": len(sc), "shown": len(ev)}


class Scorer:
    """one clip set, one cost variant: gold prepared once"""
    def __init__(self, xs, **parts):
        self.parts = dict(V2, **parts)
        self.xs = xs
        self.g2 = [gold2(x.c, **self.parts) for x in xs]
        self.minutes = sum(x.c["duration"] for x in xs) / 60.0

    def rows(self, evs):
        return [cost2(g, ev, **self.parts) for g, ev in zip(self.g2, evs)]

    def summ(self, rows):
        n = max(1, sum(r["n_conseq"] for r in rows))
        return {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
                "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / n, "recall_onset": 1 - sum(r["miss_onset"] for r in rows) / n,
                "fp": int(sum(r["fp"] for r in rows)), "fp_per_min": sum(r["fp"] for r in rows) / self.minutes,
                "shown": int(sum(r["shown"] for r in rows)), "n_events": int(sum(r["n_conseq"] for r in rows)),
                "clips_with_events": int(sum(r["n_conseq"] > 0 for r in rows))}


def verdict(d):
    return "better" if d[2] < 0 else ("worse" if d[1] > 0 else "same")


def compare(rows, brows):
    dov = M.boot8([a["C_overlap"] - b["C_overlap"] for a, b in zip(rows, brows)])
    don = M.boot8([a["C_onset"] - b["C_onset"] for a, b in zip(rows, brows)])
    return {"dC_overlap": dov, "dC_onset": don, "verdict_overlap": verdict(dov), "verdict_onset": verdict(don)}


def show(name, S):
    f = lambda v: "" if not v else f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    print(f"{name:14s} C-ov {S['C_overlap']:.3f} {f(S.get('dC_overlap')):26s} {S.get('verdict_overlap', ''):6s} | "
          f"C-on {S['C_onset']:.3f} {f(S.get('dC_onset')):26s} {S.get('verdict_onset', ''):6s} | rec {S['recall_overlap']:.1%} / "
          f"{S['recall_onset']:.1%} fp {S['fp']} ({S['fp_per_min']:.2f}/min) shown {S['shown']}"
          + (f" | old C {S['old_C_overlap']:.3f}/{S['old_C_onset']:.3f}" if 'old_C_overlap' in S else ""), flush=True)


# ============================================================================= step 1
def step1(log):
    cl, xs, base_evs, brows_old, _rd, Sb_old = M.gate_set("calib", log.setdefault("_gate", {}))
    res = {"gate0_old_C": {k: Sb_old[k] for k in ("C_overlap", "C_onset", "recall_overlap", "fp", "shown")}, "clips": len(xs)}
    # sanity: the generic scorer with every v2 part off equals detector_round2.clip_cost on every clip
    so = Scorer(xs, **OLD)
    for x, g, ev in zip(xs, so.g2, base_evs):
        a, b = cost2(g, ev, **OLD), D.clip_cost(x.c, ev)
        assert (a["C_overlap"], a["C_onset"], a["fp"]) == (b["C_overlap"], b["C_onset"], b["fp"]), x.cid
    res["sanity_old_equals_clip_cost"] = True
    variants = [("old C (all parts off)", OLD),
                ("+ MID names", dict(OLD, remap=True)),
                ("+ depictable filter", dict(OLD, remap=True, filt="depictable")),
                ("+ runs merged (<= 2 s)", dict(OLD, remap=True, filt="depictable", merge=True)),
                ("+ window hit rule (old false rule)", dict(V2, fwin=False)),
                ("v2 (+ window false rule)", V2)]
    empty = [[] for _ in xs]
    tab = {}
    for name, parts in variants:
        sc = Scorer(xs, **parts)
        ro, rn = sc.rows(base_evs), sc.rows(empty)
        So, Sn = sc.summ(ro), sc.summ(rn)
        d = compare(ro, rn)                              # ours - nothing
        tab[name] = {"ours": So, "nothing": Sn, "ours_minus_nothing": d}
        print(f"[step1] {name:36s} ours C-ov {So['C_overlap']:.3f} C-on {So['C_onset']:.3f} (fp {So['fp']}, events {So['n_events']}, "
              f"rec {So['recall_overlap']:.1%}/{So['recall_onset']:.1%}) | nothing {Sn['C_overlap']:.3f} | ours - nothing "
              f"ov {d['dC_overlap'][0]:+.3f} [{d['dC_overlap'][1]:+.3f}, {d['dC_overlap'][2]:+.3f}] on {d['dC_onset'][0]:+.3f} "
              f"[{d['dC_onset'][1]:+.3f}, {d['dC_onset'][2]:+.3f}]", flush=True)
    res["table"] = tab
    v = tab["v2 (+ window false rule)"]["ours_minus_nothing"]["dC_overlap"]
    res["ours_beats_nothing"] = bool(v[2] < 0)
    sc = Scorer(xs)
    res["v2_event_counts"] = {"events": int(sum(len(g[0]) for g in sc.g2)), "merged_from": int(sum(g["n"] for gg in sc.g2 for g in gg[0])),
                              "clips_with_events": int(sum(bool(g[0]) for g in sc.g2)),
                              "by_family": dict(Counter(canonical(g["label"]) for gg in sc.g2 for g in gg[0]).most_common())}
    log["step1"] = res
    save_log(log)
    print(f"[step1] ours beats nothing under v2 (upper CI of dC-overlap < 0): {res['ours_beats_nothing']}", flush=True)
    return res["ours_beats_nothing"]


# ============================================================================= cell builders (frozen values from each round's JSON)
def rj(n):
    return json.loads((_ROOT / "benchmark" / f"detector_round{n}.json").read_text(encoding="utf-8"))


class Ctx:
    def __init__(self, set_name):
        from benchmark import detector_round10 as R10
        self.set = set_name
        self.R10 = R10
        self.cl, self.xs, self.base_evs, self.brows_old, _rd, self.Sb_old = M.gate_set(set_name, {})
        self.xs10 = [R10.X(x.c, set_name) for x in self.xs]
        for x, y in zip(self.xs, self.xs10):
            y.b, y.f, y.p = x.b, x.f, x.p
        self._c = {}

    def cands10(self):
        if "c10" not in self._c:
            self._c["c10"] = json.loads(self.R10.cands_path().read_text(encoding="utf-8"))
        return self._c["c10"]

    def vlm(self):
        if "vlm" not in self._c:
            self._c["vlm"] = json.loads(M.vlm_path().read_text(encoding="utf-8"))
        return self._c["vlm"]


def b_round4(ctx, cell):
    from benchmark import detector_round4 as R4
    if cell == "R4-PANNs":
        return [R4.stack_tagged(x.cid, caches=(x.b, x.f, x.p))[0] for x in ctx.xs]
    out = []
    for x in ctx.xs:
        ev, info = M.stack8(x, tag=True)
        out.append(R4.trim(ev, info["flex"], x.f, {"beats_spans": 0, "trimmed": 0, "cut_s": [], "low_peak": 0}))
    return out


def b_round5(ctx, cell):
    from benchmark import detector_round5 as R5
    fr = rj(5)["fit"][cell]
    model = "eat" if cell.startswith("EAT") else "dasheng"
    out = []
    for x in ctx.xs:
        tag = R.load(R.E.WIN / R5.CACHE[model] / f"{x.cid}.npz")
        out.append(R5.finish(R5.pre(tag, x.f, float(fr["aed"])), float(fr["b"]), float(fr["disp"])))
    return out


def b_round6(ctx, cell):
    from benchmark import detector_round6 as R6
    bars = rj(6)["bars"]
    return [R6.stack6((x.b, x.f, x.p, R.load(R.E.WIN / R6.CACHE / f"{x.cid}.npz")), cell, bars["g"], bars["v"])[0] for x in ctx.xs]


def b_round8(ctx, cell):
    prm = dict(rj(8)["params"])
    prm["vlm"] = {ctx.set: ctx.vlm()}
    return M.run_cell(cell, ctx.xs, ctx.base_evs, prm)


def b_round9(ctx, cell):
    from benchmark import detector_round9 as R9
    return R9.run(cell, ctx.xs, {"beats_only": 0, "no_flank": 0, "dropped": 0})


def b_round10(ctx, cell):
    stat = {"considered": 0, "admitted": 0, "_adm": []}
    return ctx.R10.run_cell(cell, ctx.xs10, ctx.cands10(), ctx.vlm(), stat)


def b_round11(ctx, cell):
    from benchmark import detector_round11 as R11
    if "r11" not in ctx._c:
        xs11, base11, _b, _fl, C, _Sb = R11.prepare(ctx.set, copy.deepcopy(R11.load_log()))
        assert [x.cid for x in xs11] == [x.cid for x in ctx.xs]
        ctx._c["r11"] = (xs11, base11, C)
    xs11, base11, C = ctx._c["r11"]
    evs, _adm = R11.run_cell(cell, xs11, base11, C, best=rj(11).get("m7_base", "M2"))
    return evs


OLD_CELLS = ([("R4-PANNs", 4), ("R4-T1", 4)] + [(c, 5) for c in ("EAT-P", "EAT-R", "Dasheng-P", "Dasheng-R")]
             + [(c, 6) for c in ("D1", "D2")] + [(c, 8) for c in M.RP_CELLS + M.T_CELLS] + [(c, 9) for c in ("J1", "J2")]
             + [(c, 10) for c in ["R0"] + ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]]
             + [(c, 11) for c in ["A0", "M1", "M1m", "M1b", "M2", "M3", "M5", "M7", "C", "W"]])
BUILD = {4: b_round4, 5: b_round5, 6: b_round6, 8: b_round8, 9: b_round9, 10: b_round10, 11: b_round11}


def published_old(cell, rnd):
    """the round's own 280 number under the old C (a reproduction check), where the JSON has it"""
    try:
        if rnd == 5:
            v = rj(5)["fit"][cell]
        elif rnd == 6:
            v = rj(6)["fit"][cell]
        elif rnd in (8, 9, 10, 11):
            v = rj(rnd)["fit"][cell]
        else:
            return None
        return [v["C_overlap"], v["C_onset"]]
    except KeyError:
        return None


# ============================================================================= new cells (step 3)
def short_runs(x):
    """N2: runs >= 0.4 of 2..(< 0.5 s) frames with peak >= 0.5 -> 1-s spans centred on the peak, unioned per family"""
    fw, ts, labs = x.f
    dt = float(ts[1] - ts[0])
    t_end = float(ts[-1] + dt)
    raw = []
    for j, lab in enumerate(labs):
        s = fw[:, j]
        act = s >= BAND
        i, n = 0, len(s)
        while i < n:
            if not act[i]:
                i += 1; continue
            k = i
            while k < n and act[k]:
                k += 1
            dur = float(ts[k - 1] + dt - ts[i])
            if k - i >= SHORT_MIN_FRAMES and dur < config.AED_MIN_DUR and s[i:k].max() >= SHORT_PEAK:
                p = i + int(np.argmax(s[i:k]))
                tp = float(ts[p])
                raw.append(AudioEvent(label=lab, start=max(0.0, tp - 0.5), end=min(t_end, tp + 0.5), confidence=float(s[p])))
            i = k
    out = []
    for k_ in sorted({key(e) for e in raw}):
        es = sorted([e for e in raw if key(e) == k_], key=lambda e: e.start)
        cur = es[0]
        for e in es[1:]:
            if e.start < cur.end:
                cur = AudioEvent(label=cur.label, start=cur.start, end=max(cur.end, e.end), confidence=max(cur.confidence, e.confidence))
            else:
                out.append(cur); cur = e
        out.append(cur)
    return out


def near(e, evs):
    return any(key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end for y in evs)


def ovl_same(e, evs):
    return any(key(y) == key(e) and min(y.end, e.end) - max(y.start, e.start) > 0 for y in evs)


def n1_spans(ctx, x, filt=True):
    cs = ctx.R10.cands_opt(x, twin="shown")
    return [e for e in cs if ctx.R10.r1(x, e)] if filt else cs


def n2_spans(ctx, x, shown, filt=True):
    bp = D.clip_peak(x.b)
    out = []
    for e in short_runs(x):
        if near(e, shown) or bp.get(key(e), 1.0) < M.B:
            continue
        if filt and not ctx.R10.r1(x, e):
            continue
        out.append(e)
    return out


def b_new(ctx, cell):
    out, stat = [], {"added": 0, "added_false_v2": None}
    for x, be in zip(ctx.xs10, ctx.base_evs):
        if cell in ("N1", "N1-all"):
            add = n1_spans(ctx, x, cell == "N1")
            ev = list(be) + add
        elif cell in ("N2", "N2-all"):
            add = n2_spans(ctx, x, be, cell == "N2")
            ev = list(be) + add
        elif cell == "N3":
            ev = ctx.R10.stack10(x, i4="conf", i6=("conf", ctx.vlm()[x.cid]["families"]))
            for e in n1_spans(ctx, x):
                if not ovl_same(e, ev):
                    ev.append(e)
            for e in n2_spans(ctx, x, list(ev)):
                if not ovl_same(e, ev):
                    ev.append(e)
            add = ev
        elif cell == "N4":
            i9 = M.stack8(x, {"i9": True})
            ours = Counter(M.sig(be))
            extra = []
            for e in i9:
                s = M.sig([e])[0]
                if ours[s] > 0:
                    ours[s] -= 1
                else:
                    extra.append(e)
            assert sum(ours.values()) == 0, f"I9 is not a superset of ours on {x.cid}"
            add = [e for e in extra if ctx.R10.r1(x, e, match=True)]
            stat.setdefault("i9_extra", 0); stat["i9_extra"] += len(extra)
            ev = list(be) + add
        else:
            raise KeyError(cell)
        stat["added"] += len(ev) - len(be)
        out.append(ev)
    return out, stat


NEW_CELLS = ["N1-all", "N1", "N2-all", "N2", "N3", "N4"]


# ============================================================================= steps 2-4
def score_cells(ctx, cells, sc, brows, log_key, log, extra_old=True):
    res = log.setdefault(log_key, {})
    for name, fn in cells:
        evs = fn()
        stat = None
        if isinstance(evs, tuple):
            evs, stat = evs
        rows = sc.rows(evs)
        S = sc.summ(rows)
        S.update(compare(rows, brows))
        S["same_as_ours_every_clip"] = all(M.sig(a) == M.sig(b) for a, b in zip(evs, ctx.base_evs))
        if extra_old:
            ro = [D.clip_cost(x.c, ev) for x, ev in zip(ctx.xs, evs)]
            S["old_C_overlap"] = float(np.mean([r["C_overlap"] for r in ro])); S["old_C_onset"] = float(np.mean([r["C_onset"] for r in ro]))
        if stat:
            S["stat"] = stat
        res[name] = S
        show(name, S)
        save_log(log)
    return res


def base_rows(ctx, sc, log_key, log):
    brows = sc.rows(ctx.base_evs)
    Sb = sc.summ(brows)
    nothing = sc.rows([[] for _ in ctx.xs])
    Sn = sc.summ(nothing)
    Sn.update(compare(nothing, brows))
    res = log.setdefault(log_key, {})
    res["ours"] = Sb; res["nothing"] = Sn
    show("ours", Sb); show("nothing", Sn)
    save_log(log)
    return brows


def rescore(log):
    assert log.get("step1", {}).get("ours_beats_nothing"), "step 1 did not pass: STOP"
    ctx = Ctx("calib")
    sc = Scorer(ctx.xs)
    brows = base_rows(ctx, sc, "fit", log)
    cells = [(c, (lambda c=c, r=r: BUILD[r](ctx, c))) for c, r in OLD_CELLS]
    res = score_cells(ctx, cells, sc, brows, "fit", log)
    for c, r in OLD_CELLS:
        pub = published_old(c, r)
        res[c]["round"] = r
        res[c]["published_old_C"] = pub
        res[c]["old_C_reproduced"] = (None if pub is None else bool(abs(pub[0] - res[c]["old_C_overlap"]) < 1e-9
                                                                   and abs(pub[1] - res[c]["old_C_onset"]) < 1e-9))
    save_log(log)
    bad = [c for c, _r in OLD_CELLS if res[c]["old_C_reproduced"] is False]
    print(f"[rescore] old C reproduced for every cell with a published number: {not bad}; differ: {bad}", flush=True)


def new(log):
    assert log.get("step1", {}).get("ours_beats_nothing"), "step 1 did not pass: STOP"
    ctx = Ctx("calib")
    sc = Scorer(ctx.xs)
    brows = base_rows(ctx, sc, "fit", log)
    cells = [(c, (lambda c=c: b_new(ctx, c))) for c in NEW_CELLS]
    res = score_cells(ctx, cells, sc, brows, "fit", log)
    for c in NEW_CELLS:
        res[c]["round"] = 12
    save_log(log)


def picks(log):
    F = log["fit"]
    Sb = F["ours"]
    order = [c for c, _r in OLD_CELLS] + NEW_CELLS
    rnd = {c: F[c]["round"] for c in order if c in F}
    elig = [c for c in order if c in F and c not in REFERENCE and not F[c]["same_as_ours_every_clip"]
            and F[c]["C_overlap"] < Sb["C_overlap"] and F[c]["C_onset"] < Sb["C_onset"]]
    pk = sorted(elig, key=lambda c: (F[c]["C_overlap"], F[c]["C_onset"], rnd[c]))[:3]
    log["picks"] = {"eligible": elig, "picks": pk}
    print(f"[picks] eligible {elig}; picks {pk}", flush=True)
    save_log(log)


def builder(ctx, c):
    rn = dict(OLD_CELLS).get(c)
    if rn is not None:
        return lambda: BUILD[rn](ctx, c)
    return lambda: b_new(ctx, c)


def heldout(log):
    pk = log["picks"]["picks"]
    if not pk:
        log["heldout"] = "no pick"; save_log(log); print("no pick"); return
    ctx = Ctx("heldout")
    sc = Scorer(ctx.xs)
    brows = base_rows(ctx, sc, "heldout", log)
    H = score_cells(ctx, [(c, builder(ctx, c)) for c in pk], sc, brows, "heldout", log)
    idx = {s: [i for i, x in enumerate(ctx.xs) if x.c.get("stratum") == s] for s in ("complex", "random")}
    for c in pk:
        H[c]["pass"] = bool(H[c]["dC_overlap"][2] < 0)
        evs = builder(ctx, c)()
        evs = evs[0] if isinstance(evs, tuple) else evs
        d = [a["C_overlap"] - b["C_overlap"] for a, b in zip(sc.rows(evs), brows)]
        H[c]["strata"] = {s: {"n": len(ix), "dC_overlap": M.boot8([d[i] for i in ix]) if ix else None} for s, ix in idx.items()}
    H["holm"] = M.holm({c: H[c]["dC_overlap"][3] for c in pk})
    for c in pk:
        H[c]["pass_and_holm"] = bool(H[c]["pass"] and H["holm"][c]["reject"])
        print(f"[415] {c}: pass {H[c]['pass']} holm {H['holm'][c]} strata {H[c]['strata']}", flush=True)
    save_log(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("step1", "rescore", "new", "picks", "heldout"))
    a = ap.parse_args()
    log = load_log()
    if a.step == "step1":
        ok = step1(log)
        if not ok:
            sys.exit("STOP: ours does not beat 'show nothing' under v2 on the 280")
        return
    {"rescore": rescore, "new": new, "picks": picks, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
