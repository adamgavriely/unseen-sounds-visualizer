"""Round 55 FRAME-PAUSE (docs/prereg_round13_detector_push.md "Round 55 FRAME-PAUSE", written before any number of it).

The pause between two same-family pictures is measured on the FlexSED frames (the family's query max, 25 fps): hole = the longest
stretch with max < f in [first picture end - 1.0 s, second picture start]. A GRP-A same/same pair merges iff hole < 2.0 s (the
gold's own split rule). f is calibrated on the held-out 415 strong labels (step 1); the merged DEV is scored through the shipped
display path only if step 1 passes (step 2). Nothing in src/ or config.py is edited: the rule is applied by wrapping
group._answers at scoring time, exactly as grpp_screen.setup_dev does for Round 51.

    python benchmark/gold/framepause_screen.py calib415                 (CPU)  -> framepause/calib_415.json ; exit 3 below the bar
    TG_ARMS="SHIP8+MD3" python benchmark/gold/framepause_screen.py scoredev   (CPU)  -> framepause/score_dev.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

T, PRE, MAX_GAP, BAR, MIN_SEG = 2.0, 1.0, 8.0, 0.75, 3.0
F_GRID = [round(0.05 * i, 2) for i in range(1, 17)]            # 0.05 ... 0.80
L_GRID = [2.0 + 0.5 * i for i in range(13)]                     # 2.0 ... 8.0
DIR = _ROOT / "benchmark" / "gold" / "framepause"
GRPP = _ROOT / "benchmark" / "gold" / "grpp"
HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
FLEX_HELD = Path(os.environ.get("FLEX_HELD", str(_ROOT / "data" / "work" / "flexsed_heldout")))
ARM = "SHIP8+MD3"


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================ frames
def fam_cols(labs, lab):
    from src.labels import canonical
    c = canonical(lab)
    return [i for i, l in enumerate(labs) if canonical(l) == c]


def series(fr, cols, w0, w1):
    """the family's frame max inside [w0, w1] and the frame step; None when the family has no query"""
    if not cols:
        return None, None
    fw, t, _ = fr
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.04
    sel = (t >= w0 - 1e-9) & (t <= w1 + 1e-9)
    return fw[sel][:, cols].max(axis=1) if sel.any() else np.zeros(0, np.float32), dt


def longest_hole(m, dt, f):
    """longest contiguous stretch (s) with max < f"""
    best = run = 0
    for v in (m < f):
        run = run + 1 if v else 0
        best = max(best, run)
    return best * dt


# ============================================================================ step 1: held-out 415
def _joined_segments(events):
    by = {}
    for e in events:
        by.setdefault(e["label"], []).append(e)
    out = {}
    for lab, ev in by.items():
        ev = sorted(ev, key=lambda e: (float(e["start"]), float(e["end"])))
        segs = []
        for e in ev:
            a, b, m = float(e["start"]), float(e["end"]), bool(e.get("masked"))
            if segs and a <= segs[-1][1]:
                segs[-1][1] = max(segs[-1][1], b); segs[-1][2] = segs[-1][2] or m
            else:
                segs.append([a, b, m])
        out[lab] = segs
    return out


def cmd_calib415():
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import SPEECH_LABELS, is_music
    H = _load(HELDOUT)
    assert len(H["clips"]) == 415
    pairs = [p for p in _load(GRPP / "pairs_415.json")["pairs"] if p["gap"] > T and not p["speech_music"]]
    frs = {}

    def fr_of(c):
        if c not in frs:
            frs[c] = DCC.load_fr(FLEX_HELD / f"{c}.npz")
        return frs[c]

    items = []                                                  # side, clip, lab, window, masked, series
    noq = {"P": 0, "C": 0}
    for p in pairs:
        fr = fr_of(p["clip"])
        m, dt = series(fr, fam_cols(fr[2], p["lab"]), p["a"][1] - PRE, p["b"][0])
        if m is None:
            noq["P"] += 1; continue
        items.append({"side": "P", "clip": p["clip"], "lab": p["lab"], "w": [p["a"][1] - PRE, p["b"][0]], "gap": p["gap"],
                      "masked": bool(p["masked"]), "m": m, "dt": dt})
    nseg = 0
    for c in H["clips"]:
        for lab, segs in _joined_segments(c["events"]).items():
            if lab in SPEECH_LABELS or is_music(lab):
                continue
            for a, b, msk in segs:
                if b - a < MIN_SEG:
                    continue
                nseg += 1
                fr = fr_of(c["id"])
                m, dt = series(fr, fam_cols(fr[2], lab), a, b)
                if m is None:
                    noq["C"] += 1; continue
                items.append({"side": "C", "clip": c["id"], "lab": lab, "w": [a, b], "gap": None, "masked": msk, "m": m, "dt": dt})
    P = [i for i in items if i["side"] == "P"]
    C = [i for i in items if i["side"] == "C"]
    res = {"what": "Round 55 FRAME-PAUSE step 1: FlexSED hole vs the strong-label 2-s truth on the held-out 415",
           "rule": "hole = longest stretch with family FlexSED max < f; predicted pause iff hole >= 2.0 s",
           "side_P": {"n": len(P), "no_query_excluded": noq["P"], "candidates": len(pairs), "window": "seg1 end - 1.0 ... seg2 start"},
           "side_C": {"n": len(C), "no_query_excluded": noq["C"], "segments": nseg, "window": "segment start ... end, >= 3.0 s, non-speech/music"},
           "bar": BAR, "grid": {}}
    best = None
    for f in F_GRID:
        hp = [longest_hole(i["m"], i["dt"], f) for i in P]
        hc = [longest_hole(i["m"], i["dt"], f) for i in C]
        accP = sum(h >= T for h in hp) / len(P) if P else 0.0
        accC = sum(h < T for h in hc) / len(C) if C else 0.0
        um_P = [h for i, h in zip(P, hp) if not i["masked"]]
        um_C = [h for i, h in zip(C, hc) if not i["masked"]]
        row = {"acc_P": round(accP, 3), "acc_C": round(accC, 3), "balanced": round((accP + accC) / 2, 3),
               "acc_P_unmasked": round(sum(h >= T for h in um_P) / len(um_P), 3) if um_P else None,
               "acc_C_unmasked": round(sum(h < T for h in um_C) / len(um_C), 3) if um_C else None,
               "acc_P_by_gap": {b: round(sum(h >= T for i, h in zip(P, hp) if lo < i["gap"] <= hi)
                                         / max(1, sum(1 for i in P if lo < i["gap"] <= hi)), 3)
                                for b, lo, hi in (("2-4", 2, 4), ("4-8", 4, 8))}}
        res["grid"][str(f)] = row
        if best is None or row["balanced"] > best[1] + 1e-12:         # ties -> the lower f (grid ascends)
            best = (f, row["balanced"])
    f_star = best[0]
    r = res["grid"][str(f_star)]
    res["f_star"], res["pass"] = f_star, bool(r["acc_P"] >= BAR and r["acc_C"] >= BAR)
    # Y* (reported): among all holes >= L at f*, the share inside a continuous segment
    hp = [longest_hole(i["m"], i["dt"], f_star) for i in P]
    hc = [longest_hole(i["m"], i["dt"], f_star) for i in C]
    ystar, ytab = None, {}
    for L in L_GRID:
        nP, nC = sum(h >= L for h in hp), sum(h >= L for h in hc)
        frac = nC / (nP + nC) if nP + nC else None
        ytab[str(L)] = {"holes_P": nP, "holes_C": nC, "dropout_share": None if frac is None else round(frac, 3)}
        if ystar is None and frac is not None and frac < 0.5:
            ystar = L
    res["y_star"], res["y_table"] = ystar, ytab
    _dump(DIR / "calib_415.json", res)
    print(f"side P {len(P)} (no query {noq['P']} of {len(pairs)}), side C {len(C)} (no query {noq['C']} of {nseg})", flush=True)
    for f, row in res["grid"].items():
        print(f"  f {f:5s} acc_P {row['acc_P']:.3f} acc_C {row['acc_C']:.3f} balanced {row['balanced']:.3f} by gap {row['acc_P_by_gap']}", flush=True)
    print(f"f* {f_star}: acc_P {r['acc_P']} acc_C {r['acc_C']} -> {'PASS' if res['pass'] else 'FAIL'} (bar {BAR} both); Y* {ystar} {ytab}", flush=True)
    sys.exit(0 if res["pass"] else 3)


# ============================================================================ step 2: merged DEV
CHAIN = {"on": False}


def apply_chain(spans, clip, G, config):
    """a copy of group.apply with the chain fix: after a absorbs b, the pair b->c is applied to the merged span (its answer is
    keyed by the absorbed span's start), to a fixed point. The shipped apply skips it because `a in out` fails for b."""
    if not getattr(config, "GROUP_ASK", False):
        return spans
    ans = G._answers(clip or getattr(config, "GROUP_CLIP", None))
    out = sorted(spans, key=lambda p: p[1])
    last = {id(s): s[1] for s in out}                           # span -> start of the last span it absorbed (own start at first)
    changed = True
    while changed:
        changed = False
        for a, b in G.pairs(out, float(getattr(config, "GROUP_MAX_GAP", 4.0))):
            rep = ans.get(G.key(a[0], last[id(a)]), [])
            if rep and all(str(x).lower().startswith("same") for x in rep) and b in out and a in out:
                a[2] = max(a[2], b[2])
                last[id(a)] = b[1]
                out.remove(b)
                changed = True
                break
    return out


def cmd_scoredev():
    import config
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import grpp_screen as GP
    from src.stage6_visual_augmentation import group as G
    calib = _load(DIR / "calib_415.json")
    assert calib["pass"], "step 1 did not pass; step 2 is not run"
    f = float(calib["f_star"])
    pairs = _load(GRPP / "pairs_dev.json")["pairs"]
    verdict, frs = {}, {}
    for p in pairs:
        c = p["clip"]
        if c not in frs:
            frs[c] = DCC.load_fr(Path(DCC.FLEX_DIR) / f"{c}.npz")
        m, dt = series(frs[c], fam_cols(frs[c][2], p["lab"]), p["a"][1] - PRE, p["b"][0])
        h = None if m is None else longest_hole(m, dt, f)
        verdict[f"{c}|{p['key']}"] = {"hole": h, "pause": (h is not None and h >= T), "gap": p["gap"], "GRP-A": p["group_answers"]}
    orig_ans, orig_apply = G._answers, G.apply

    def answers(clip):
        ans = orig_ans(clip)
        if GP.MODE["m"] in ("fp", "fp_chain"):
            out = {}
            for k, v in ans.items():
                vd = verdict.get(f"{clip}|{k}")
                out[k] = list(v) + (["pause:yes"] if vd and vd["pause"] else [])
            return out
        return ans

    def apply(spans, clip):
        return apply_chain(spans, clip, G, config) if GP.MODE["m"].endswith("_chain") else orig_apply(spans, clip)

    G._answers, G.apply = answers, apply                        # installed BEFORE setup_dev, which wraps them in turn
    M = GP.setup_dev()
    modes = ["base", "fp", "base_chain", "fp_chain"]
    sc = GP.score_modes(M, modes)
    res = {"what": "Round 55 FRAME-PAUSE step 2: merged DEV, SHIP8+MD3 (GRP-A shipped) vs FP; both with the chain fix",
           "f_star": f, "rows": {}, "parts": {}, "pairs": verdict}
    flat = {m: [(p, s, r) for p in ("dev", "dev2") for s, r in sc[m][p]] for m in sc}
    for m, rr in flat.items():
        res["rows"][m] = GP.row_of([r for _p, _s, r in rr])
        res["parts"][m] = {p: GP.row_of([r for s, r in sc[m][p]]) for p in ("dev", "dev2")}
    B, F = res["rows"]["base"], res["rows"]["fp"]
    hb = {(p, s): r["hit"] for p, s, r in flat["base"]}
    hf = {(p, s): r["hit"] for p, s, r in flat["fp"]}
    lost = {f"{p}:{s}": hb[(p, s)] - hf[(p, s)] for (p, s) in hb if hf[(p, s)] < hb[(p, s)]}
    gained = {f"{p}:{s}": hf[(p, s)] - hb[(p, s)] for (p, s) in hb if hf[(p, s)] > hb[(p, s)]}
    cb = [DCC.clip_cost(r) for _p, _s, r in flat["base"]]
    cf = [DCC.clip_cost(r) for _p, _s, r in flat["fp"]]
    res["d_fp_vs_base"] = DCC.boot(np.subtract(cf, cb))
    res["hits_lost_clips"], res["hits_gained_clips"] = lost, gained
    tie = all(B[k] == F[k] for k in ("hits", "wrong", "visible", "cross", "phantom")) and abs(B["viewer_cost"] - F["viewer_cost"]) < 1e-9
    n_lost = sum(lost.values())
    main = F["hits"] >= 29 and not lost and F["viewer_cost"] < B["viewer_cost"] - 1e-12
    fewer = F["viewer_cost"] < B["viewer_cost"] - 1e-12 and (B["wrong"] - F["wrong"]) >= 3 * n_lost and n_lost <= 3
    res["rule"] = {"base_reproduced": (B["hits"], B["wrong"], B["visible"], B["cross"], B["phantom"]) == (29, 18, 6, 10, 2),
                   "tie": tie, "main_rule": main, "fewer_pictures_clause": fewer,
                   "verdict": "GO (main rule)" if main else "GO (fewer-pictures clause)" if fewer
                   else "TIE -> FP by the pre-registered tie-break" if tie else "STOP (GRP-A stays)"}
    res["chain_fix_changes_dev"] = {m: res["rows"][m] != res["rows"][m + "_chain"] for m in ("base", "fp")}
    _dump(DIR / "score_dev.json", res)
    for m in modes:
        print(f"merged DEV {m:10s} {GP.fmt(res['rows'][m])}  parts dev {res['parts'][m]['dev']['viewer_cost']:.3f} "
              f"dev2 {res['parts'][m]['dev2']['viewer_cost']:.3f}", flush=True)
    for k, v in verdict.items():
        print("  pair", k, v, flush=True)
    print("d fp-base", res["d_fp_vs_base"], "lost", lost, "gained", gained, flush=True)
    print("chain fix changes DEV:", res["chain_fix_changes_dev"], flush=True)
    print("VERDICT", res["rule"], flush=True)


def cmd_chaindev():
    """step 4 (independent of step 1): the shipped GRP-A base vs the same through the chain-fixed apply copy; no FP rule"""
    import config
    from benchmark.gold import grpp_screen as GP
    from src.stage6_visual_augmentation import group as G
    orig_apply = G.apply
    G.apply = lambda spans, clip: apply_chain(spans, clip, G, config) if GP.MODE["m"].endswith("_chain") else orig_apply(spans, clip)
    M = GP.setup_dev()
    sc = GP.score_modes(M, ["base", "base_chain"])
    res = {"what": "Round 55 step 4: shipped GRP-A vs the chain-fixed apply copy on merged DEV", "rows": {}, "parts": {}}
    for m in sc:
        res["rows"][m] = GP.row_of([r for p in ("dev", "dev2") for _s, r in sc[m][p]])
        res["parts"][m] = {p: GP.row_of([r for _s, r in sc[m][p]]) for p in ("dev", "dev2")}
    res["chain_fix_changes_dev"] = res["rows"]["base"] != res["rows"]["base_chain"]
    res["base_reproduced"] = tuple(res["rows"]["base"][k] for k in ("hits", "wrong", "visible", "cross", "phantom")) == (29, 18, 6, 10, 2)
    _dump(DIR / "chain_dev.json", res)
    for m in sc:
        print(f"merged DEV {m:10s} {GP.fmt(res['rows'][m])}", flush=True)
    print("base reproduced:", res["base_reproduced"], "| chain fix changes DEV:", res["chain_fix_changes_dev"], flush=True)


if __name__ == "__main__":
    {"calib415": cmd_calib415, "scoredev": cmd_scoredev, "chaindev": cmd_chaindev}[sys.argv[1]]()
