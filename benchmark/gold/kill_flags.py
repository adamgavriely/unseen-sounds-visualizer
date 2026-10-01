"""Which SHIP8 stage-4 step removes the 6 strongly heard DEV misses (miss_ears.json bucket B, "above stage-4 bar ... no row").
CPU only, caches only: no model is run, nothing is written into the shared stage4.json caches.

For each DEV part (old DEV "dev", tagger DEV2 "dev2" via tagger_prep.configure) and every clip, stage-4 rows of system
"proposed" are rebuilt in memory exactly as round13_dev.stage4 does per clip (build -> reloc_rows -> filter_rows ->
add_breaks), for:
  SHIP8, B0r, SHIP8 with ONE flag that differs from B0r reverted to its B0r value (each flag in turn), SHIP8 with a
  logical group of flags reverted (the SHIP lineage steps), and SHIP8 with a shared (B0r too) step loosened
  (minimum span, impulse span, FlexSED / PANNs vetoes, FlexSED bar).
CAM onset refinement (GPU) is not run: a "live" row takes the refined start of an identical (label, pre_start, end) row in
the cached stage4.json of that part (any arm), else keeps its unrefined start (flagged "live-unrefined").
Checks: rebuilt SHIP8 and B0r rows equal the cached SHIP8|proposed / B0r|proposed rows (label, pre_start, end, conf,
origin, rescued) on every clip.
Per sound: the rows of the same family (score_per_sound.same_family) with start in [onset - 0.5, onset + 1.0] per variant.
Per variant: rows with conf >= DISPLAY_THRESHOLD not in SHIP8 (extra, the wrong-picture risk) and SHIP8 rows lost, over all
71 DEV clips.

    python benchmark/gold/kill_flags.py            # from ~/MscProj_tg: both parts, combos, report (subprocesses)
    python benchmark/gold/kill_flags.py combos     # only the multi-step chains (COMBOS) on the 6 sound clips
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TG_ARMS", "SHIP8")
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
G = _ROOT / "benchmark" / "gold"
OUT = G / "kill_flags.json"
MD = _ROOT / "docs" / "review" / "kill_flags_ship8.md"
TMP = _ROOT / "data" / "work" / "kill_flags_tmp"
SYS = "proposed"
TOL = 0.05

# logical groups of the SHIP lineage (round13_dev.ARMS), each reverted together
GROUPS = {
    "R13-1 twin max": ["TWIN_MAX"],
    "F7 mirror veto + listener keep": ["MIRROR_VETO", "LISTENER_CONFIRMED_MIRROR", "LISTENER_CACHE"],
    "listener rescue (TIER, ONCE, F8)": ["LISTENER_RESCUE", "LISTENER_RULE", "LISTENER_VCACHE", "LISTENER_AFCACHE",
                                         "LISTENER_LO", "LISTENER_ONCE", "LISTENER_DASM_VOTE", "LISTENER_DASM_DIR"],
    "N2b masked weak veto": ["MASKED_WEAK_VETO", "MASKED_WEAK_AF"],
    "DR2 DASM rescue": ["DASM_RESCUE", "DASM_RESCUE_NEW_ONLY", "DASM_P4_CACHE"],
    "KV4 keep needs V4": ["KEEP_NEEDS_V4", "RELABEL_P1V4"],
    "DV DASM clip veto": ["DASM_CLIP_VETO"],
    "BTP band twin pull": ["BAND_TWIN_PULL"],
    "CONT continuation veto": ["CONTINUATION_VETO"],
    "FLAP FineLAP veto": ["FINELAP_VETO", "FINELAP_DIR"],
    "K4A-D keep needs V4 all": ["KEEP_NEEDS_V4_ALL", "KEEP_NEEDS_V4_ALL_DASM_KEEP"],
    "all vetoes (F7 mirror, N2b, DV, K4A-D, CONT, FLAP)": ["MIRROR_VETO", "MASKED_WEAK_VETO", "DASM_CLIP_VETO",
                                                           "KEEP_NEEDS_V4_ALL", "CONTINUATION_VETO", "FINELAP_VETO"],
}
# steps SHIP8 shares with B0r, loosened on SHIP8 (diagnostic only)
SHARED = {
    "AED_MIN_DUR 0.5->0.1": {"AED_MIN_DUR": 0.1},
    "AED_MIN_DUR 0.5->0.3": {"AED_MIN_DUR": 0.3},
    "IMPULSE_MIN_SPAN 0.2": {"IMPULSE_MIN_SPAN": 0.2},
    "FLEXSED_VETO 0.3->0": {"FLEXSED_VETO": 0.0},
    "PANNS_VETO 0.05->0": {"PANNS_VETO": 0.0},
    "FLEXSED_BAR 0.8->0.75": {"FLEXSED_BAR": 0.75},
    "AED_THRESHOLD 0.175->0.1": {"AED_THRESHOLD": 0.1},
}


# multi-step chains on the 6 sound clips only: each chain adds one more step (found from build's TRACE and its
# [stage4] log lines) until the row comes back
COMBOS = {
    "as_explosion_XJ8lc3I6": [{"AED_MIN_DUR": 0.1}, {"AED_MIN_DUR": 0.1, "MASKED_WEAK_VETO": False},
                              {"AED_MIN_DUR": 0.1, "MASKED_WEAK_VETO": False, "DASM_CLIP_VETO": None}],
    "b3_favela_rio": [{"CONTINUATION_VETO": None}],
    "tg_d029": [{"MIRROR_VETO": None}, {"FLEXSED_VETO": 0.0}, {"MIRROR_VETO": None, "FLEXSED_VETO": 0.0}],
    "tg_d095": [{"MIRROR_VETO": None}, {"MIRROR_VETO": None, "MASKED_WEAK_VETO": False},
                {"MIRROR_VETO": None, "MASKED_WEAK_VETO": False, "KEEP_NEEDS_V4_ALL": None}],
    "tg_d107": [{"AED_MIN_DUR": 0.3}],
    "tg_d125": [{"FLEXSED_BAR": 0.7}, {"FLEXSED_BAR": 0.7, "AED_MIN_DUR": 0.1},
                {"FLEXSED_BAR": 0.7, "AED_MIN_DUR": 0.05}, {"FLEXSED_BAR": 0.7, "AED_MIN_DUR": 0.05, "PANNS_VETO": 0.0}],
}
KILLER = {   # read off the traces and the COMBOS chains
    "as_explosion_XJ8lc3I6": "minimum span AED_MIN_DUR 0.5 (shared with B0r: BEATs span 6.50-6.75 = 0.25 s; FlexSED 0.758 "
                             "< bar 0.8); behind it N2 masked weak veto, then DASM clip veto (SHIP8 only)",
    "b3_favela_rio": "CONT continuation veto (SHIP8 only)",
    "tg_d029": "mirror veto MIRROR_VETO 0.7 (SHIP8 only); behind it the cross-detector veto FLEXSED_VETO 0.3 (shared: "
               "FlexSED never reaches 0.3 for the bird family, so B0r drops it too)",
    "tg_d095": "three SHIP8-only vetoes, each enough alone: mirror veto, N2 masked weak veto, K4A-D (KEEP_NEEDS_V4_ALL); "
               "mirror+N2 off gives only Cutlery 0.268 (< display 0.35)",
    "tg_d107": "minimum span AED_MIN_DUR 0.5 (shared: BEATs 9.00-9.25 = 0.25 s, FlexSED 8.92-9.40 = 0.48 s)",
    "tg_d125": "FlexSED bar 0.8 + minimum span + PANNs veto on FlexSED-only spans (all shared with B0r): clap frames "
               "0.74-0.80, pieces 0.08 s at bar 0.7",
}


def jdump(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")


def sounds():
    m = json.loads((G / "miss_ears.json").read_text(encoding="utf-8"))["misses"]
    out = [x for x in m if x["bucket"] == "B" and str(x.get("b_reason", "")).startswith("above stage-4 bar")]
    assert len(out) == 6, len(out)
    return [{k: x[k] for k in ("part", "clip", "sound", "family", "at", "end")} for x in out]


# ============================================================================= one part (subprocess)
def run_part(part):
    import config
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import round13_dev as R
    from benchmark.gold import score_per_sound as S
    from src.labels import canonical
    if part == "dev2":
        from benchmark.gold import tagger_prep as TP
        DCC, R, stems = TP.configure("dev2")
    else:
        _g, stems = DCC.dev_stems()
    stems = list(stems)
    ship, b0 = R.arm_cfg("SHIP8"), R.arm_cfg("B0r")
    default = lambda k: b0[k] if k in b0 else getattr(config, k, None)
    diff = sorted(k for k in set(ship) | set(b0) if ship.get(k, getattr(config, k, None)) != default(k))
    variants = {"SHIP8": None, "B0r": None}
    for k in diff:
        variants[f"-{k}"] = {k: default(k)}
    for g, ks in GROUPS.items():
        variants[f"-[{g}]"] = {k: default(k) for k in ks}
    for g, o in SHARED.items():
        variants[f"+[{g}]"] = dict(o)
    for v, o in variants.items():
        if o is not None:
            R.ARMS[f"KF{v}"] = {**R.ARMS["SHIP8"], **o}
    arm_of = lambda v: v if v in ("SHIP8", "B0r") else f"KF{v}"
    cache = json.loads(R.STAGE4.read_text(encoding="utf-8"))["arms"]
    refined = {}                                           # (clip, label, pre_start, end) -> refined start, any arm
    for k, by in cache.items():
        if not k.endswith(f"|{SYS}"):
            continue
        for st, rr in by.items():
            for r in rr:
                if r.get("refine") in ("trace", "live"):
                    refined.setdefault((st, r["label"], round(r["pre_start"], 3), round(r["end"], 3)), r["start"])
    snd = [s for s in sounds() if s["part"] == part]
    sclips = {s["clip"] for s in snd}
    res = {"part": part, "stems": stems, "diff": {k: [default(k), ship.get(k)] for k in diff},
           "variants": {v: (o if o is None else {k: str(x) for k, x in o.items()}) for v, o in variants.items()},
           "rows": {}, "errors": {}, "trace": {}, "listener": {}, "filtered": {}, "raw": {}, "check": {}}
    for st in stems:
        C = {"beats": DCC.load_fr(DCC.BEATS_DIR / f"{st}.npz"), "flex": DCC.load_fr(DCC.FLEX_DIR / f"{st}.npz")}
        tr = json.loads((DCC.scored_dir(SYS) / st / "onset_trace.json").read_text(encoding="utf-8"))
        for v in variants:
            arm = arm_of(v)
            try:
                rows, info = R.build(st, SYS, arm, C, tr, offline=False)
                trc = [dict(x) for x in R.TRACE]
                C2 = {**C, "flex": info["flex"]}
                scratch = {}
                rows = R.reloc_rows(rows, arm, C2, info, scratch, v, st)
                rows = R.filter_rows(rows, arm, C2, info["ffw"], scratch, v, st)
                R.add_breaks(rows, arm, C2, info["ffw"])
            except Exception as e:                          # noqa: BLE001
                res["errors"].setdefault(v, {})[st] = f"{type(e).__name__}: {e}"
                if len(res["errors"][v]) == 1:
                    traceback.print_exc()
                continue
            for r in rows:
                if r["refine"] == "live":
                    s0 = refined.get((st, r["label"], round(r["pre_start"], 3), round(r["end"], 3)))
                    if s0 is not None:
                        r["start"], r["refine"] = float(s0), "live->cached"
                    else:
                        r["refine"] = "live-unrefined"
            res["rows"].setdefault(v, {})[st] = [{k: r[k] for k in ("label", "start", "pre_start", "end", "conf", "origin",
                                                                    "rescued", "refine")} for r in rows]
            if st in sclips:
                fams = [s["sound"] for s in snd if s["clip"] == st]
                sf = lambda l: any(S.same_family(l, f) for f in fams)
                res["trace"].setdefault(v, {})[st] = [x for x in trc if sf(x["label"])]
                if "listener" in info:
                    L = info["listener"]
                    res["listener"].setdefault(v, {})[st] = {k: [x for x in L.get(k, []) if sf(x[0])] for k in
                                                             ("a_added", "a_missing_list", "b_kept", "b_missing_list", "f7")
                                                             if isinstance(L.get(k), list)}
                    res["listener"][v][st]["f7"] = [x for x in info.get("f7", []) if sf(x[0])]
                drop = scratch.get("r14_dropped", {}).get(f"{v}|{st}", {})
                res["filtered"].setdefault(v, {})[st] = {k: [x for x in xs if sf(x[0])] for k, xs in drop.items()
                                                         if any(sf(x[0]) for x in xs)}
        # check: rebuilt SHIP8 / B0r == cached (label, pre_start, end, conf, origin, rescued)
        sig = lambda rr: sorted((r["label"], round(r["pre_start"], 3), round(r["end"], 3), round(r["conf"], 4), r["origin"],
                                 bool(r.get("rescued", False))) for r in rr)
        for a in ("SHIP8", "B0r"):
            mine = res["rows"].get(a, {}).get(st)
            ref = cache.get(f"{a}|{SYS}", {}).get(st)
            res["check"][f"{a}|{st}"] = (mine is not None and ref is not None and sig(mine) == sig(ref))
            ref_s = sorted(round(r["start"], 3) for r in ref or [])
            res["check"][f"{a}|{st}|start"] = mine is not None and sorted(round(r["start"], 3) for r in mine) == ref_s
        # raw spans without the minimum span (shared step diagnostic), sound clips only
        if st in sclips:
            with R.flags(ship):
                for s in [s for s in snd if s["clip"] == st]:
                    sf = lambda l: S.same_family(l, s["sound"])
                    out = {}
                    for nm, fr, bar in (("beats", C["beats"], float(ship["AED_THRESHOLD"])),
                                        ("flex", C["flex"], float(ship["FLEXSED_BAR"])), ("flex_band0.5", C["flex"], 0.5)):
                        for md in (float(ship["AED_MIN_DUR"]), 0.0):
                            ev = R._extract_events(fr[0], fr[1], fr[2], bar, None, md - 1e-6 if md else 0.0,
                                                   low=bar * float(ship["AED_HYSTERESIS"]))
                            out[f"{nm}@{bar}|min{md}"] = [[e.label, round(e.start, 2), round(e.end, 2), round(e.confidence, 3)]
                                                          for e in ev if sf(e.label)]
                    res["raw"][f"{st}|{s['sound']}"] = out
        print(f"[{part}] {st} done", flush=True)
    jdump(TMP / f"{part}.json", res)
    print(f"[{part}] -> {TMP / (part + '.json')}", flush=True)


def run_combos():
    """COMBOS on the 6 sound clips, same chain as run_part (build -> reloc_rows -> filter_rows)"""
    from benchmark.gold import score_per_sound as S
    out = {}
    for part in ("dev", "dev2"):
        from benchmark.gold import dev_candidates_check as DCC
        from benchmark.gold import round13_dev as R
        if part == "dev2":
            from benchmark.gold import tagger_prep as TP
            DCC, R, _ = TP.configure("dev2")
        for s in [x for x in sounds() if x["part"] == part]:
            st = s["clip"]
            C = {"beats": DCC.load_fr(DCC.BEATS_DIR / f"{st}.npz"), "flex": DCC.load_fr(DCC.FLEX_DIR / f"{st}.npz")}
            tr = json.loads((DCC.scored_dir(SYS) / st / "onset_trace.json").read_text(encoding="utf-8"))
            w0, w1 = float(s["at"]) - S.EARLY, float(s["at"]) + S.LATE
            res = []
            for o in COMBOS[st]:
                R.ARMS["KFC"] = {**R.ARMS["SHIP8"], **o}
                rows, info = R.build(st, SYS, "KFC", C, tr, offline=False)
                C2 = {**C, "flex": info["flex"]}
                rows = R.reloc_rows(rows, "KFC", C2, info, {}, "KFC", st)
                rows = R.filter_rows(rows, "KFC", C2, info["ffw"], {}, "KFC", st)
                win = [[r["label"], round(r["start"], 2), round(r["end"], 2), round(r["conf"], 3), r["origin"], r["refine"]]
                       for r in rows if S.same_family(r["label"], s["sound"]) and w0 - 1e-6 <= r["start"] <= w1 + 1e-6]
                res.append({"override": {k: str(v) for k, v in o.items()}, "in_window": win})
            out[st] = res
    jdump(TMP / "combos.json", out)
    print("->", TMP / "combos.json")


# ============================================================================= report (both parts)
def report():
    import config  # noqa: F401
    from benchmark.gold import round13_dev as R
    from benchmark.gold import score_per_sound as S
    from src.labels import is_salient_nonspeech
    P = {p: json.loads((TMP / f"{p}.json").read_text(encoding="utf-8")) for p in ("dev", "dev2")}
    CB = json.loads((TMP / "combos.json").read_text(encoding="utf-8")) if (TMP / "combos.json").exists() else {}
    disp = float(R.arm_cfg("SHIP8")["DISPLAY_THRESHOLD"])
    variants = [v for v in P["dev"]["variants"] if v.lstrip("-") not in R.DISPLAY_KEYS]   # display-only flags ignored
    assert list(P["dev"]["variants"]) == list(P["dev2"]["variants"]), "variant lists differ between parts"
    diff = P["dev"]["diff"]
    out = {"display_threshold": disp, "window": [-S.EARLY, S.LATE], "diff_ship8_vs_b0r": diff,
           "diff_dev2": P["dev2"]["diff"], "groups": GROUPS, "shared": SHARED, "errors": {}, "check": {}, "sounds": [],
           "variants": {}}
    for p in P:
        bad = [k for k, v in P[p]["check"].items() if not v]
        out["check"][p] = {"pass": sum(1 for v in P[p]["check"].values() if v), "total": len(P[p]["check"]), "fail": bad}
        for v, e in P[p]["errors"].items():
            out["errors"].setdefault(v, {}).update({f"{p}|{k}": x for k, x in e.items()})
    # per sound
    for s in sounds():
        pt, st, snd, at = s["part"], s["clip"], s["sound"], float(s["at"])
        w0, w1 = at - S.EARLY, at + S.LATE
        sf = lambda l: S.same_family(l, snd)
        per = {}
        for v in variants:
            rr = P[pt]["rows"].get(v, {}).get(st)
            if rr is None:
                per[v] = "error"; continue
            fam = [r for r in rr if sf(r["label"])]
            win = [r for r in fam if w0 - 1e-6 <= r["start"] <= w1 + 1e-6]
            pre = [r for r in fam if r["refine"] == "live-unrefined" and w0 - 1e-6 <= r["pre_start"] <= w1 + 1e-6]
            per[v] = {"in_window": [[r["label"], round(r["start"], 2), round(r["end"], 2), round(r["conf"], 3), r["origin"],
                                     r["rescued"], r["refine"]] for r in win],
                      "unrefined_pre_start_in_window": [[r["label"], round(r["pre_start"], 2), round(r["conf"], 3)] for r in pre],
                      "same_family_elsewhere": [[r["label"], round(r["start"], 2), round(r["end"], 2), round(r["conf"], 3)]
                                                for r in fam if r not in win]}
        restored = [v for v in variants if v != "SHIP8" and isinstance(per[v], dict) and per[v]["in_window"]]
        out["sounds"].append({**s, "window": [w0, w1], "killer": KILLER[st], "combos": CB.get(st), "ship8_has": bool(per["SHIP8"]["in_window"]),
                              "b0r_has": bool(isinstance(per["B0r"], dict) and per["B0r"]["in_window"]),
                              "restored_by": {v: per[v]["in_window"] for v in restored}, "per_variant": per,
                              "trace_ship8": P[pt]["trace"].get("SHIP8", {}).get(st),
                              "trace_b0r": P[pt]["trace"].get("B0r", {}).get(st),
                              "listener_ship8": P[pt]["listener"].get("SHIP8", {}).get(st),
                              "filtered_ship8": P[pt]["filtered"].get("SHIP8", {}).get(st),
                              "raw_spans": P[pt]["raw"].get(f"{st}|{snd}")})
    # per variant: extra / lost displayable rows vs SHIP8 over all DEV clips
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        for v in variants:
            ex, lo, n_err = [], [], 0
            for p in P:
                for st in P[p]["stems"]:
                    a, b = P[p]["rows"].get(v, {}).get(st), P[p]["rows"]["SHIP8"].get(st)
                    if a is None or b is None:
                        n_err += 1; continue
                    same = lambda r, o: (r["label"] == o["label"] and abs(r["pre_start"] - o["pre_start"]) <= TOL
                                         and abs(r["end"] - o["end"]) <= TOL and o["conf"] >= disp)
                    for r in a:
                        if r["conf"] >= disp and not any(same(r, o) for o in b):
                            ex.append([p, st, r["label"], round(r["start"], 2), round(r["conf"], 3), bool(is_salient_nonspeech(r["label"]))])
                    for o in b:
                        if o["conf"] >= disp and not any(same(o, r) for r in a):
                            lo.append([p, st, o["label"], round(o["start"], 2), round(o["conf"], 3), bool(is_salient_nonspeech(o["label"]))])
            out["variants"][v] = {"override": P["dev"]["variants"][v], "clips_error": n_err,
                                  "extra_rows": len(ex), "extra_rows_depictable": sum(1 for x in ex if x[5]),
                                  "lost_rows": len(lo), "lost_rows_depictable": sum(1 for x in lo if x[5]),
                                  "extra": ex, "lost": lo}
    finally:
        config.LABEL_FILTER = old
    jdump(OUT, out)
    print("->", OUT)
    md(out)


def md(out):
    L = ["# Which SHIP8 stage-4 step removes 6 strongly heard DEV misses (CPU, caches only)", "",
         "Script `benchmark/gold/kill_flags.py`, data `benchmark/gold/kill_flags.json`. Stage-4 rows of SHIP8|proposed rebuilt "
         "in memory (build, filter_rows) per variant; a sound is *restored* when a same-family row starts in "
         "[onset - 0.5, onset + 1.0]. Extra rows = rows with conf >= display threshold "
         f"({out['display_threshold']}) not in SHIP8, over all 71 DEV clips (both parts; depictable labels in brackets).", "",
         f"Rebuild check (rebuilt == cached rows, label/pre_start/end/conf/origin/rescued, and start): "
         + "; ".join(f"{p} {c['pass']}/{c['total']}" for p, c in out["check"].items()), ""]
    if out["errors"]:
        L += ["Variants that crashed (flag reverted alone leaves an incoherent config): "
              + ", ".join(f"`{v}` ({len(e)} clips)" for v, e in out["errors"].items()), ""]
    L += ["| sound | in B0r? | killer step (SHIP8) | variants that restore it (row conf) | smallest chain that restores it "
          "(row label start conf) |", "|---|---|---|---|---|"]
    for s in out["sounds"]:
        rb = "; ".join(f"`{v}` ({', '.join(str(r[3]) for r in rows)})" for v, rows in s["restored_by"].items()) or "none"
        ok = [c for c in (s["combos"] or []) if c["in_window"]]
        ch = "none found"
        if ok:
            c = ok[-1]
            ch = (", ".join(f"{k}={v}" for k, v in c["override"].items()) + " ("
                  + "; ".join(f"{r[0]} {r[1]} {r[3]}" + (" unrefined" if str(r[5]).startswith("live") else "")
                              for r in c["in_window"]) + ")")
        L.append(f"| {s['part']} {s['clip']} {s['sound']} @{s['at']} | {'yes' if s['b0r_has'] else 'no'} | {s['killer']} "
                 f"| {rb} | {ch} |")
    rest = sorted({v for s in out["sounds"] for v in s["restored_by"]})
    L += ["", "| variant (restores >= 1) | extra rows (depictable) | lost SHIP8 rows (depictable) |", "|---|---|---|"]
    for v in rest:
        x = out["variants"][v]
        L.append(f"| `{v}` | {x['extra_rows']} ({x['extra_rows_depictable']}) | {x['lost_rows']} ({x['lost_rows_depictable']}) |")
    L += ["", "Variant names: `-FLAG` = SHIP8 with that flag at its B0r value; `-[group]` = a group reverted; "
          "`+[step]` = a step SHIP8 shares with B0r loosened. Per-sound trace / listener / filter detail in the JSON.", "",
          "Not determined (needs GPU, skipped): CAM onset refinement of new BEATs rows; a restored row with no cached "
          "refined start keeps its unrefined start (marked 'unrefined'). Listener caches were not refilled for spans a "
          "variant creates. `-LISTENER_VCACHE` alone crashes (LISTENER_RULE=TIER needs it). MERGE_GAP, PICTURE_MIN_CONF, "
          "GROUP_* (display-only) ignored."]
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", MD)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("dev", "dev2"):
        run_part(sys.argv[1])
    elif len(sys.argv) > 1 and sys.argv[1] == "combos":
        run_combos()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for p in ("dev", "dev2"):
            subprocess.run([sys.executable, __file__, p], check=True)
        subprocess.run([sys.executable, __file__, "combos"], check=True)
        report()
