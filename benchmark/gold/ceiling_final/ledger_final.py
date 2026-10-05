"""CEILING_FINAL copy of ship7_errors.py (ledger half only; SPOT-V / K4A simulations dropped) for the FINAL arm
SHIP8+MD3+WW5+SL (D') on merged DEV. Writes only into benchmark/gold/ceiling_final/. Changes vs the original:
CHAIN extended to the final arm; every SHIP6+FLAP literal -> ARM; floor 0.40 -> 0.35 (final arm: PICTURE_MIN_CONF None,
DISPLAY / AUGMENT threshold 0.35); placed() passes clip= (GROUP / DEPICT display rules, as load_pictures); the dev2 gold
temp file is written here; new miss class "display-dropped" (augment spec in window, not placed).

    python benchmark/gold/ceiling_final/ledger_final.py      # from ~/MscProj_tg

Original docstring: Round 33 (docs/prereg_round13_detector_push.md "Round 33"): the SHIP7 error ledger and the SpotSound rescued-picture
veto (SPOT-V), CPU only, on the saved SHIP7 pictures of merged DEV (arm `SHIP6+FLAP|proposed`; rows = `SHIP7|proposed`).
Nothing in src/ or config.py is edited; no picture is changed on disk.

  ledger: every wrong picture and every missed needed sound with the stage that produced / removed it (fixed order, see the
          prereg entry), plus the evidence around it (listeners, DASM, FlexSED, BEATs, FineLAP-free) -> ship7_errors.json
  SPOT-V: drop a rescued placed picture whose matched P2/PV SpotSound EXIST answer starts with "no"; rescored, combined rule
          vs SHIP7. Reported beside: SPOT-no as round 31 (EXIST no OR no GROUND overlap).

    python benchmark/gold/ship7_errors.py            # from ~/MscProj_tg on the cluster (msproj)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TG_ARMS", "SHIP8+MD3+WW5+SL SHIP8+MD3+WW5 SHIP8+MD3+WW SHIP8+MD3 SHIP8 SHIP7+K4AD SHIP6+FLAP SHIP7 B0r "
                                  "TO1+F7F8 TO1F7F8+N2b SHIP+DR2 SHIP2+KV4 SHIP3+DV SHIP4+BTP SHIP5+CONT TO1+F7")
_ROOT = Path.home() / "MscProj_tg"
OUTD = _ROOT / "benchmark" / "gold" / "ceiling_final"
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from benchmark.gold.spotsound_screen import flags as spot_flags
from src.labels import canonical
from src.types import AudioEvent
from src.stage4_audio_event_detection import (_p1v4_lists, _af_p1_accepts, _dasm_keeps, listener_p1_lookup, _cache_items)
from src.labels import is_descendant
from benchmark.gold.flap_joint_sim import load_fl, fl_span

from benchmark.gold import round13_dev as R
ARM = "SHIP8+MD3+WW5+SL"                               # D' = config.use_shipped (detector frozen 2 Oct)
B.ARM = ARM
CHAIN = ["B0r", "TO1+F7F8", "TO1F7F8+N2b", "SHIP+DR2", "SHIP2+KV4", "SHIP3+DV", "SHIP4+BTP", "SHIP5+CONT", "SHIP6+FLAP",
         "SHIP7+K4AD", "SHIP8", "SHIP8+MD3", "SHIP8+MD3+WW", "SHIP8+MD3+WW5", "SHIP8+MD3+WW5+SL"]
STAGE_OF = {"TO1+F7F8": "TO1+F7F8 (mirror veto F7 / rescue filters)", "TO1F7F8+N2b": "N2b masked-weak veto",
            "SHIP+DR2": "DR2", "SHIP2+KV4": "K-V4 keep test", "SHIP3+DV": "DV DASM clip veto", "SHIP4+BTP": "BTP",
            "SHIP5+CONT": "CONT continuation veto", "SHIP6+FLAP": "FLAP FineLAP veto",
            "SHIP7+K4AD": "K4A-D keep test (KEEP_NEEDS_V4_ALL onto + DASM keep)", "SHIP8": "SHIP8 (display only: MERGE_GAP 2.5 / GROUP)",
            "SHIP8+MD3": "MD3 min duration 0.3 s", "SHIP8+MD3+WW": "WW weak-witness DASM local veto 0.35",
            "SHIP8+MD3+WW5": "WW5 scene-margin", "SHIP8+MD3+WW5+SL": "SL scene logit"}
G = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
BASE = {"hits": 29, "wrong": 15, "cost": 2.056}
FLOOR = 0.35                # final arm: PICTURE_MIN_CONF None; DISPLAY_THRESHOLD = AUGMENT_THRESHOLD = 0.35
OUT = OUTD / "ledger_final.json"
SPOT = json.loads(next(f for f in (G / "spotsound_raw.json", G / "spotsound_raw_run.json") if f.exists()).read_text(encoding="utf-8"))["items"]   # same md5 on the cluster


def spot_item(part, it):
    """the SpotSound record of a P2/PV listener item: the item's own key (part, clip, pool, label, start, end)"""
    if it is None:
        return None
    c = [x for x in SPOT if x["part"] == part and x["clip"] == it["clip"] and x["pool"] == it["pool"] and x["label"] == it["label"]
         and abs(x["start"] - it["start"]) < 0.02 and abs(x["end"] - it["end"]) < 0.02]
    return c[0] if c else None


def p2_item(items, label, a, b):
    """the P2/PV listener item a rescued row came from: same family (same_family, or canonical equality: a child span such
    as Gunshot under Explosion inherits the rescue), run overlapping the row [a, b] (short runs were cut to a 1-s window, so
    the row may be wider than the run); preference: the item with the row's exact span, then a V4-accepted item (only those
    can have been rescued), then the nearest start"""
    def fam_ok(x):
        f = x.get("family") or x["label"]
        return S.same_family(f, label) or canonical(f) == canonical(label) or canonical(x["label"]) == canonical(label)
    c = [x for x in items if x.get("pool") in ("P2", "PV") and fam_ok(x) and x["start"] <= b + 0.05 and x["end"] >= a - 0.05]
    if not c:
        return None
    exact = lambda x: abs(x["start"] - a) <= 0.05 and abs(x["end"] - b) <= 0.05
    return min(c, key=lambda x: (not exact(x), not (x.get("accept") or {}).get("V4"), abs(x["start"] - a)))


def spot_p1(clip, fam, a, b):
    c = [x for x in SPOT if x["part"] == "p1" and x["clip"] == clip and canonical(x["label"]) == fam
         and abs(x["end"] - b) <= 0.05 and a - 0.05 <= x["start"] <= b]
    return min(c, key=lambda x: abs(x["start"] - a)) if c else None


def specs_of(root, st):
    f = root / st / "augmentations.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def set_clip(pt, st):
    cfg = CG.PARTS[pt]
    config._CURRENT_CLIP = st
    config.RELABEL_P1V4 = str(G / f"{cfg['lis']}_listener_p1v4.json")
    config.LISTENER_AFCACHE = str(G / f"{cfg['lis']}_listener_afn.json")
    config.LISTENER_DASM_DIR = str(cfg["dasm"]); config.LISTENER_DASM_BAR = 0.575
    return cfg


def row_info(pt, st, r, C, lp1, flex, beats, dasm):
    fam = canonical(r["label"])
    e = AudioEvent(r["label"], r["start"], r["end"], r["conf"], rescued=r.get("rescued", False))
    q, af = _p1v4_lists(e)
    it2 = p2_item(C["v"], r["label"], r["start"], r["end"]) if r.get("rescued") else None
    af2 = p2_item(C["afn"], r["label"], r["start"], r["end"]) if r.get("rescued") else None
    sp = spot_item(pt, it2)
    sp1 = spot_p1(st, fam, r["start"], r["end"]) if pt == "dev" else None
    return {"label": r["label"], "start": r["start"], "end": r["end"], "conf": round(r["conf"], 3), "origin": r["origin"],
            "rescued": r.get("rescued", False), "pre_start": r.get("pre_start"),
            "p1_qwen": q, "p1_af": af, "p1_lookup": list(lp1(r["label"], r["start"], r["end"])), "p1_af_accept": _af_p1_accepts(e),
            "p2_pool": (it2 or {}).get("pool"), "p2_label": (it2 or {}).get("label"), "p2_accept": (it2 or {}).get("accept"), "p2_peak": (it2 or {}).get("peak"),
            "p2_af_accept": (af2 or {}).get("accept"),
            "spot_p2": ({"exist": sp["x_exist"], "ground": sp["x_ground"], **spot_flags(sp)} if sp else None),
            "spot_p1": ({"exist": sp1["x_exist"], "ground": sp1["x_ground"], **spot_flags(sp1)} if sp1 else None),
            "dasm_keeps": _dasm_keeps(e), "dasm_span": CG.fam_peak(dasm, fam, r["start"], r["end"], 0.5),
            "dasm_clip": CG.clip_peak(dasm, fam), "flex_span": CG.fam_peak(flex, fam, r["start"], r["end"]),
            "flex_clip": CG.clip_peak(flex, fam), "beats_span": CG.fam_peak(beats, fam, r["start"], r["end"]),
            "beats_clip": CG.clip_peak(beats, fam)}


def produced_by(r, lis):
    """the stage that made this stage-4 row a candidate (from the arm's listener record)"""
    lab, a = r["label"], r["start"]
    near = lambda rows: any(S.same_family(x[0], lab) and x[1] <= r["end"] + 0.1 and (x[2] if len(x) > 2 else x[1]) >= a - 0.1 for x in rows)
    if r.get("rescued"):
        if near(lis.get("a_added", [])):
            return "band rescue (TIER listener, P2)"
        if near(lis.get("b_kept", [])):
            return "PV keep (PANNs-vetoed, listener kept)"
        if near(lis.get("c_added", [])):
            return "BEATs-band rescue (P3)"
        return "DASM rescue (DR2, P4)" if r["origin"] == "tagger" else "rescued (P2 item, not in the arm's list)"
    if any(S.same_family(x[0], lab) and abs(x[1] - a) <= 0.3 and x[3] for x in lis.get("f7", [])):
        return f"{r['origin']} span, mirror-vetoed, kept by listener (F7)"
    return f"{r['origin']} span" + (" (BTP moved from %.2f)" % r["pre_start"] if r.get("pre_start") not in (None, r["start"]) else "")


def in_win(a, onset):
    return S.in_window(a, onset, S.EARLY, S.LATE)


def placed(work_root, stem):
    """btp_screen.placed with clip=stem (the final arm's GROUP / DEPICT display rules need the clip name, as load_pictures)"""
    f = work_root / stem / "augmentations.json"
    if not f.exists():
        return []
    specs = json.loads(f.read_text(encoding="utf-8"))
    m = work_root / stem / "media.json"
    dur = (float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None) if m.exists() else None
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs, raw = [], {}
    for s in specs:
        o = AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])])
        objs.append(o); raw[id(o)] = s
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    pl, _ = _assign_rows(_display_spans(objs, d, require_image=True, clip=stem))
    return [(lab, float(a), float(b), bool(raw[id(sp)].get("rescued", False))) for _, lab, a, b, sp in pl]


def parts(arm=None):
    """btp_screen.parts for ARM with the clip-aware placed(); the dev2 gold temp file goes to OUTD (no existing file written)"""
    arm = arm or ARM
    gold, stems = DCC.dev_stems()
    with R.flags({k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}):
        dev = [("dev", st, gold[st], placed(R.R13 / f"{arm}_proposed", st)) for st in stems]
    from benchmark.gold import tagger_prep as T
    DCC2, R2, stems2 = T.configure("dev2")
    keep = set(stems2)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = OUTD / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    g2 = T._REAL_LOAD_GOLD([tmp])
    o = T.out("dev2")
    with R2.flags({k: R2.arm_cfg(arm)[k] for k in R2.DISPLAY_KEYS}):
        dev2 = [("dev2", st, g2[st], placed(o / f"{arm}_proposed", st)) for st in stems2 if st in g2]
    return dev + dev2


B.placed = placed
B.parts = parts


def main():
    P = parts()
    s4 = {pt: json.loads(CG.PARTS[pt]["stage4"].read_text(encoding="utf-8")) for pt in CG.PARTS}
    caches = {}
    for pt, cfg in CG.PARTS.items():
        g = cfg["lis"]
        caches[pt] = {"v": _cache_items(str(G / f"{g}_listener_v.json")), "afn": _cache_items(str(G / f"{g}_listener_afn.json")),
                      "yn": _cache_items(str(G / f"{g}_listener.json"))}
    roots = {"dev": WORK / "r13", "dev2": WORK / "r13dev2"}         # explicit: tagger_prep.configure redirects R.R13
    rows_all = {"dev": [], "dev2": []}
    wrong, misses = [], []
    # SPOT-V simulation state
    sim = {k: {"rows": {"dev": [], "dev2": []}, "dropped": []} for k in ("exist_no", "spot_no", "K4AO", "K4A4E")}
    n_resc = {"placed": 0, "no_item": 0, "no_spot": 0}
    p1_nonmissing = 0
    for pt, st, gold, pics in P:
        cfg = set_clip(pt, st); C = caches[pt]
        r0 = S.score_clip(gold, [p[:3] for p in pics]); rows_all[pt].append(r0)
        cl = CG.classify(gold, [p[:3] for p in pics])
        tally = {k: sum(1 for x in cl if x[3] == k) for k in ("hit", "visible", "cross", "phantom", "dup")}
        assert all(tally[k] == r0[k] for k in tally), (st, tally)
        lp1 = listener_p1_lookup(str(G / f"{cfg['lis']}_listener_v.json"), str(G / f"{cfg['lis']}_listener.json"), st)
        fx = B.FLEX_DIR / f"{st}.npz"; flex = DCC.load_fr(fx) if fx.exists() else None
        bx = cfg["beats"] / f"{st}.npz"; beats = DCC.load_fr(bx) if bx.exists() else None
        dx = cfg["dasm"] / f"{st}.npz"; dasm = DCC.load_fr(dx) if dx.exists() else None
        arms = {a: s4[pt]["arms"].get(f"{a}|proposed", {}).get(st, []) for a in CHAIN}
        s7 = arms[ARM]
        lis = s4[pt]["listener"].get(f"{ARM}|proposed|{st}", {})
        rd = s4[pt]["r14_dropped"].get(f"{ARM}|proposed|{st}", {})
        specs = specs_of(roots[pt] / f"{B.ARM}_proposed", st)
        gsorted = sorted(gold, key=lambda g: g["start"])
        # ---------------------------------------------------------------- wrong pictures
        for lab, a, b, k, gi in cl:
            if k not in ("cross", "visible", "phantom"):
                continue
            fam = canonical(lab)
            resc = next((r for (l2, a2, b2, r) in pics if l2 == lab and abs(a2 - a) < 1e-6), False)
            at = [g for g in gold if in_win(a, g["start"]) or (g["start"] <= a <= g["end"])]
            own = [g for g in gold if S.same_family(lab, g["label"])]
            own_s = None
            if own:
                gg = min(own, key=lambda g: abs(a - g["start"])); d = a - gg["start"]
                own_s = {"label": gg["label"], "start": gg["start"], "end": gg["end"], "delta": round(d, 2),
                         "how": "late" if d > S.LATE else "early" if d < -S.EARLY else "window", "needed": gg["needed"],
                         "visible": gg["visible"], "obvious": gg["obvious"], "importance": gg["importance"]}
            srows = [r for r in s7 if canonical(r["label"]) == fam and r["end"] >= a - 0.1 and r["start"] <= b + 0.1]
            sp = [x for x in specs if canonical(x["event_label"]) == fam and abs(float(x["start"]) - a) <= 0.5]
            info = [row_info(pt, st, r, C, lp1, flex, beats, dasm) for r in srows]
            p1_nonmissing += sum(1 for i in info if i["p1_lookup"][1] != "missing")
            prod = [produced_by(r, lis) for r in srows]
            wrong.append({"kind": "wrong", "part": pt, "clip": st, "label": lab, "family": fam, "start": round(a, 2), "end": round(b, 2),
                          "class": k, "rescued": resc, "produced_by": prod, "gate_reason": [x.get("reason") for x in sp][:2],
                          "gold_at": [{"label": g["label"], "start": g["start"], "end": g["end"], "needed": g["needed"], "visible": g["visible"],
                                       "obvious": g["obvious"], "importance": g["importance"]} for g in at],
                          "own_family_gold": own_s, "stage4": info})
        # ---------------------------------------------------------------- misses
        taken = set()
        for x in cl:
            if x[4] is not None:
                taken.add(id(gsorted[x[4]]))
        r0m = S.score_clip(gold, [p[:3] for p in pics])
        matched = set()
        # re-derive matched needed indices exactly as score_clip does (covered sounds included)
        gs = gsorted; tk = [False] * len(gs)
        for lab, a, b in sorted([p[:3] for p in pics], key=lambda p: p[1]):
            cands = [i for i, g in enumerate(gs) if S.same_family(lab, g["label"]) and in_win(a, g["start"])]
            if not cands:
                continue
            free = [i for i in cands if not tk[i]]
            if not free:
                continue
            i = min(free, key=lambda i: abs(a - gs[i]["start"])); tk[i] = True
            for j in [i] + [j for j in cands if not tk[j] and S.same_family(gs[j]["label"], gs[i]["label"])]:
                tk[j] = True; matched.add(j)
        for gi, g in enumerate(gs):
            if not (g["needed"] and g["importance"] >= S.MIN_IMPORTANCE) or gi in matched:
                continue
            fam = canonical(g["label"]); on = g["start"]
            rec = {"kind": "miss", "part": pt, "clip": st, "label": g["label"], "family": fam, "start": on, "end": g["end"],
                   "visible": g["visible"], "obvious": g["obvious"], "importance": g["importance"]}
            # 1 gate-silenced / 3 floor: a same-family spec in the window
            spw = [x for x in specs if S.same_family(x["event_label"], g["label"]) and in_win(float(x["start"]), on)]
            # 2 timing: a same-family placed picture in the clip outside the window
            pw = [(l, a2, b2) for (l, a2, b2, _r) in pics if S.same_family(l, g["label"])]
            # 4 arm chain: same-family rows in the window per arm
            chain = {arm: [r for r in rows if S.same_family(r["label"], g["label"]) and in_win(r["start"], on)] for arm, rows in arms.items()}
            over7 = [r for r in s7 if S.same_family(r["label"], g["label"]) and r["start"] <= g["end"] and r["end"] >= on - 0.5]
            # 5 rescue filters
            rf = {k: [x for x in v if S.same_family(x[0], g["label"]) and in_win(float(x[1]), on)] for k, v in rd.items() if isinstance(v, list)}
            rf = {k: v for k, v in rf.items() if v}
            # 6 listener candidates (P2 / PV) in the window
            li = [x for x in C["v"] if x.get("clip") == st and x.get("pool") in ("P2", "PV") and S.same_family(x.get("family") or x["label"], g["label"])
                  and in_win(float(x.get("run_start", x["start"])), on)]
            la = [x for x in C["afn"] if x.get("clip") == st and x.get("pool") in ("P2", "PV") and S.same_family(x.get("family") or x["label"], g["label"])
                  and in_win(float(x.get("run_start", x["start"])), on)]
            spl = [spot_item(pt, x) for x in li]
            fxp = CG.fam_peak(flex, fam, on - S.EARLY, on + S.LATE); btp = CG.fam_peak(beats, fam, on - S.EARLY, on + S.LATE)
            dsp = CG.fam_peak(dasm, fam, on - S.EARLY, on + S.LATE)
            rec.update({"specs_in_window": [{"label": x["event_label"], "start": x["start"], "conf": round(float(x.get("confidence", 0)), 3),
                                             "augment": x.get("augment"), "reason": x.get("reason")} for x in spw],
                        "pictures_of_family": [[l, round(a2, 2), round(b2, 2)] for l, a2, b2 in pw],
                        "rows_in_window_by_arm": {a: [[r["label"], r["start"], r["end"], round(r["conf"], 3), r["origin"], r.get("rescued", False)] for r in v]
                                                  for a, v in chain.items()},
                        "ship7_rows_overlapping": [[r["label"], r["start"], r["end"], round(r["conf"], 3), r["origin"], r.get("rescued", False)] for r in over7],
                        "rescue_filter_dropped": rf,
                        "listener_cands": [{"pool": x["pool"], "label": x["label"], "start": x["start"], "end": x["end"], "peak": x.get("peak"),
                                            "qwen": x.get("accept"), "af": next((y.get("accept") for y in la if y["label"] == x["label"] and abs(y["start"] - x["start"]) < 0.05), None),
                                            "spot": ({"exist": s_["x_exist"], **spot_flags(s_)} if s_ else None)} for x, s_ in zip(li, spl)],
                        "flex_peak_window": fxp, "beats_peak_window": btp, "dasm_peak_window": dsp,
                        "flex_clip": CG.clip_peak(flex, fam), "beats_clip": CG.clip_peak(beats, fam)})
            # attribution (fixed order, as registered)
            lio = [x for x in C["v"] if x.get("clip") == st and x.get("pool") in ("P2", "PV") and S.same_family(x.get("family") or x["label"], g["label"])
                   and float(x.get("run_start", x["start"])) <= on + S.LATE and float(x.get("run_end", x["end"])) >= on - S.EARLY]
            rows_win = [r for a_ in CHAIN for r in chain[a_]]
            max_conf = max((r["conf"] for r in rows_win), default=None)
            b0_has = bool(chain["B0r"])
            if spw and any(not x.get("augment") and float(x.get("confidence", 0)) >= FLOOR for x in spw):
                stage = "gate-silenced"
            elif spw and any(x.get("augment") for x in spw):
                stage = "display-dropped (augment spec %s in window, not placed: GROUP repeat merge / DEPICT / merge / row overflow)" % ", ".join(
                    "%s %.2f conf %.2f" % (x["event_label"], float(x["start"]), float(x.get("confidence", 0))) for x in spw if x.get("augment"))
            elif spw:
                stage = "below picture floor (spec conf %.2f < 0.35)" % max(float(x.get("confidence", 0)) for x in spw)
            elif chain[ARM] and max(r["conf"] for r in chain[ARM]) < FLOOR:
                stage = "below display bar / floor (row conf %.2f)" % max(r["conf"] for r in chain[ARM])
            elif chain[ARM]:
                stage = "stage-4 row in window but no spec (display merge)"
            elif rows_win:
                last = max(i for i, a_ in enumerate(CHAIN) if chain[a_])
                stage = "removed by %s (present in %s, row conf %.2f%s" % (STAGE_OF.get(CHAIN[last + 1], CHAIN[last + 1]), CHAIN[last], max_conf,
                                                                         " < floor anyway)" if max_conf < FLOOR else ")")
            elif rf:
                stage = "rescue filter " + "/".join(sorted(rf))
            elif li:
                q = any((x.get("accept") or {}).get("V4") for x in li)
                a_ = any((y.get("accept") or {}).get("V4") for y in la)
                pk = max(float(x.get("peak") or 0) for x in li)
                stage = "listener refused (Qwen V4 %s, AF V4 %s; peak %.2f %s)" % ("yes" if q else "no", "yes" if a_ else "no", pk,
                                                                                    "high tier: Qwen alone" if pk >= 0.6 else "low tier: both")
            elif lio:
                stage = "listener candidate mistimed (P2 run starts %.2f, outside the onset window)" % min(float(x.get("run_start", x["start"])) for x in lio)
            elif over7:
                stage = "stage-4 timing (row overlaps the sound, start outside the window)"
            elif btp is not None and btp >= 0.35 and not b0_has:
                fc = CG.clip_peak(flex, fam)
                stage = "B0 veto: BEATs %s >= display bar in window, no B0r row (%s" % (btp,
                        ("FlexSED clip max %s < 0.3: FlexSED clip veto)" % fc) if fc is not None and fc < 0.3 else "PANNs veto / min span)")
            elif fxp is not None and fxp >= 0.5:
                stage = "FlexSED >= 0.5 in window but no P2 run (covered / never asked)"
            elif fxp is not None and fxp >= 0.3:
                stage = "below LO 0.5 (FlexSED %s)" % fxp
            elif btp is not None and btp >= 0.175:
                stage = "BEATs weak (%s < display bar), FlexSED %s" % (btp, fxp)
            else:
                stage = "unheard (FlexSED %s, BEATs %s, DASM %s)" % (fxp, btp, dsp)
            if pw:
                d = min((a2 - on for _, a2, _ in pw), key=abs)
                rec["family_picture_elsewhere"] = "%s by %.2f s" % ("late" if d > 0 else "early", abs(d))
            ceiling = ("annotation: visible" if g["visible"] else "") or ("unheard" if stage.startswith("unheard") else "")
            rec.update({"removed_by": stage, "ceiling": ceiling or None})
            misses.append(rec)
    base = {pt: B.summ(rows_all[pt]) for pt in rows_all}; base["merged"] = B.summ(rows_all["dev"] + rows_all["dev2"])
    print(f"BASE {ARM}: merged {B.fmt(base['merged'])} | DEV {B.fmt(base['dev'])} | DEV2 {B.fmt(base['dev2'])}")
    m0 = base["merged"]
    assert (m0["hits"], m0["wrong"], round(m0["cost"], 3)) == (BASE["hits"], BASE["wrong"], BASE["cost"]), m0
    assert len(misses) == m0["n"] - m0["hits"] and len(wrong) == m0["wrong"], (len(misses), len(wrong))
    res = {"arm": ARM, "base": base, "wrong": wrong, "misses": misses}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    from collections import Counter
    print("\nWRONG", Counter((w["class"], w["rescued"]) for w in wrong))
    for w in wrong:
        o = w["own_family_gold"]
        print(f"  {w['part']:4s} {w['clip'][:28]:28s} {w['label'][:22]:22s} {w['start']:6.2f} {w['class']:8s} resc={w['rescued']!s:5s} "
              f"by={'; '.join(w['produced_by'])[:60]} own={(o['how'] + ' d' + str(o['delta'])) if o else '-'} "
              f"at={','.join(g['label'][:14] + ('V' if g['visible'] else '') for g in w['gold_at'])[:50]}")
    print("\nMISSES", Counter(m["removed_by"].split(" (")[0] for m in misses))
    for m in misses:
        print(f"  {m['part']:4s} {m['clip'][:28]:28s} {m['label'][:22]:22s} {m['start']:6.2f} vis={m['visible']!s:5s} {m['removed_by']}")
    print(f"\n{len(wrong)} wrong, {len(misses)} misses; {OUT}")


if __name__ == "__main__":
    main()
