"""Round 31 cross-group listing (Fable; docs/prereg_round13_detector_push.md "Round 31"). CPU only, on the saved
SHIP5 (= SHIP4+BTP) proposed pictures of merged DEV: every wrong picture is classed per picture with the scorer's own
greedy loop (score_per_sound.score_clip), and each CROSS picture gets its cause: the gold sounds at its moment, its own
family's gold in the clip (late / early / repeat), its stage-4 rows (origin, confidence, rescued), the listeners' answers
on its P1/P2 cut, DASM / FlexSED / BEATs family evidence. Nothing in src/ or config.py is edited; no picture is changed.

    python benchmark/gold/cross_group.py            -> benchmark/gold/cross_group.json (+ a table on stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import btp_screen as B
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from src.labels import canonical
from src.types import AudioEvent
from src.stage4_audio_event_detection import (_p1v4_lists, _af_p1_accepts, _dasm_keeps, listener_p1_lookup, _cache_items)

B.ARM = "SHIP4+BTP"                                  # = SHIP5 (config.use_shipped): pictures saved by the round-30 arm
WORK = _ROOT / "data" / "work"
G = _ROOT / "benchmark" / "gold"
PARTS = {
    "dev": {"stage4": WORK / "r13" / "stage4.json", "lis": "dev", "dasm": WORK / "devcand" / "dasm_cache",
            "beats": WORK / "j2_dev_beats"},
    "dev2": {"stage4": WORK / "r13dev2" / "stage4.json", "lis": "dev2", "dasm": WORK / "dasm_dev2",
             "beats": WORK / "j2_dev2_beats"},
}
OUT = G / "cross_group.json"


def classify(gold, pics, early=S.EARLY, late=S.LATE):
    """score_clip's greedy loop, verbatim, but returning the class of every picture (in start order)"""
    gold = sorted(gold, key=lambda g: g["start"]); pics = sorted(pics, key=lambda p: p[1])
    taken = [False] * len(gold)
    scored = lambda g: S.OLD_RULE or g["importance"] >= S.MIN_IMPORTANCE
    out = []
    for lab, a, b in pics:
        cands = [i for i, g in enumerate(gold) if S.same_family(lab, g["label"]) and S.in_window(a, g["start"], early, late)]
        if cands:
            free = [i for i in cands if not taken[i]]
            if not free:
                out.append((lab, a, b, "dup", None)); continue
            i = min(free, key=lambda i: abs(a - gold[i]["start"]))
            taken[i] = True
            covered = [i] + [j for j in cands if not taken[j] and S.same_family(gold[j]["label"], gold[i]["label"])]
            hit_here = False
            for j in covered:
                taken[j] = True
                g = gold[j]
                if g["needed"] and scored(g):
                    hit_here = True
            if any(not gold[j]["needed"] for j in covered):
                out.append((lab, a, b, "collision" if hit_here else "visible", i))
            else:
                out.append((lab, a, b, "hit" if hit_here else "dontcare", i))
        else:
            any_sound = any(S.in_window(a, g["start"], early, late) or (g["start"] <= a <= g["end"]) for g in gold)
            out.append((lab, a, b, "cross" if any_sound else "phantom", None))
    return out


def fam_peak(fr, fam, a, b, pad=0.0):
    if fr is None:
        return None
    fw, t, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    m = (t >= a - pad) & (t <= b + pad)
    if not cols or not m.any():
        return None
    return round(float(fw[m][:, cols].max()), 3)


def clip_peak(fr, fam):
    if fr is None:
        return None
    fw, t, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    return round(float(fw[:, cols].max()), 3) if cols else None


def p2_item(items, fam, a, b):
    """the P2/PV listener item of a rescued row: same family, run overlapping the row"""
    c = [x for x in items if x.get("family") == fam and x.get("pool") in ("P2", "PV") and x["start"] <= b + 0.05 and x["end"] >= a - 0.05]
    return min(c, key=lambda x: abs(x["start"] - a)) if c else None


def main():
    P = B.parts()
    caches = {}
    for pt, cfg in PARTS.items():
        g = cfg["lis"]
        caches[pt] = {"stage4": json.loads(cfg["stage4"].read_text(encoding="utf-8"))["arms"][f"{B.ARM}|proposed"],
                      "v": [x for x in _cache_items(str(G / f"{g}_listener_v.json"))],
                      "afn": [x for x in _cache_items(str(G / f"{g}_listener_afn.json"))],
                      "yn": [x for x in _cache_items(str(G / f"{g}_listener.json"))]}
    rows_all = {"dev": [], "dev2": []}
    table = []
    for pt, st, gold, pics in P:
        cfg = PARTS[pt]; C = caches[pt]
        r0 = S.score_clip(gold, [p[:3] for p in pics]); rows_all[pt].append(r0)
        cl = classify(gold, [p[:3] for p in pics])
        tally = {k: sum(1 for x in cl if x[3] == k) for k in ("hit", "visible", "cross", "phantom", "dup")}
        assert all(tally[k] == r0[k] for k in tally), (st, tally, {k: r0[k] for k in tally})
        if not any(x[3] in ("cross", "visible", "phantom") for x in cl):
            continue
        # per-clip context
        config._CURRENT_CLIP = st
        config.RELABEL_P1V4 = str(G / f"{cfg['lis']}_listener_p1v4.json")
        config.LISTENER_AFCACHE = str(G / f"{cfg['lis']}_listener_afn.json")
        config.LISTENER_DASM_DIR = str(cfg["dasm"]); config.LISTENER_DASM_BAR = 0.575
        lp1 = listener_p1_lookup(str(G / f"{cfg['lis']}_listener_v.json"), str(G / f"{cfg['lis']}_listener.json"), st)
        fx = B.FLEX_DIR / f"{st}.npz"; flex = DCC.load_fr(fx) if fx.exists() else None
        bx = cfg["beats"] / f"{st}.npz"; beats = DCC.load_fr(bx) if bx.exists() else None
        dx = cfg["dasm"] / f"{st}.npz"; dasm = DCC.load_fr(dx) if dx.exists() else None
        s4 = C["stage4"].get(st, [])
        for lab, a, b, k, gi in cl:
            if k not in ("cross", "visible", "phantom"):
                continue
            fam = canonical(lab)
            resc = next((r for (l2, a2, b2, r) in pics if l2 == lab and abs(a2 - a) < 1e-6), False)
            at = [g for g in gold if S.in_window(a, g["start"], S.EARLY, S.LATE) or (g["start"] <= a <= g["end"])]
            own = [g for g in gold if S.same_family(lab, g["label"])]
            own_s = None
            if own:
                gg = min(own, key=lambda g: abs(a - g["start"]))
                d = a - gg["start"]
                how = "late" if d > S.LATE else "early" if d < -S.EARLY else "window"
                own_s = {"label": gg["label"], "start": gg["start"], "end": gg["end"], "delta": round(d, 2), "how": how,
                         "needed": gg["needed"], "visible": gg["visible"], "obvious": gg["obvious"], "importance": gg["importance"],
                         "hit_elsewhere": any(x[3] in ("hit", "dontcare", "collision", "visible") and x[4] is not None
                                              and sorted(gold, key=lambda g: g["start"])[x[4]] is gg for x in cl)}
            srows = []
            for r in s4:
                if canonical(r["label"]) != fam or r["end"] < a - 0.1 or r["start"] > b + 0.1:
                    continue
                e = AudioEvent(r["label"], r["start"], r["end"], r["conf"], rescued=r.get("rescued", False))
                q, af = _p1v4_lists(e)
                it2 = p2_item(C["v"], fam, r["start"], r["end"]) if r.get("rescued") else None
                af2 = p2_item(C["afn"], fam, r["start"], r["end"]) if r.get("rescued") else None
                yn = [x for x in C["yn"] if x.get("clip") == st and x.get("family") == fam and x.get("pool") == "P1"
                      and abs(x["end"] - r["end"]) <= 0.02 and r["start"] - 0.02 <= x["start"] <= r["end"]]
                srows.append({"label": r["label"], "start": r["start"], "end": r["end"], "conf": round(r["conf"], 3),
                              "origin": r["origin"], "rescued": r.get("rescued", False), "agree": r.get("agree", False),
                              "pre_start": r.get("pre_start"), "refine": r.get("refine"),
                              "p1_qwen": q, "p1_af": af, "p1_lookup": list(lp1(r["label"], r["start"], r["end"])),
                              "p1_af_accept": _af_p1_accepts(e), "p1_yesno": (yn[0].get("score") if yn else None),
                              "p2_accept": (it2 or {}).get("accept"), "p2_peak": (it2 or {}).get("peak"),
                              "p2_af_accept": (af2 or {}).get("accept"),
                              "dasm_keeps": _dasm_keeps(e),
                              "dasm_span": fam_peak(dasm, fam, r["start"], r["end"], 0.5), "dasm_clip": clip_peak(dasm, fam),
                              "flex_span": fam_peak(flex, fam, r["start"], r["end"]), "flex_clip": clip_peak(flex, fam),
                              "beats_span": fam_peak(beats, fam, r["start"], r["end"]), "beats_clip": clip_peak(beats, fam)})
            rec = {"part": pt, "clip": st, "label": lab, "start": round(a, 2), "end": round(b, 2), "class": k, "rescued": resc,
                   "gold_at": [{"label": g["label"], "start": g["start"], "end": g["end"], "needed": g["needed"],
                                "visible": g["visible"], "obvious": g["obvious"], "importance": g["importance"]} for g in at],
                   "own_family_gold": own_s, "stage4": srows}
            table.append(rec)
    M = {pt: B.summ(rows_all[pt]) for pt in rows_all}
    M["merged"] = B.summ(rows_all["dev"] + rows_all["dev2"])
    print(f"BASE {B.ARM} (no change): merged {B.fmt(M['merged'])} | DEV {B.fmt(M['dev'])} | DEV2 {B.fmt(M['dev2'])}")
    OUT.write_text(json.dumps({"base": M, "wrong": table}, indent=1, default=float), encoding="utf-8")
    for rec in table:
        if rec["class"] != "cross":
            continue
        o = rec["own_family_gold"]
        own = f"own {o['label']} @{o['start']} {o['how']} d{o['delta']}{' hitElse' if o['hit_elsewhere'] else ''}" if o else "own: none"
        at = "; ".join(f"{g['label']}@{g['start']}-{g['end']}{'V' if g['visible'] else ''}{'O' if g['obvious'] else ''}{'N' if g['needed'] else ''}" for g in rec["gold_at"])
        print(f"\n{rec['part']:4s} {rec['clip']} {rec['label']} {rec['start']}-{rec['end']} resc={rec['rescued']} | {own} | at: {at}")
        for r in rec["stage4"]:
            print(f"      s4 {r['label']} {r['start']}-{r['end']} c{r['conf']} {r['origin']}{' RESC' if r['rescued'] else ''}{' agree' if r['agree'] else ''} "
                  f"pre{r['pre_start']} | p1 qwen={r['p1_qwen']} af={r['p1_af']} look={r['p1_lookup']} afacc={r['p1_af_accept']} yn={r['p1_yesno']} "
                  f"| p2={r['p2_accept']} pk={r['p2_peak']} af2={r['p2_af_accept']} | dasm span {r['dasm_span']} clip {r['dasm_clip']} keeps={r['dasm_keeps']} "
                  f"| flex {r['flex_span']}/{r['flex_clip']} beats {r['beats_span']}/{r['beats_clip']}")
    n = sum(1 for r in table if r["class"] == "cross")
    print(f"\n{n} cross pictures listed; {OUT}")


if __name__ == "__main__":
    main()
