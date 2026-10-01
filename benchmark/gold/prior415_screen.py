"""Round 41 PRIOR415 (docs/prereg_round13_detector_push.md "Round 41 PRIOR415"): an independent per-family precision prior
for the two detectors from the held-out 415 AudioSet-Strong clips (strong labels, disjoint from DEV/TEST), then a screen on
the saved SHIP8 pictures of merged DEV that drops a non-rescued picture whose family is UNRELIABLE for the model that
produced it unless a listener names the family on its P1 cut. Nothing in src/ or config.py is edited.

    python benchmark/gold/prior415_screen.py prior                  # CPU, cluster: -> prior415_lists.json (frozen, committed)
    TG_ARMS=SHIP8 python benchmark/gold/prior415_screen.py screen   # CPU, from ~/MscProj_tg: -> prior415_screen.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
import config
from src.labels import canonical
from src.stage4_audio_event_detection import _extract_events

HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
FLEX_HELD = Path(os.environ.get("FLEX_HELD", str(_ROOT / "data" / "work" / "flexsed_heldout")))
BEATS_HELD = Path(os.environ.get("BEATS_HELD", "/home/dsi/adamg/MscProj/benchmark/audioset_heldout_windows/beats"))
LISTS = _ROOT / "benchmark" / "gold" / "prior415_lists.json"
OUT = _ROOT / "benchmark" / "gold" / "prior415_screen.json"
LEDGER = _ROOT / "benchmark" / "gold" / "ledger_ship8.json"
EARLY, LATE = 0.5, 1.0
MIN_INST, LOW_P, HIGH_P = 10, 0.3, 0.8
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}                 # SHIP8 on merged DEV (gold_AG.json), copied, never imported
ORIGIN_MODEL = {"tagger": "beats", "flex": "flexsed"}


def _load(p):
    """dev_candidates_check.load_fr: (fw [T, Q] float32, times, labels); FlexSED caches store fw.T + fps, BEATs times"""
    from benchmark.gold import dev_candidates_check as DCC
    return DCC.load_fr(p)


def shipped_flags():
    config.use_shipped()
    f = {"FLEXSED_BAR": config.FLEXSED_BAR, "FLEXSED_FAMILY_BARS": config.FLEXSED_FAMILY_BARS,
         "IMPULSE_MIN_SPAN": config.IMPULSE_MIN_SPAN, "AED_HYSTERESIS": config.AED_HYSTERESIS,
         "AED_MIN_DUR": config.AED_MIN_DUR, "AED_THRESHOLD": config.AED_THRESHOLD,
         "DISPLAY_THRESHOLD": config.DISPLAY_THRESHOLD, "AED_RELEASE": config.AED_RELEASE}
    assert f["FLEXSED_BAR"] == 0.8 and f["FLEXSED_FAMILY_BARS"] is None and f["IMPULSE_MIN_SPAN"] is None, f
    assert f["AED_HYSTERESIS"] == 1.0 and f["AED_MIN_DUR"] == 0.5 and f["AED_THRESHOLD"] == 0.175, f
    assert f["DISPLAY_THRESHOLD"] == 0.35 and f["AED_RELEASE"] is None, f
    return f


def flex_runs(fw, t, labs):
    """FlexSED runs as fuse_flexsed forms them under the shipped flags (raw: before the vetoes and the listener)"""
    return [(e.label, float(e.start), float(e.end)) for e in _extract_events(fw, t, labs, 0.8, None, 0.5, low=0.8)]


def beats_runs(fw, t, labs):
    """BEATs spans at the display bar: extraction at AED_THRESHOLD 0.175 keeping peak >= 0.35 (raw: before ONSET_CAM)"""
    return [(e.label, float(e.start), float(e.end)) for e in _extract_events(fw, t, labs, 0.35, None, 0.5, low=0.175)]


RUNS = {"flexsed": flex_runs, "beats": beats_runs}
DIRS = {"flexsed": FLEX_HELD, "beats": BEATS_HELD}


def run_ok(fam, a, events):
    return any(canonical(ev["label"]) == fam and ev["start"] - EARLY <= a <= ev["start"] + LATE for ev in events)


def prior():
    flags = shipped_flags()
    H = json.loads(HELDOUT.read_text(encoding="utf-8"))
    clips = H["clips"]
    inst = Counter(canonical(ev["label"]) for c in clips for ev in c["events"])
    scored = sorted(f for f, n in inst.items() if n >= MIN_INST)
    res = {"flags": flags, "n_clips": len(clips), "min_instances": MIN_INST, "low": LOW_P, "high": HIGH_P,
           "rule": "run of family F correct iff a strong event with canonical(label)==F starts in [run.start-0.5, run.start+1.0]",
           "scored_families": {f: inst[f] for f in scored}, "models": {}}
    for model, fn in RUNS.items():
        d = DIRS[model]
        per = {f: {"runs": 0, "correct": 0, "instances": inst[f]} for f in scored}
        n_cached = 0
        for c in clips:
            p = d / f"{c['id']}.npz"
            if not p.exists():
                continue
            n_cached += 1
            fw, t, labs = _load(p)
            for lab, a, b in fn(fw, t, labs):
                f = canonical(lab)
                if f not in per:
                    continue
                per[f]["runs"] += 1
                per[f]["correct"] += run_ok(f, a, c["events"])
        for f, x in per.items():
            x["precision"] = round(x["correct"] / x["runs"], 4) if x["runs"] else None
        unrel = sorted(f for f, x in per.items() if x["precision"] is not None and x["precision"] < LOW_P)
        rel = sorted(f for f, x in per.items() if x["precision"] is not None and x["precision"] > HIGH_P)
        res["models"][model] = {"cache": str(d), "clips_cached": n_cached, "families": per, "UNRELIABLE": unrel,
                                "RELIABLE": rel, "thin": sorted(f for f in unrel + rel if per[f]["runs"] < 5)}
        print(f"{model}: {n_cached} clips cached; {len(scored)} scored families; UNRELIABLE {len(unrel)}: {unrel}; "
              f"RELIABLE {len(rel)}: {rel}; thin (<5 runs) {res['models'][model]['thin']}")
        for f in scored:
            x = per[f]
            print(f"   {f:40s} inst {x['instances']:4d} runs {x['runs']:4d} correct {x['correct']:4d} p {x['precision']}")
    LISTS.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(LISTS)


# ----------------------------------------------------------------------------- step 2
def passes(X, lost_parts, lost_n):
    gain = X["hits"] - BASE["hits"]
    old = (X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"]
           and not lost_parts)
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * lost_n and lost_n <= 3
    return old, few


def screen():
    from benchmark.gold import btp_screen as B
    from benchmark.gold import cross_group as CG
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    from benchmark.gold.gbtp_screen import cost_w
    from src.stage4_audio_event_detection import _p1v4_lists
    from src.types import AudioEvent
    B.ARM = "SHIP8"
    L = json.loads(LISTS.read_text(encoding="utf-8"))
    UNREL = {m: set(L["models"][m]["UNRELIABLE"]) for m in L["models"]}
    REL = {m: set(L["models"][m]["RELIABLE"]) for m in L["models"]}
    G = _ROOT / "benchmark" / "gold"
    P = B.parts()
    n_resc = sum(r for *_x, pics in P for *_y, r in pics)
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {n_resc} rescued")
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    assert abs(cost_w(Bm["merged"], Bm["merged"]["n"], 2) - Bm["merged"]["cost"]) < 1e-9
    BASE["cost"] = Bm["merged"]["cost"]
    s4 = {pt: json.loads(CG.PARTS[pt]["stage4"].read_text(encoding="utf-8"))["arms"]["SHIP8|proposed"] for pt in CG.PARTS}

    rows = {"dev": [], "dev2": []}; dropped = []; lost = []
    why_n = Counter(); named_n = Counter()
    for pt, st, g, pics in P:
        config._CURRENT_CLIP = st
        config.RELABEL_P1V4 = str(G / f"{CG.PARTS[pt]['lis']}_listener_p1v4.json")
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        keep = []
        for l, a, b, resc in pics:
            fam = canonical(l)
            crow = [r for r in s4[pt].get(st, []) if canonical(r["label"]) == fam and r["end"] >= a - 0.1 and r["start"] <= b + 0.1]
            if resc:
                why = "rescued"
            elif not crow:
                why = "no stage4 row"
            else:
                r0 = min(crow, key=lambda r: abs(r["start"] - a))
                model = ORIGIN_MODEL[r0["origin"]]
                if fam not in UNREL[model]:
                    why = f"{model} not unreliable"
                else:
                    names, n_items = set(), 0
                    for r in crow:
                        q, af = _p1v4_lists(AudioEvent(r["label"], r["start"], r["end"], r["conf"]))
                        if q is not None or af is not None:
                            n_items += 1
                        names |= {canonical(x) for x in (q or []) + (af or [])}
                    if fam in names:
                        why = "listener names family"
                        named_n["kept: named"] += 1
                    else:
                        why = "dropped"
                        named_n["no P1 item" if n_items == 0 else "P1 item, not named"] += 1
                        dropped.append({"part": pt, "clip": st, "label": l, "family": fam, "start": round(a, 2),
                                        "end": round(b, 2), "model": model, "row": [r0["label"], r0["start"], r0["end"]],
                                        "p1_items": n_items, "names": sorted(names), "before": before[(l, round(a, 3))]})
            if why != "dropped":
                why_n[why] += 1
                keep.append((l, a, b))
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
        rows[pt].append(r1)
        oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
        for d in dropped:
            if d["part"] == pt and d["clip"] == st and "clip_after" not in d:
                d["clip_before"], d["clip_after"] = oc(r0), oc(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    for k in X:
        X[k]["cost_w1"] = cost_w(X[k], X[k]["n"], 1)
    ln = sum(x[2] for x in lost)
    old, few = passes(X["merged"], lost, ln)
    verdict = "GO (main rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
    print(f"PRIOR415: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  dropped {len(dropped)}  "
          f"untouched {dict(why_n)}  exception {dict(named_n)}  hits lost {lost} -> {verdict}")
    for d in dropped:
        print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['start']}-{d['end']} [{d['model']}] p1 items {d['p1_items']} "
              f"names {d['names']}: {d['before']} dropped   clip {d['clip_before']} -> {d['clip_after']}")

    # RELIABLE: list only, not scored -- needed SHIP8 misses a RELIABLE family's raw run would reach
    allow = []
    led = json.loads(LEDGER.read_text(encoding="utf-8"))["misses"]
    for m in led:
        fam = canonical(m["sound"]); pt, st = m["part"], m["clip"]
        for model in ("beats", "flexsed"):
            if fam not in REL[model]:
                continue
            p = (CG.PARTS[pt]["beats"] if model == "beats" else B.FLEX_DIR) / f"{st}.npz"
            if not p.exists():
                allow.append({**m, "family": fam, "model": model, "run": None, "note": "no DEV cache"}); continue
            fw, t, labs = DCC.load_fr(p)
            rr = [(l, a, b) for l, a, b in RUNS[model](fw, t, labs) if canonical(l) == fam
                  and m["at"] - EARLY <= a <= m["at"] + LATE]
            allow.append({**m, "family": fam, "model": model, "run": rr[0] if rr else None})
    print(f"RELIABLE would allow (list only): {sum(1 for x in allow if x['run'])} of {len(led)} needed misses have a raw run "
          f"of a RELIABLE family at the onset")
    for x in allow:
        print(f"   {x['part']:4s} {x['clip']} {x['sound']} @{x['at']} [{x['model']}] run {x['run']}")
    res = {"base": Bm, "lists": {m: {"UNRELIABLE": sorted(UNREL[m]), "RELIABLE": sorted(REL[m])} for m in UNREL},
           "rows": X, "dropped": dropped, "hits_lost": lost, "untouched": dict(why_n), "exception": dict(named_n),
           "main_rule": old, "fewer_pictures": few, "verdict": verdict, "reliable_would_allow": allow}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"prior": prior, "screen": screen}[sys.argv[1]]()
