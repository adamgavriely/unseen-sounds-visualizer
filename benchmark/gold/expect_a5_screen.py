"""Round 42b EXPECT-A5 (docs/prereg_round13_detector_push.md "Round 42b EXPECT-A5"): EXPECT-A4 exactly as frozen, restricted to the
families the held-out 415 vouch for (Round 42 precision >= 0.8, n >= 3). Nothing re-run: DEV = the kept pictures of expect_a4_screen.json,
TEST = the added pictures of expect_test.json, each kept iff its family is in FAMILIES. CPU only.

    TG_ARMS=SHIP8 python benchmark/gold/expect_a5_screen.py        # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

FAMILIES = {"Laughter", "Crowd", "Vehicle", "Dog", "Bell", "Cough", "Toilet flush"}     # frozen in the prereg (held-out >= 0.8, n >= 3)
A4 = _ROOT / "benchmark" / "gold" / "expect_a4_screen.json"
TJ = _ROOT / "benchmark" / "gold" / "expect_test.json"
OUT = _ROOT / "benchmark" / "gold" / "expect_a5_screen.json"
PIC_LEN = 2.0
BASE_DEV = {"hits": 28, "wrong": 21, "cost": 2.282}
BASE_TEST = {"hits": 23, "misses": 42, "wrong": 29, "cost": 2.568}


def check_list():
    R = json.loads((_ROOT / "benchmark" / "gold" / "heldout_a4_screen.json").read_text(encoding="utf-8"))["per_family_kept"]
    sel = {f for f, x in R.items() if x["precision"] >= 0.8 and x["n"] >= 3}
    assert sel == FAMILIES, (sel, FAMILIES)


def dev():
    from benchmark.gold import expect_screen as E
    from benchmark.gold import btp_screen as B
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    kept = [p for p in json.loads(A4.read_text(encoding="utf-8"))["pictures"] if p["outcome"] == "kept"]
    assert len(kept) == 8, len(kept)
    P = E.parts()
    base = {"dev": [], "dev2": []}; rows = {"dev": [], "dev2": []}; added = []; lost = []; skipped = []
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, g, pics in P:
        old = [p[:3] for p in pics]; new = list(old); adds = []
        for d in [a for a in kept if a["clip"] == st and a["part"] == pt]:
            if d["family"] in FAMILIES:
                new.append((d["family"], float(d["start"]), float(d["start"]) + PIC_LEN)); adds.append(d)
            else:
                skipped.append({"part": pt, "clip": st, "family": d["family"], "start": d["start"], "class_40e": d["class_40d"]})
        r0 = S.score_clip(g, old); r1 = S.score_clip(g, new)
        base[pt].append(r0); rows[pt].append(r1)
        if adds:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for d in adds:
                added.append({"part": pt, "clip": st, "family": d["family"], "start": d["start"], "dasm": d["dasm_onset"],
                              "class": cl[(d["family"], round(float(d["start"]), 3))], "clip_before": oc(r0), "clip_after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    Bm = {k: B.summ(base["dev"] + base["dev2"]) if k == "merged" else B.summ(base[k]) for k in ("merged", "dev", "dev2")}
    X = {k: B.summ(rows["dev"] + rows["dev2"]) if k == "merged" else B.summ(rows[k]) for k in ("merged", "dev", "dev2")}
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE_DEV["hits"], BASE_DEV["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE_DEV["cost"]) < 0.001, Bm["merged"]
    n = len(P)
    c0 = {w: G.cost_w(Bm["merged"], n, w) for w in (1, 2)}; c1 = {w: G.cost_w(X["merged"], n, w) for w in (1, 2)}
    ln = sum(x[2] for x in lost)
    main_go = X["merged"]["hits"] >= BASE_DEV["hits"] and not lost and c1[2] < c0[2]
    few = c1[2] < c0[2] and X["merged"]["wrong"] <= BASE_DEV["wrong"] - 3 * ln and ln <= 3
    print(f"BASE SHIP8 DEV: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    print(f"EXPECT-A5 DEV:  merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  added {len(added)} "
          f"skipped {len(skipped)}  hits lost {lost}")
    print(f"  main rule: {'GO' if main_go else 'STOP'}   fewer-pictures: {'GO' if few else 'STOP'}   cost w=2 {c0[2]:.3f} -> {c1[2]:.3f}, "
          f"w=1 {c0[1]:.3f} -> {c1[1]:.3f}")
    for a in added:
        print(f"   + {a['part']:4s} {a['clip']} {a['family']} @{a['start']} (dasm {a['dasm']}): {a['class']}  clip {a['clip_before']} -> {a['clip_after']}")
    for s in skipped:
        print(f"   - {s['part']:4s} {s['clip']} {s['family']} @{s['start']} [{s['class_40e']}] not in the list")
    return {"base": Bm, "rows": X, "cost": {str(w): [round(c0[w], 3), round(c1[w], 3)] for w in (1, 2)}, "added": added,
            "skipped": skipped, "hits_lost": lost, "main_GO": main_go, "fewer_pictures_GO": few}


def test():
    from benchmark.gold import expect_test as ET
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import final_test as FT
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import tagger_prep as TP
    from benchmark.gold.cross_group import classify
    P = ET.parts()
    S.load_gold = FT._REAL_LOAD_GOLD
    gold1 = S.load_gold([FT.G / "annotations" / "gold_AG.json"])
    stems1 = [st for pt, st, _p, _c in P if pt == "test"]; stems2 = [st for pt, st, _p, _c in P if pt == "test2"]
    o2 = TP.out("test2")
    dg = json.loads(TP.TAGGER_GOLD.read_text(encoding="utf-8"))
    dg["clips"] = [c for c in dg.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in set(stems2)]
    tmp = o2 / "test2_gold_only.json"; DCC.dump(tmp, dg)
    gold2 = S.load_gold([tmp])
    gold = {("test", st): gold1[st] for st in stems1} | {("test2", st): gold2[st] for st in stems2 if st in gold2}
    A = json.loads(TJ.read_text(encoding="utf-8"))["added"]
    assert len(A) == 8, len(A)
    r0s, r1s, c0, c1, added, skipped, lost = [], [], [], [], [], [], []
    pr = {"base": {"test": [], "test2": []}, "new": {"test": [], "test2": []}}
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, pics, _c in P:
        if (pt, st) not in gold:
            continue
        g = gold[(pt, st)]; new = list(pics); adds = []
        for a in [x for x in A if x["clip"] == st and x["part"] == pt]:
            if a["family"] in FAMILIES:
                new.append((a["family"], float(a["start"]), float(a["start"]) + PIC_LEN)); adds.append(a)
            else:
                skipped.append({"part": pt, "clip": st, "family": a["family"], "start": a["start"], "class_40e": a["class"]})
        r0 = S.score_clip(g, pics); r1 = S.score_clip(g, new)
        r0s.append(r0); r1s.append(r1); c0.append(DCC.clip_cost(r0)); c1.append(DCC.clip_cost(r1))
        pr["base"][pt].append(r0); pr["new"][pt].append(r1)
        if adds:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for a in adds:
                added.append({"part": pt, "clip": st, "family": a["family"], "start": a["start"], "dasm": a["dasm"],
                              "class": cl[(a["family"], round(float(a["start"]), 3))], "clip_before": oc(r0), "clip_after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    m0, m1 = DCC.metrics(r0s), DCC.metrics(r1s)
    n = len(r0s)
    cw = lambda m, w: (4 * m["misses"] + w * m["visible"] + 2 * m["cross"] + 2 * m["phantom"]) / n
    assert (m0["hits"], m0["misses"], m0["wrong"]) == (BASE_TEST["hits"], BASE_TEST["misses"], BASE_TEST["wrong"]), m0
    assert abs(m0["viewer_cost"] - BASE_TEST["cost"]) < 0.001, m0["viewer_cost"]
    d_ = DCC.boot(np.subtract(c1, c0))
    fmt = lambda m: f"{m['hits']}/{m['hits'] + m['misses']} wrong {m['wrong']} ({m['visible']}/{m['cross']}/{m['phantom']}) cost {m['viewer_cost']:.3f}"
    print(f"BASE SHIP8 TEST: {fmt(m0)}  (old TEST {fmt(DCC.metrics(pr['base']['test']))} | tagger TEST {fmt(DCC.metrics(pr['base']['test2']))})")
    print(f"EXPECT-A5 TEST:  {fmt(m1)}  (old TEST {fmt(DCC.metrics(pr['new']['test']))} | tagger TEST {fmt(DCC.metrics(pr['new']['test2']))})")
    print(f"  cost w=2 {cw(m0, 2):.3f} -> {cw(m1, 2):.3f}; w=1 {cw(m0, 1):.3f} -> {cw(m1, 1):.3f}; d cost {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}] "
          f"one-sided p {d_[3]:.3f}; added {len(added)} skipped {len(skipped)} hits lost {lost}")
    for a in added:
        print(f"   + {a['part']:5s} {a['clip']} {a['family']} @{a['start']} (dasm {a['dasm']}): {a['class']}  clip {a['clip_before']} -> {a['clip_after']}")
    for s in skipped:
        print(f"   - {s['part']:5s} {s['clip']} {s['family']} @{s['start']} [{s['class_40e']}] not in the list")
    return {"clips": n, "base": m0, "new": m1, "parts": {k: {pt: DCC.metrics(v[pt]) for pt in v} for k, v in pr.items()},
            "cost": {"w2": [round(cw(m0, 2), 3), round(cw(m1, 2), 3)], "w1": [round(cw(m0, 1), 3), round(cw(m1, 1), 3)]},
            "delta_cost_vs_base": d_, "added": added, "skipped": skipped, "hits_lost": lost}


def main():
    """dev and test each in their own process: the DEV loaders (tagger_prep.configure("dev2")) stub score_per_sound.load_gold before
    final_test can capture the real one, which breaks the TEST loader in the same process"""
    import subprocess
    check_list()
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("dev", "test"):
        res = {"dev": dev, "test": test}[step]()
        (OUT.parent / f"expect_a5_{step}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        return
    for st in ("dev", "test"):
        subprocess.run([sys.executable, __file__, st], check=True, cwd=str(_ROOT))
    res = {"round": "Round 42b EXPECT-A5", "families": sorted(FAMILIES)}
    for st in ("dev", "test"):
        res[st] = json.loads((OUT.parent / f"expect_a5_{st}.json").read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
