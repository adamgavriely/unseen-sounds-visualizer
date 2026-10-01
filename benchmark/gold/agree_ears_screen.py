"""Round 45 AGREE-EARS (docs/prereg_round13_detector_push.md "Round 45 AGREE-EARS"): an EXPECT-A4 picture is kept only if a second,
independent ear (Audio Flamingo Next) also names its family on the whole clip, same question (expect_a_screen.LIST_Q), same frozen map
(expect_a_screen.items_of / map_item). Secondary: AFN yes/no on the onset cut [onset - 1, onset + 3]. Nothing else re-run: DEV = the 8
kept pictures of expect_a4_screen.json, TEST = the 8 added of expect_test.json, held-out = the 92 kept detections of
heldout_a4_screen.json. Gold is read only inside the dev / test score steps (as expect_a5_screen.py).

    TG_ARMS=SHIP8 python benchmark/gold/agree_ears_screen.py listen        # GPU msproj: AFN whole-clip lists + yes/no -> agree_ears/
    TG_ARMS=SHIP8 python benchmark/gold/agree_ears_screen.py score         # CPU: heldout, dev, test (separate processes), pooled
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

GOLD = _ROOT / "benchmark" / "gold"
DIR = GOLD / "agree_ears"
OUT = GOLD / "agree_ears_screen.json"
WORKD = _ROOT / "data" / "work"
WAV = {"dev": WORKD / "devcand" / "wav16", "dev2": WORKD / "r13dev2" / "wav16", "test": WORKD / "r13test" / "wav16",
       "test2": WORKD / "r13test2" / "wav16", "heldout": GOLD / "heldout_a4" / "wav16"}
A4 = GOLD / "expect_a4_screen.json"
TJ = GOLD / "expect_test.json"
HJ = GOLD / "heldout_a4_screen.json"
MAX_NEW, PIC_LEN, SR = 96, 2.0, 16000
PRE, POST = 1.0, PIC_LEN + 1.0                                    # yes/no cut = [onset - 1, onset + 3]
BASE_DEV = {"hits": 28, "wrong": 21, "cost": 2.282}
BASE_TEST = {"hits": 23, "misses": 42, "wrong": 29, "cost": 2.568}
HELD_PREC, HELD_KEEP = 0.85, 0.5                                   # reading rule for the held-out 415 (prereg)


# ---------------------------------------------------------------- the three sets: clips (part, stem) and candidate pictures
def clips(which):
    """[(part, stem)] — DEV / TEST: the clips Omni listened to (expect_a / expect_test listen caches); held-out: the 415 ids"""
    if which == "heldout":
        H = json.loads((GOLD / "audioset_heldout.json").read_text(encoding="utf-8"))
        out = [("heldout", c["id"]) for c in H["clips"]]
        assert len(out) == 415, len(out)
        return out
    d = GOLD / ("expect_a" if which == "dev" else "expect_test") / "listen"
    out = []
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append((r["part"], r["clip"]))
    assert len(out) == {"dev": 71, "test": 88}[which], (which, len(out))
    return out


def cands(which):
    """[(part, stem, family, onset)] — the pictures this round selects among (classes are NOT read here)"""
    if which == "dev":
        P = json.loads(A4.read_text(encoding="utf-8"))["pictures"]
        out = [(p["part"], p["clip"], p["family"], float(p["start"])) for p in P if p["outcome"] == "kept"]
        assert len(out) == 8, len(out)
    elif which == "test":
        A = json.loads(TJ.read_text(encoding="utf-8"))["added"]
        out = [(a["part"], a["clip"], a["family"], float(a["start"])) for a in A]
        assert len(out) == 8, len(out)
    else:
        D = json.loads(HJ.read_text(encoding="utf-8"))["detections"]
        out = [("heldout", d["clip"], d["family"], float(d["onset"])) for d in D if d["outcome"] == "kept"]
        assert len(out) == 92, len(out)
    return out


def ykey(fam, onset):
    return f"{fam}@{onset:.2f}"


def cache_file(which, st):
    return DIR / which / f"{st}.json"


# ---------------------------------------------------------------- listen (GPU msproj)
def cmd_listen():
    import soundfile as sf
    from benchmark.gold import listener_afnext as AF
    from benchmark.gold import expect_a_screen as A
    M = None
    for which in ("heldout", "dev", "test"):
        cl = clips(which)
        need = {}
        for pt, st, fam, on in cands(which):
            need.setdefault(st, []).append((fam, on))
        d = DIR / which; d.mkdir(parents=True, exist_ok=True)
        todo = []
        for pt, st in cl:
            f = cache_file(which, st)
            if f.exists():
                r = json.loads(f.read_text(encoding="utf-8"))
                if all(ykey(fam, on) in r.get("yn", {}) for fam, on in need.get(st, [])):
                    continue
            todo.append((pt, st))
        print(f"[{which}] {len(cl)} clips, {len(todo)} to listen, {sum(len(v) for v in need.values())} yes/no cuts", flush=True)
        if not todo:
            continue
        M = M or AF.load_model()
        t0 = time.time()
        for n, (pt, st) in enumerate(todo, 1):
            w, sr = sf.read(str(WAV[pt] / f"{st}.wav"), dtype="float32")
            assert sr == SR and w.ndim == 1, (st, sr, w.shape)
            text = M["gen"](w, A.LIST_Q, MAX_NEW)
            items = A.items_of(text)
            fams = []
            for it in items:
                f = A.map_item(it)
                if f and f not in fams:
                    fams.append(f)
            yn = {}
            for fam, on in need.get(st, []):
                a, b = max(0.0, on - PRE), min(len(w) / SR, on + POST)
                seg = w[int(a * SR):max(int(b * SR), int(a * SR) + SR)]
                s, top = M["yesno"](seg, fam)
                yn[ykey(fam, on)] = {"margin": round(s, 4), "top1": top, "cut": [round(a, 2), round(b, 2)]}
            rec = {"part": pt, "clip": st, "seconds": round(len(w) / SR, 2), "prompt": A.LIST_Q, "max_new": MAX_NEW,
                   "decode": "greedy, no repetition penalty (listener_afnext.load_model gen)", "model": AF.AFN, "text": text,
                   "items": items, "families": fams, "yn_question": "benchmark.listener_round.QUESTION on [onset-1, onset+3]", "yn": yn}
            cache_file(which, st).write_text(json.dumps(rec, indent=1), encoding="utf-8")
            if n <= 3 or n % 50 == 0:
                print(f"[{which}] {n}/{len(todo)} {st} ({time.time() - t0:.0f} s): {text!r} -> {fams} yn {yn}", flush=True)
        print(f"[{which}] done {len(todo)} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- selection (no gold)
def verdicts(which):
    """{(part, stem, family, onset): {"agree": bool, "yn": margin|None, "agree_yn": bool, "afn_families": [...]}}"""
    out = {}
    for pt, st, fam, on in cands(which):
        r = json.loads(cache_file(which, st).read_text(encoding="utf-8"))
        agree = fam in r["families"]
        y = r["yn"].get(ykey(fam, on))
        m = None if y is None else float(y["margin"])
        out[(pt, st, fam, on)] = {"agree": agree, "yn": m, "agree_yn": bool(agree and m is not None and m > 0),
                                  "afn_families": r["families"], "afn_items": r["items"]}
    return out


VARIANTS = ("agree", "agree_yn")


# ---------------------------------------------------------------- held-out 415 (classes recorded in heldout_a4_screen.json)
def heldout():
    D = [d for d in json.loads(HJ.read_text(encoding="utf-8"))["detections"] if d["outcome"] == "kept"]
    V = verdicts("heldout")
    base = {"n": len(D), "correct": sum(d["class"] == "correct" for d in D)}
    base["precision"] = round(base["correct"] / base["n"], 3)
    assert (base["n"], base["correct"]) == (92, 74), base
    res = {"base": base, "variants": {}, "per_family": {}, "detections": []}
    for d in D:
        v = V[("heldout", d["clip"], d["family"], float(d["onset"]))]
        res["detections"].append({"clip": d["clip"], "family": d["family"], "onset": d["onset"], "class": d["class"], **v})
    for var in VARIANTS:
        kept = [x for x in res["detections"] if x[var]]
        c = sum(x["class"] == "correct" for x in kept)
        prec = c / len(kept) if kept else 0.0
        vouched = prec >= HELD_PREC and c >= HELD_KEEP * base["correct"]
        res["variants"][var] = {"n": len(kept), "correct": c, "precision": round(prec, 3), "correct_kept_share": round(c / base["correct"], 3),
                                "wrong_removed": (base["n"] - base["correct"]) - (len(kept) - c), "vouched": vouched}
        print(f"HELD-OUT {var}: kept {len(kept)}/92, correct {c}/74 survive, precision {prec:.3f} (base 0.804), "
              f"wrong removed {res['variants'][var]['wrong_removed']}/18 -> {'VOUCHED' if vouched else 'not vouched'}")
    fams = sorted({x["family"] for x in res["detections"]})
    for f in fams:
        xs = [x for x in res["detections"] if x["family"] == f]
        res["per_family"][f] = {"n": len(xs), "correct": sum(x["class"] == "correct" for x in xs),
                                "agree": sum(x["agree"] for x in xs), "agree_correct": sum(x["agree"] and x["class"] == "correct" for x in xs)}
    return res


# ---------------------------------------------------------------- DEV (gold read inside, as expect_a5_screen.dev)
def dev():
    from benchmark.gold import expect_screen as E
    from benchmark.gold import btp_screen as B
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold.cross_group import classify
    kept = [p for p in json.loads(A4.read_text(encoding="utf-8"))["pictures"] if p["outcome"] == "kept"]
    V = verdicts("dev")
    P = E.parts()
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    res = {"pictures": [], "variants": {}}
    costs = {"base": [], "a4": []} | {v: [] for v in VARIANTS}
    rows = {"a4": {"dev": [], "dev2": []}} | {v: {"dev": [], "dev2": []} for v in VARIANTS}
    base = {"dev": [], "dev2": []}
    lost = {v: [] for v in ("a4",) + VARIANTS}
    for pt, st, g, pics in P:
        old = [p[:3] for p in pics]
        r0 = S.score_clip(g, old); base[pt].append(r0); costs["base"].append(DCC.clip_cost(r0))
        mine = [d for d in kept if d["clip"] == st and d["part"] == pt]
        sel = {"a4": mine} | {v: [d for d in mine if V[(pt, st, d["family"], float(d["start"]))][v]] for v in VARIANTS}
        for var, ds in sel.items():
            new = old + [(d["family"], float(d["start"]), float(d["start"]) + PIC_LEN) for d in ds]
            r1 = S.score_clip(g, new); rows[var][pt].append(r1); costs[var].append(DCC.clip_cost(r1))
            if r1["hit"] < r0["hit"]:
                lost[var].append((pt, st, r0["hit"] - r1["hit"]))
        if mine:
            new = old + [(d["family"], float(d["start"]), float(d["start"]) + PIC_LEN) for d in mine]
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for d in mine:
                v = V[(pt, st, d["family"], float(d["start"]))]
                res["pictures"].append({"part": pt, "clip": st, "family": d["family"], "start": d["start"], "dasm": d["dasm_onset"],
                                        "class": cl[(d["family"], round(float(d["start"]), 3))], **v})
    Bm = {k: B.summ(base["dev"] + base["dev2"]) if k == "merged" else B.summ(base[k]) for k in ("merged", "dev", "dev2")}
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE_DEV["hits"], BASE_DEV["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE_DEV["cost"]) < 0.001, Bm["merged"]
    n = len(P)
    c0 = {w: G.cost_w(Bm["merged"], n, w) for w in (1, 2)}
    print(f"BASE SHIP8 DEV: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    res["base"] = Bm
    for var in ("a4",) + VARIANTS:
        X = {k: B.summ(rows[var]["dev"] + rows[var]["dev2"]) if k == "merged" else B.summ(rows[var][k]) for k in ("merged", "dev", "dev2")}
        c1 = {w: G.cost_w(X["merged"], n, w) for w in (1, 2)}
        ln = sum(x[2] for x in lost[var])
        main_go = X["merged"]["hits"] >= BASE_DEV["hits"] and not lost[var] and c1[2] < c0[2]
        few = c1[2] < c0[2] and X["merged"]["wrong"] <= BASE_DEV["wrong"] - 3 * ln and ln <= 3
        kept_n = sum(1 for p in res["pictures"] if var == "a4" or p[var])
        print(f"{var.upper():9s} DEV: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  kept {kept_n}/8  "
              f"hits lost {lost[var]}  main rule: {'GO' if main_go else 'STOP'}  fewer-pictures: {'GO' if few else 'STOP'}  "
              f"cost w=2 {c0[2]:.3f} -> {c1[2]:.3f}, w=1 {c0[1]:.3f} -> {c1[1]:.3f}")
        res["variants"][var] = {"rows": X, "cost": {str(w): [round(c0[w], 3), round(c1[w], 3)] for w in (1, 2)}, "kept": kept_n,
                                "hits_lost": lost[var], "main_GO": main_go, "fewer_pictures_GO": few}
    for p in res["pictures"]:
        print(f"   {p['part']:4s} {p['clip']} {p['family']} @{p['start']} [{p['class']}] agree {p['agree']} yn {p['yn']} "
              f"-> {'KEPT' if p['agree'] else 'dropped'}  AFN: {p['afn_families']}")
    res["clip_costs"] = costs
    return res


# ---------------------------------------------------------------- TEST (gold read inside, as expect_a5_screen.test)
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
    V = verdicts("test")
    vars_ = ("a4",) + VARIANTS
    rows = {v: [] for v in ("base",) + vars_}; costs = {v: [] for v in ("base",) + vars_}
    parts = {v: {"test": [], "test2": []} for v in ("base",) + vars_}
    lost = {v: [] for v in vars_}
    res = {"pictures": [], "variants": {}}
    for pt, st, pics, _c in P:
        if (pt, st) not in gold:
            continue
        g = gold[(pt, st)]
        r0 = S.score_clip(g, pics); rows["base"].append(r0); costs["base"].append(DCC.clip_cost(r0)); parts["base"][pt].append(r0)
        mine = [a for a in A if a["clip"] == st and a["part"] == pt]
        sel = {"a4": mine} | {v: [a for a in mine if V[(pt, st, a["family"], float(a["start"]))][v]] for v in VARIANTS}
        for var, ds in sel.items():
            new = list(pics) + [(a["family"], float(a["start"]), float(a["start"]) + PIC_LEN) for a in ds]
            r1 = S.score_clip(g, new); rows[var].append(r1); costs[var].append(DCC.clip_cost(r1)); parts[var][pt].append(r1)
            if r1["hit"] < r0["hit"]:
                lost[var].append((pt, st, r0["hit"] - r1["hit"]))
        if mine:
            new = list(pics) + [(a["family"], float(a["start"]), float(a["start"]) + PIC_LEN) for a in mine]
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for a in mine:
                v = V[(pt, st, a["family"], float(a["start"]))]
                res["pictures"].append({"part": pt, "clip": st, "family": a["family"], "start": a["start"], "dasm": a["dasm"],
                                        "class": cl[(a["family"], round(float(a["start"]), 3))], **v})
    m0 = DCC.metrics(rows["base"]); n = len(rows["base"])
    assert (m0["hits"], m0["misses"], m0["wrong"]) == (BASE_TEST["hits"], BASE_TEST["misses"], BASE_TEST["wrong"]), m0
    assert abs(m0["viewer_cost"] - BASE_TEST["cost"]) < 0.001, m0["viewer_cost"]
    cw = lambda m, w: (4 * m["misses"] + w * m["visible"] + 2 * m["cross"] + 2 * m["phantom"]) / n
    fmt = lambda m: f"{m['hits']}/{m['hits'] + m['misses']} wrong {m['wrong']} ({m['visible']}/{m['cross']}/{m['phantom']}) cost {m['viewer_cost']:.3f}"
    print(f"BASE SHIP8 TEST: {fmt(m0)}  (old TEST {fmt(DCC.metrics(parts['base']['test']))} | tagger TEST {fmt(DCC.metrics(parts['base']['test2']))})")
    res["clips"] = n; res["base"] = m0
    for var in vars_:
        m1 = DCC.metrics(rows[var])
        d_ = DCC.boot(np.subtract(costs[var], costs["base"]))
        kept_n = sum(1 for p in res["pictures"] if var == "a4" or p[var])
        print(f"{var.upper():9s} TEST: {fmt(m1)}  (old TEST {fmt(DCC.metrics(parts[var]['test']))} | tagger TEST {fmt(DCC.metrics(parts[var]['test2']))})  "
              f"kept {kept_n}/8; cost w=2 {cw(m0, 2):.3f} -> {cw(m1, 2):.3f}; w=1 {cw(m0, 1):.3f} -> {cw(m1, 1):.3f}; "
              f"d cost {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}] one-sided p {d_[3]:.3f}; hits lost {lost[var]}")
        res["variants"][var] = {"metrics": m1, "parts": {pt: DCC.metrics(parts[var][pt]) for pt in parts[var]},
                                "cost": {"w2": [round(cw(m0, 2), 3), round(cw(m1, 2), 3)], "w1": [round(cw(m0, 1), 3), round(cw(m1, 1), 3)]},
                                "delta_cost_vs_base": d_, "kept": kept_n, "hits_lost": lost[var]}
    for p in res["pictures"]:
        print(f"   {p['part']:5s} {p['clip']} {p['family']} @{p['start']} [{p['class']}] agree {p['agree']} yn {p['yn']} "
              f"-> {'KEPT' if p['agree'] else 'dropped'}  AFN: {p['afn_families']}")
    res["clip_costs"] = costs
    return res


def pooled(dv, tv):
    """one paired clip bootstrap over DEV + TEST (159 clips) of d cost vs base, for EXPECT-A4 and each variant (secondary)"""
    from benchmark.gold import dev_candidates_check as DCC
    out = {}
    for var in ("a4",) + VARIANTS:
        d = np.subtract(dv["clip_costs"][var] + tv["clip_costs"][var], dv["clip_costs"]["base"] + tv["clip_costs"]["base"])
        out[var] = {"clips": int(len(d)), "boot": DCC.boot(d)}
        b = out[var]["boot"]
        print(f"POOLED DEV+TEST {var.upper():9s} ({len(d)} clips): d cost {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] one-sided p {b[3]:.3f}")
    return out


def main():
    import subprocess
    step = sys.argv[1] if len(sys.argv) > 1 else "score"
    if step == "listen":
        cmd_listen(); return
    if step in ("dev", "test"):
        r = {"dev": dev, "test": test}[step]()
        (DIR / f"{step}_score.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        return
    res = {"round": "Round 45 AGREE-EARS", "heldout": heldout()}
    for st in ("dev", "test"):
        subprocess.run([sys.executable, __file__, st], check=True, cwd=str(_ROOT))
        res[st] = json.loads((DIR / f"{st}_score.json").read_text(encoding="utf-8"))
    res["pooled"] = pooled(res["dev"], res["test"])
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
