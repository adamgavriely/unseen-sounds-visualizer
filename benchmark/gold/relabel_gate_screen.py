"""Round 39 RELABEL-GATE screen (docs/prereg_round13_detector_push.md "Round 39 RELABEL-GATE"), on the saved SHIP8 proposed
pictures of merged DEV, scored on gold_AG.json. For every placed NON-rescued picture of family A: its P1 cut's open-inventory
listener answers (qwen_fams / af_fams, _p1v4_lists); if both name a depictable B (not same_family with A) and at most one names
A, the picture is renamed B; then the shipped visibility gate is asked about B on that picture's frames (cached gate_gold votes
when B matches a gold sound at that time, else a live call) -- seen -> dropped, not seen -> kept as B. Nothing in src/ or
config.py is edited. Base (SHIP8) must reproduce 28/58, 21 (6/13/2), 2.282 first.

    TG_ARMS=SHIP8 python benchmark/gold/relabel_gate_screen.py scan     # CPU, from ~/MscProj_tg: relabels, cached gates, GPU list
    TG_ARMS=SHIP8 python benchmark/gold/relabel_gate_screen.py gate     # GPU: live gate for the pictures scan left open
    TG_ARMS=SHIP8 python benchmark/gold/relabel_gate_screen.py score    # CPU: rescoring + verdict
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
import config
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical
from src.types import AudioEvent
from src.stage4_audio_event_detection import _p1v4_lists

B.ARM = "SHIP8"
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}
G = _ROOT / "benchmark" / "gold"
GATE_DIR = G / "gate_gold" / "Qwen38-27B"
MODEL = "Qwen/Qwen3.8-27B"
OUT = G / "relabel_gate_screen.json"
DEPICTABLE = set(json.loads((G / "depictable_vocab.json").read_text(encoding="utf-8"))["families"])


def passes(X, lost_parts, lost_n):
    """dbr_screen.passes against the SHIP8 base (28 / 21 / 2.282), not dbr_screen's SHIP7 base"""
    gain = X["hits"] - BASE["hits"]
    old = (X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"]
           and not lost_parts)
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * lost_n and lost_n <= 3
    return old, few


def summ_cls(cl):
    return "h{} v{} c{} p{} d{}".format(*(sum(1 for x in cl if x[3] == k) for k in ("hit", "visible", "cross", "phantom", "dup")))


def p1_answer(stage4_rows, st, lab, a, b):
    """(qwen_fams, af_fams) of the answered same-family stage-4 row nearest the picture start, or None"""
    fam = canonical(lab)
    config._CURRENT_CLIP = st
    best = None
    for r in stage4_rows:
        if canonical(r["label"]) != fam or r["end"] < a - 0.1 or r["start"] > b + 0.1:
            continue
        q, af = _p1v4_lists(AudioEvent(r["label"], r["start"], r["end"], r["conf"], rescued=r.get("rescued", False)))
        if q is None or af is None:
            continue
        d = abs(r["start"] - a)
        if best is None or d < best[0]:
            best = (d, r, q, af)
    return None if best is None else best[1:]


def relabel_target(lab, q, af):
    """B per the prereg (first qwen-order candidate in both lists) or None"""
    fam = canonical(lab)
    names_a = [sum(1 for x in L if S.same_family(x, fam) or canonical(x) == fam) > 0 for L in (q, af)]
    if sum(names_a) > 1:
        return None, names_a
    afs = set(af)
    cands = [x for x in q if x in afs and not (S.same_family(x, fam) or canonical(x) == fam)
             and x in DEPICTABLE and S._specific(x)]
    return (cands[0] if cands else None), names_a


def _stretch_seen(st):
    votes = [st.get("name"), st.get("ab"), st.get("desc")]
    return sum(v is True for v in votes) > sum(v is False for v in votes)


def cached_gate(stem, b_fam, a):
    """shipped majority on the cached gold votes of a gold sound of B at the picture start, or None if not cached"""
    f = GATE_DIR / f"{stem}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    hits = [s for s in d["sounds"] if S.same_family(b_fam, s["label"]) and s["start"] - S.EARLY <= a <= s["end"]]
    if not hits:
        return None
    s = min(hits, key=lambda s: abs(s["start"] - a))
    sts = [x for x in s["stretches"] if x["start"] <= a + 1.0 and x["end"] >= a - 1.0] or s["stretches"]
    return {"seen": all(_stretch_seen(x) for x in sts), "gold": [s["label"], s["start"], s["end"]],
            "votes": [[x.get("name"), x.get("ab"), x.get("desc")] for x in sts], "source": "cached"}


def scan():
    P = B.parts()
    s4 = {pt: json.loads(CG.PARTS[pt]["stage4"].read_text(encoding="utf-8"))["arms"]["SHIP8|proposed"] for pt in CG.PARTS}
    rows = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        rows[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    BASE["cost"] = Bm["merged"]["cost"]
    rel = []; tally = {"placed": 0, "rescued": 0, "no answered cut": 0, "both name A": 0, "no candidate": 0, "relabelled": 0}
    for pt, st, g, pics in P:
        config.RELABEL_P1V4 = str(G / f"{CG.PARTS[pt]['lis']}_listener_p1v4.json")
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        for lab, a, b, resc in pics:
            tally["placed"] += 1
            if resc:
                tally["rescued"] += 1; continue
            ans = p1_answer(s4[pt].get(st, []), st, lab, a, b)
            if ans is None:
                tally["no answered cut"] += 1; continue
            r, q, af = ans
            b_fam, names_a = relabel_target(lab, q, af)
            if b_fam is None:
                tally["both name A" if sum(names_a) > 1 else "no candidate"] += 1; continue
            tally["relabelled"] += 1
            gate = cached_gate(st, b_fam, a)
            rel.append({"part": pt, "clip": st, "label": lab, "A": canonical(lab), "B": b_fam, "start": round(a, 2),
                        "end": round(b, 2), "before": before[(lab, round(a, 3))], "cut": [r["start"], r["end"]],
                        "qwen": q, "af": af, "names_A": names_a, "gate": gate})
    print(f"{tally}")
    for x in rel:
        print(f"   {x['part']:4s} {x['clip']} {x['A']} -> {x['B']} @{x['start']} ({x['before']}) gate={x['gate'] and x['gate']['seen']} "
              f"{'cached' if x['gate'] else 'NEEDS GPU'} | qwen {x['qwen']} af {x['af']}")
    OUT.write_text(json.dumps({"base": Bm, "tally": tally, "relabels": rel}, indent=1, default=float), encoding="utf-8")
    print(f"{sum(1 for x in rel if x['gate'] is None)} need a live gate; {OUT}")


def gate(device="cuda"):
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    from benchmark.gold.detector_dry import clip_path
    from benchmark.gold import tagger_prep as T
    res = json.loads(OUT.read_text(encoding="utf-8"))
    todo = [x for x in res["relabels"] if x["gate"] is None]
    if not todo:
        print("nothing to gate"); return
    mdl, proc = reason._load(MODEL, device)
    for x in todo:
        p = clip_path(x["clip"]) if x["part"] == "dev" else None
        if p is None:
            cands = [q for q in T.clips(x["part"]).glob(x["clip"] + ".*")] if hasattr(T, "clips") else []
            cands = cands or list((_ROOT / "data" / "input" / f"tagger_{x['part']}").glob(x["clip"] + ".*")) \
                or list((_ROOT / "data" / "input" / "tagger_set").glob(x["clip"] + ".*"))
            p = cands[0] if cands else None
        if p is None:
            print("missing clip", x["clip"]); continue
        a = x["start"]; n = 6; lo, hi = a - 1.0, a + 1.0
        times = [max(0.0, lo + (hi - lo) * t / (n - 1)) for t in range(n)]
        seen, named = reason._sound_is_visible(x["B"], _sample_frames_at(p, times), mdl, proc, device)
        x["gate"] = {"seen": bool(seen), "gold": None, "votes": [[reason.LAST_VOTES.get(k) for k in ("name", "ab", "desc")]],
                     "named": reason.LAST_VOTES.get("named"), "source": "live"}
        print(x["clip"], x["B"], x["gate"], flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def score():
    res = json.loads(OUT.read_text(encoding="utf-8"))
    BASE["cost"] = res["base"]["merged"]["cost"]                       # exact base cost, not the rounded 2.282
    rel = {(x["part"], x["clip"], x["label"], x["start"]): x for x in res["relabels"]}
    assert all(x["gate"] is not None for x in rel.values()), "run `gate` first"
    P = B.parts()
    rows = {"dev": [], "dev2": []}; lost = []
    for pt, st, g, pics in P:
        new = []
        for lab, a, b, resc in pics:
            x = rel.get((pt, st, lab, round(a, 2)))
            if x is None:
                new.append((lab, a, b))
            elif not x["gate"]["seen"]:
                new.append((x["B"], a, b))
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, new); rows[pt].append(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
        cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
        for x in rel.values():
            if x["part"] == pt and x["clip"] == st:
                x["outcome"] = "dropped (seen)" if x["gate"]["seen"] else cl[(x["B"], round(x["start"], 3))]
                x["clip_before"], x["clip_after"] = summ_cls(classify(g, [p[:3] for p in pics])), summ_cls(classify(g, new))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    ln = sum(x[2] for x in lost)
    old, few = passes(X["merged"], lost, ln)
    verdict = "GO (main rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
    print(f"RELABEL-GATE: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  hits lost {lost} -> {verdict}")
    for x in rel.values():
        print(f"   {x['part']:4s} {x['clip']} {x['A']} -> {x['B']} @{x['start']} gate {x['gate']['source']} seen={x['gate']['seen']}: "
              f"{x['before']} -> {x['outcome']}   clip {x['clip_before']} -> {x['clip_after']}")
    res.update({"rows": X, "hits_lost": lost, "main_rule": old, "fewer_pictures": few, "verdict": verdict})
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"scan": scan, "gate": gate, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "scan"]()
