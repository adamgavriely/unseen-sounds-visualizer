"""Round 61 SCENE-EXPLAIN (docs/prereg_round13_detector_push.md "Round 61 SCENE-EXPLAIN"): on the 44 drawn specs of the
saved SHIP8+MD3 merged-DEV arm (DEPICT on, caches filled; base 29/58, 18 (6/9/3), 2.141), Qwen3.8-27B with 6 frames over
picture start +-1 s: Q1 is a real {X} likely the source (both option orders); Q2 (only if Q1 unlikely in both) what visible
thing could a detector mistake for {X}; Q3 text-only, could {Q2} be mistaken for {X} (both sound orders, as Round 57 L1/L2).
Drop iff Q1 unlikely x2 AND Q2 names a thing AND Q3 yes x2. Re-placement and scoring as sign_screen.cmd_score (unit 44).

    TG_ARMS=SHIP8+MD3 python benchmark/gold/explain_screen.py specs    # CPU: harness check (44 specs / 74 stretches)
    TG_ARMS=SHIP8+MD3 python benchmark/gold/explain_screen.py sanity   # GPU: 10 specs; exit 3 if >= 9 share a Q1 pair
    TG_ARMS=SHIP8+MD3 python benchmark/gold/explain_screen.py run      # GPU -> benchmark/gold/explain_pics/
    TG_ARMS=SHIP8+MD3 python benchmark/gold/explain_screen.py score    # CPU -> benchmark/gold/explain_screen.json (stack)
    TG_ARMS=SHIP8+MD3 python benchmark/gold/explain_screen.py score_replace   # CPU -> explain_screen_replace.json (decisive)
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")

CACHE = _ROOT / "benchmark" / "gold" / "explain_pics"
OUT = _ROOT / "benchmark" / "gold" / "explain_screen.json"
SANITY = _ROOT / "benchmark" / "gold" / "explain_sanity.json"
N_SPECS, N_STRETCH = 44, 74
BASE = {"hits": 29, "wrong": 18, "cost": 2.141}
C_ROW = {"hits": 27, "wrong": 13, "cost": 2.113}
LAUND = ("dev", "b3_laundromat", "Train")
Q1A = ("A sound detector says it heard {X} here. Looking at this scene, is a real {X} likely to be the source, "
       "on or off screen? Answer likely or unlikely.")
Q1B = ("A sound detector says it heard {X} here. Looking at this scene, is a real {X} likely to be the source, "
       "on or off screen? Answer unlikely or likely.")
Q2 = ("What visible thing in this scene most likely made a sound that a detector could mistake for {X}? "
      "Answer with a short noun phrase, or 'nothing'.")
L1 = "Could the sound of {B} be mistaken for the sound of {A}? Answer yes or no."
L2 = "Could the sound of {A} be mistaken for the sound of {B}? Answer yes or no."


def yn(reply: str):
    t = reply.strip().lower().lstrip("*\"'([ ").split()
    w = t[0].strip(".,!*\"')]") if t else ""
    return w if w in ("yes", "no") else None


def lu(reply: str):
    """'unlikely' checked before 'likely', in the first 3 words; else None (unparsed)"""
    w = [x.strip(".,!*\"'()[]:") for x in reply.strip().lower().split()[:3]]
    if "unlikely" in w:
        return "unlikely"
    if "likely" in w:
        return "likely"
    return None


def thing(reply: str):
    """Q2 reply -> confuser phrase, or None for nothing/none/no .../empty"""
    t = reply.strip().splitlines()[0] if reply.strip() else ""
    t = t.strip().strip("*\"'`.,!;:()[] ").lower()
    t = re.sub(r"^(a|an|the)\s+", "", t).strip()
    if not t or t.startswith("nothing") or t.startswith("none") or t.startswith("no ") or t == "no":
        return None
    return t


def specs():
    """[(part, stem, label, start)] in sign_screen.asked order (the 44 drawn specs)"""
    from benchmark.gold import sign_screen as SS
    A = SS.asked(SS.parts())
    assert len(A) == N_SPECS and sum(len(a[5]) for a in A) == N_STRETCH, (len(A), sum(len(a[5]) for a in A))
    return [(pt, st, lab, a0) for pt, st, w, lab, a0, sts, src in A]


def ask(reason, mdl, proc, pt, st, lab, a0):
    from benchmark.gold.imp_v_screen import clip_file
    from benchmark.gold.depict_screen import _frames
    vp = Path(clip_file(pt, st))
    X = lab.lower()
    rec = {"part": pt, "clip": st, "label": lab, "start": a0, "X": X, "clip_file": str(vp)}
    fr = _frames(vp, a0)
    rec["n_frames"] = len(fr)
    if len(fr) < 2:
        rec.update({"q1_replies": [], "q1": None, "q1_unlikely": False, "silenced": False, "note": "frames"})
        return rec
    r1 = reason._ask(mdl, proc, Q1A.format(X=X), images=fr, max_new=4)
    r2 = reason._ask(mdl, proc, Q1B.format(X=X), images=fr, max_new=4)
    q1 = [lu(r1), lu(r2)]
    rec.update({"q1_replies": [r1, r2], "q1": q1, "q1_unlikely": q1 == ["unlikely", "unlikely"]})
    rec.update({"q2_reply": None, "q2_thing": None, "q3_replies": [], "q3": None, "q3_yes": False})
    if rec["q1_unlikely"]:
        r = reason._ask(mdl, proc, Q2.format(X=X), images=fr, max_new=16)
        B = thing(r)
        rec.update({"q2_reply": r, "q2_thing": B})
        if B:
            x1 = reason._ask(mdl, proc, L1.format(B=B, A=X), images=None, max_new=4)
            x2 = reason._ask(mdl, proc, L2.format(B=B, A=X), images=None, max_new=4)
            rec.update({"q3_prompts": [L1.format(B=B, A=X), L2.format(B=B, A=X)], "q3_replies": [x1, x2],
                        "q3": [yn(x1), yn(x2)], "q3_yes": yn(x1) == "yes" and yn(x2) == "yes"})
    rec["silenced"] = bool(rec["q1_unlikely"] and rec["q2_thing"] and rec["q3_yes"])
    return rec


def _load():
    from src.stage5_cross_modal_analysis import reason
    from benchmark.gold import sign_screen as SS
    return reason, reason._load(SS.MODEL, "cuda")


def cmd_specs():
    from benchmark.gold import sign_screen as SS
    SS.cmd_specs()
    S = specs()
    print("laundromat Train specs:", [x for x in S if (x[0], x[1], x[2]) == LAUND])


def cmd_sanity():
    S = specs()
    li = [i for i, x in enumerate(S) if (x[0], x[1], x[2]) == LAUND]
    assert li, "laundromat Train not among the drawn specs"
    pick = [S[li[0]]] + [S[i] for i in range(0, 37, 4) if i != li[0]][:9]
    reason, (mdl, proc) = _load()
    rows = []
    for x in pick:
        r = ask(reason, mdl, proc, *x)
        rows.append(r)
        print("SANITY", r["part"], r["clip"], r["label"], r["start"], "Q1", r["q1_replies"], r["q1"], "| Q2",
              repr(r.get("q2_reply")), r.get("q2_thing"), "| Q3", r.get("q3_replies"), "->",
              "DROP" if r["silenced"] else "kept", flush=True)
    c = collections.Counter(str(r["q1"]) for r in rows)
    stop = c.most_common(1)[0][1] >= 9
    print("SANITY Q1 pairs", dict(c), "->", "STOP (untestable with this VLM)" if stop else "GO", flush=True)
    SANITY.write_text(json.dumps({"rows": rows, "q1_pairs": dict(c), "stop": stop,
                                  "prompts": {"Q1A": Q1A, "Q1B": Q1B, "Q2": Q2, "L1": L1, "L2": L2}}, indent=1),
                      encoding="utf-8")
    sys.exit(3 if stop else 0)


def cmd_run():
    from benchmark.gold import sign_screen as SS
    CACHE.mkdir(parents=True, exist_ok=True)
    S = [x for x in specs() if not (CACHE / f"{SS.key(*x)}.json").exists()]
    print(len(S), "specs to ask", flush=True)
    if not S:
        return
    reason, (mdl, proc) = _load()
    for x in S:
        r = ask(reason, mdl, proc, *x)
        (CACHE / f"{SS.key(*x)}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(r["part"], r["clip"], r["label"], r["start"], "Q1", r["q1_replies"], "| Q2", repr(r.get("q2_reply")),
              "| Q3", r.get("q3_replies"), "->", "DROP" if r["silenced"] else "kept", flush=True)


def cmd_score(replace=False):
    """replace=False: Round 61 stacked on B (DEPICT on). replace=True: Round 61 INSTEAD of Round 57 (DEPICT_EVENT off in
    the arm's display flags, 61's mask applied); compared with the same B constants (29/18/2.141). Decisive = replace."""
    from benchmark.gold import sign_screen as SS
    from benchmark.gold import round13_dev as R13
    if replace:
        R13.ARMS[SS.ARM] = {**R13.ARMS[SS.ARM], "DEPICT_EVENT": False}
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import btp_screen as Bt
    from benchmark.gold.cross_group import classify
    R = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(CACHE.glob("*.json"))]
    assert len(R) == N_SPECS, len(R)
    mask = {}
    for r in R:
        if r["silenced"]:
            mask.setdefault((r["part"], r["clip"]), set()).add((r["label"], round(float(r["start"]), 3)))
    mask = {k: frozenset(v) for k, v in mask.items()}
    rows = SS.parts(mask)
    base, new = {"dev": [], "dev2": []}, {"dev": [], "dev2": []}
    lost, changed, cls = [], [], {}
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, g, w, pics, pics1 in rows:
        r0, r1 = S.score_clip(g, pics), S.score_clip(g, pics1)
        base[pt].append(r0); new[pt].append(r1)
        cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, pics)}
        for l, a, b in pics:
            cls[(pt, st, l, round(a, 3))] = cl.get((l, round(a, 3)))
        if pics != pics1:
            gone = [p for p in pics if p not in pics1]; added = [p for p in pics1 if p not in pics]
            changed.append({"part": pt, "clip": st, "removed": [[l, round(a, 2), round(b, 2), cl.get((l, round(a, 3)))] for l, a, b in gone],
                            "added": [[l, round(a, 2), round(b, 2)] for l, a, b in added], "before": oc(r0), "after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    sm = lambda v: SS.summ_fmt(v)[0]
    Bm = {"merged": sm(base["dev"] + base["dev2"]), "dev": sm(base["dev"]), "dev2": sm(base["dev2"])}
    X = {"merged": sm(new["dev"] + new["dev2"]), "dev": sm(new["dev"]), "dev2": sm(new["dev2"])}
    tag = "REPLACE (61 instead of 57; base shown = SHIP8+MD3 with DEPICT off)" if replace else "STACK (61 on B)"
    print(f"[{tag}] BASE: merged {Bt.fmt(Bm['merged'])} | DEV {Bt.fmt(Bm['dev'])} | DEV2 {Bt.fmt(Bm['dev2'])}")
    if not replace:
        assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
        assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    else:
        assert Bm["merged"]["hits"] == BASE["hits"], Bm["merged"]      # DEPICT drops no hit on DEV
    M = X["merged"]; ln = sum(x[2] for x in lost); removed_wrong = BASE["wrong"] - M["wrong"]
    cheaper = M["cost"] < BASE["cost"]
    unit = (removed_wrong >= 1) if ln == 0 else (removed_wrong >= 2 * ln)
    main = M["hits"] >= BASE["hits"] and not lost and cheaper
    few = cheaper and M["wrong"] <= BASE["wrong"] - 3 * ln and M["hits"] >= 26
    verdict = "PASS" if (cheaper and unit and (main or few)) else "FAIL"
    vsC = {"d_hits": M["hits"] - C_ROW["hits"], "d_wrong": M["wrong"] - C_ROW["wrong"], "d_cost": round(M["cost"] - C_ROW["cost"], 3)}
    print(f"[{tag}] EXPLAIN: merged {Bt.fmt(M)} | DEV {Bt.fmt(X['dev'])} | DEV2 {Bt.fmt(X['dev2'])}  hits lost {lost}  "
          f"wrong removed {removed_wrong}  cheaper {cheaper} unit {unit} main {main} few {few} -> {verdict}  vs C {vsC}")
    for c in changed:
        print("  ", c)
    # every silenced spec with its class (first placed picture of the spec; None = not placed in base, e.g. DEPICT-dropped)
    dropped = []
    for r in R:
        if r["silenced"]:
            k = cls.get((r["part"], r["clip"], r["label"], round(float(r["start"]), 3)))
            if k is None:      # placed start != spec start: take the classes of this label's removed pictures
                k = [x[3] for c in changed if c["clip"] == r["clip"] for x in c["removed"] if x[0] == r["label"]] or None
            dropped.append({"part": r["part"], "clip": r["clip"], "label": r["label"], "start": r["start"], "class": k,
                            "q1": r["q1_replies"], "q2": r["q2_reply"], "q3": r["q3_replies"]})
            print("DROPPED", r["part"], r["clip"], r["label"], r["start"], k, "| Q1", r["q1_replies"], "| Q2",
                  repr(r["q2_reply"]), "| Q3", r["q3_replies"])
    q1pairs = collections.Counter(str(r.get("q1")) for r in R)
    funnel = {"q1_unlikely": sum(r["q1_unlikely"] for r in R), "q2_named": sum(bool(r.get("q2_thing")) for r in R),
              "q3_yes": sum(bool(r.get("q3_yes")) for r in R), "silenced": len(dropped)}
    unlikely = [[r["clip"], r["label"], r["start"], cls.get((r["part"], r["clip"], r["label"], round(float(r["start"]), 3))),
                 r.get("q2_reply"), r.get("q3")] for r in R if r["q1_unlikely"]]
    print("Q1 pairs", dict(q1pairs), "funnel", funnel)
    for u in unlikely:
        print("  Q1-unlikely:", u)
    steam = [d for d in dropped if d["clip"] == "b3_crossing_bells" and d["label"] == "Steam"]
    print("Round 57 drop (b3_crossing_bells Steam) also made by 61:", bool(steam))
    res = {"mode": "replace" if replace else "stack", "r57_steam_also_dropped": bool(steam), "base": Bm, "rows": X, "vs_C": vsC, "C": C_ROW, "q1_pairs": dict(q1pairs), "funnel": funnel, "q1_unlikely": unlikely,
           "dropped": dropped, "changed": changed, "hits_lost": lost, "wrong_removed": removed_wrong, "cheaper": cheaper,
           "cost_unit": unit, "main_rule": main, "fewer_pictures": few, "verdict": verdict, "model": SS.MODEL,
           "prompts": {"Q1A": Q1A, "Q1B": Q1B, "Q2": Q2, "L1": L1, "L2": L2}}
    out = OUT.with_name(OUT.stem + ("_replace" if replace else "") + ".json")
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"specs": cmd_specs, "sanity": cmd_sanity, "run": cmd_run, "score": cmd_score,
     "score_replace": lambda: cmd_score(replace=True)}[cmd]()
