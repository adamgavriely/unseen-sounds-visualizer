"""Round 61d SCENE-EXPLAIN, symmetric-polarity Q3 (docs/prereg_round13_detector_push.md "Round 61d"). Q1 / Q2 as Round 61b
(cached in explain_pics{_variant}/; specs of an arm not in that cache are asked fresh with explain_screen.ask, main variant
only, into explain_d_new/). New Q3 (text only): L1 "Could the sound of {B} be mistaken for the sound of {A}?" (cached 61b
reply reused) AND its polarity twin L1n "Is the sound of {B} clearly distinguishable from the sound of {A}?"; drop iff
Q1 unlikely x2 AND Q2 names B AND L1 yes AND L1n no. Unparsed -> one re-prompt " Reply yes or no only." (61b harness).

    ARM=SHIP8+MD3      python benchmark/gold/explain_d.py specs     # msproj CPU -> explain_d_specs_<arm>.json
    ARM=SHIP8+MD3+WW5  python benchmark/gold/explain_d.py specs
    EXPLAIN_VARIANT=main python benchmark/gold/explain_d.py ask     # GPU -> explain_d{_variant}/ (both arms' specs)
    ARM=... FORM=stack|replace [EXPLAIN_VARIANT=...] python benchmark/gold/explain_d.py score   # msproj CPU
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", os.environ.get("ARM", "SHIP8+MD3"))

from benchmark.gold import explain_screen as E          # noqa: E402  (reads EXPLAIN_VARIANT)

G = _ROOT / "benchmark" / "gold"
ARMS = {"SHIP8+MD3": {"tag": "B", "hits": 29, "wrong": 18, "cost": 2.141, "cls": (6, 9, 3)},
        "SHIP8+MD3+WW5": {"tag": "D", "hits": 29, "wrong": 14, "cost": 2.028, "cls": (6, 6, 2)}}
DCACHE = G / f"explain_d{E._SUF}"
NEW = G / "explain_d_new"                                # Q1/Q2/L1 of D-only specs (main variant)
L1N = "Is the sound of {B} clearly distinguishable from the sound of {A}? Answer yes or no."


def _arm():
    a = os.environ.get("ARM", "SHIP8+MD3")
    assert a in ARMS, a
    return a


def specfile(arm):
    return G / f"explain_d_specs_{ARMS[arm]['tag']}.json"


def cmd_specs():
    arm = _arm()
    from benchmark.gold import sign_screen as SS
    from benchmark.gold import score_per_sound as S
    SS.ARM = arm
    rows = SS.parts()
    A = SS.asked(rows)
    b = SS.summ_fmt([S.score_clip(g, pics) for pt, st, g, w, pics, _ in rows])
    print(f"{arm}: {len(rows)} clips, {sum(len(r[4]) for r in rows)} placed, {len(A)} drawn specs; BASE {b[1]}")
    m = b[0]
    assert (m["hits"], m["wrong"]) == (ARMS[arm]["hits"], ARMS[arm]["wrong"]) and abs(m["cost"] - ARMS[arm]["cost"]) < 0.001, m
    S2 = [[pt, st, lab, a0] for pt, st, w, lab, a0, sts, src in A]
    specfile(arm).write_text(json.dumps(S2), encoding="utf-8")
    from benchmark.gold.sign_screen import key
    miss = [x for x in S2 if not (E.CACHE / f"{key(*x)}.json").exists()]
    print("not in the 61b cache:", miss)


def q1q2(x):
    """61b record for spec x: from the variant's 61b cache, else explain_d_new/ (main only); None if absent"""
    from benchmark.gold.sign_screen import key
    for d in (E.CACHE, NEW):
        f = d / f"{key(*x)}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def cmd_ask():
    from benchmark.gold.sign_screen import key
    S = []
    for arm in ARMS:
        for x in json.loads(specfile(arm).read_text(encoding="utf-8")):
            if x not in S:
                S.append(x)
    DCACHE.mkdir(parents=True, exist_ok=True)
    NEW.mkdir(parents=True, exist_ok=True)
    todo = [x for x in S if not (DCACHE / f"{key(*x)}.json").exists()]
    print(E.VARIANT, len(S), "specs (union),", len(todo), "to do", flush=True)
    reason = mdl = proc = None
    for x in todo:
        r = q1q2(x)
        if r is None and E.VARIANT != "main":
            print("SKIP (no", E.VARIANT, "Q1/Q2 for this spec; report only)", x, flush=True)
            continue
        if reason is None:
            reason, (mdl, proc) = E._load()
        if r is None:                                     # D-only spec: 61b asks (Q1, Q2, L1/L2) fresh, main variant
            r = E.ask(reason, mdl, proc, *x)
            (NEW / f"{key(*x)}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        rec = {"part": x[0], "clip": x[1], "label": x[2], "start": x[3], "variant": E.VARIANT,
               "q1_unlikely": r["q1_unlikely"], "q1_replies": r.get("q1_replies"), "q2_reply": r.get("q2_reply"),
               "q2_reprompt": r.get("q2_reprompt"), "q2_thing": r.get("q2_thing"), "l1": None, "l1n": None}
        B = r.get("q2_thing") if r["q1_unlikely"] else None
        if B:
            A = r["X"]
            # L1: the cached 61b reply (q3_replies[0], re-prompt q3_reprompt[0]); asked only if absent
            if r.get("q3_replies"):
                l1r, l1rr = r["q3_replies"][0], (r.get("q3_reprompt") or [None])[0]
                l1 = r["q3"][0]
            else:
                l1r, l1rr, l1 = _yn_ask(reason, mdl, proc, E.L1.format(B=B, A=A))
            nr, nrr, n = _yn_ask(reason, mdl, proc, L1N.format(B=B, A=A))
            rec.update({"l1": l1, "l1_reply": [l1r, l1rr], "l1n": n, "l1n_reply": [nr, nrr],
                        "prompts": [E.L1.format(B=B, A=A), L1N.format(B=B, A=A)]})
        rec["silenced"] = bool(B and rec["l1"] == "yes" and rec["l1n"] == "no")
        (DCACHE / f"{key(*x)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(x, "Q1u", rec["q1_unlikely"], "Q2", repr(rec["q2_thing"]), "L1", rec.get("l1_reply"), "L1n",
              rec.get("l1n_reply"), "->", "DROP" if rec["silenced"] else "kept", flush=True)


def _yn_ask(reason, mdl, proc, q):
    x = reason._ask(mdl, proc, q, images=None, max_new=4)
    xr = None
    if E.yn(x) is None:
        xr = reason._ask(mdl, proc, q + E.RE_Q3, images=None, max_new=4)
    return x, xr, (E.yn(x) if E.yn(x) is not None else (E.yn(xr) if xr else None))


def cmd_score():
    arm, form = _arm(), os.environ.get("FORM", "replace")
    from benchmark.gold import sign_screen as SS
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import btp_screen as Bt
    from benchmark.gold import round13_dev as R13
    from benchmark.gold.cross_group import classify
    from benchmark.gold.sign_screen import key
    SS.ARM = arm
    if form == "replace":
        R13.ARMS[arm] = {**R13.ARMS[arm], "DEPICT_EVENT": False}
    specs = json.loads(specfile(arm).read_text(encoding="utf-8"))
    R, missing = [], []
    for x in specs:
        f = DCACHE / f"{key(*x)}.json"
        (R.append(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else missing.append(x))
    mask = {}
    for r in R:
        if r["silenced"]:
            mask.setdefault((r["part"], r["clip"]), set()).add((r["label"], round(float(r["start"]), 3)))
    rows = SS.parts({k: frozenset(v) for k, v in mask.items()})
    K = ARMS[arm]
    base, new, lost, changed = [], [], [], []
    for pt, st, g, w, pics, pics1 in rows:
        r0, r1 = S.score_clip(g, pics), S.score_clip(g, pics1)
        base.append(r0); new.append(r1)
        if pics != pics1:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, pics)}
            changed.append({"part": pt, "clip": st, "removed": [[l, round(a, 2), round(b, 2), cl.get((l, round(a, 3)))]
                                                                 for l, a, b in pics if (l, a, b) not in pics1],
                            "added": [[l, round(a, 2), round(b, 2)] for l, a, b in pics1 if (l, a, b) not in pics]})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    Bm, M = SS.summ_fmt(base)[0], SS.summ_fmt(new)[0]
    if form == "stack":
        assert (Bm["hits"], Bm["wrong"]) == (K["hits"], K["wrong"]) and abs(Bm["cost"] - K["cost"]) < 0.001, Bm
    ln = sum(x[2] for x in lost); removed_wrong = K["wrong"] - M["wrong"]
    cheaper = M["cost"] < K["cost"]
    unit = (removed_wrong >= 1) if ln == 0 else (removed_wrong >= 2 * ln)
    main = M["hits"] >= K["hits"] and not lost and cheaper
    few = cheaper and M["wrong"] <= K["wrong"] - 3 * ln and M["hits"] >= 26
    verdict = "PASS" if (cheaper and unit and (main or few)) else "FAIL"
    tag = f"{K['tag']} {form} {E.VARIANT}"
    print(f"[{tag}] base-as-loaded {Bt.fmt(Bm)} | 61d {Bt.fmt(M)} vs {K['tag']} {K['hits']}/{K['wrong']}/{K['cost']}: lost {lost} "
          f"wrong removed {removed_wrong} cheaper {cheaper} unit {unit} main {main} few {few} -> {verdict}; missing {len(missing)}")
    for c in changed:
        print("   changed", c)
    dropped = [{k: r.get(k) for k in ("clip", "label", "start", "q2_thing", "l1_reply", "l1n_reply")} for r in R if r["silenced"]]
    for d in dropped:
        print("   DROPPED", d)
    funnel = {"specs": len(specs), "answered": len(R), "q1_unlikely": sum(r["q1_unlikely"] for r in R),
              "q2_named": sum(bool(r["q1_unlikely"] and r["q2_thing"]) for r in R),
              "l1_yes": sum(r.get("l1") == "yes" for r in R), "l1n_no": sum(r.get("l1n") == "no" for r in R),
              "dropped": len(dropped)}
    print("   funnel", funnel, "| L1/L1n pairs", dict(collections.Counter(str([r.get("l1"), r.get("l1n")]) for r in R
                                                                         if r.get("l1n") is not None)))
    out = G / f"explain_d_{K['tag']}_{form}{E._SUF}.json"
    out.write_text(json.dumps({"arm": arm, "form": form, "variant": E.VARIANT, "base_loaded": Bm, "row": M, "ref": K,
                               "hits_lost": lost, "wrong_removed": removed_wrong, "cheaper": cheaper, "unit": unit,
                               "main_rule": main, "fewer_pictures": few, "verdict": verdict, "dropped": dropped,
                               "changed": changed, "funnel": funnel, "missing": missing, "L1N": L1N}, indent=1, default=float),
                   encoding="utf-8")
    print(out)


if __name__ == "__main__":
    {"specs": cmd_specs, "ask": cmd_ask, "score": cmd_score}[sys.argv[1]]()
