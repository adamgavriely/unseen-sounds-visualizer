"""Round 65 RETURN (docs/prereg_round13_detector_push.md "Round 65 RETURN"): the 415 guard for PERC_RETURN, and the DEV listing.

    python benchmark/gold/perc_return.py guard                 # CPU, from ~/MscProj_tg: k on half A, GO / STOP on half B (exit 0 / 3)
    TG_ARMS="A B" python benchmark/gold/perc_return.py diff A B  # CPU, from ~/MscProj_tg: changed DEV pictures 
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

HOME_PROJ = Path("/home/dsi/adamg/MscProj")
HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
BEATS_HELD = HOME_PROJ / "benchmark" / "audioset_heldout_windows" / "beats"
FLEX_HELD = _ROOT / "data" / "work" / "flexsed_heldout"
AUDIO = HOME_PROJ / "data" / "work" / "tagens" / "audio"          # 16-kHz mono float32, msproj decode (Round 63)
OUT = _ROOT / "benchmark" / "gold" / "perc_return_415.json"
KGRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EARLY, LATE, PAUSE, MIN_N, BAR = 0.5, 1.0, 2.0, 20, 0.42


def clip_inputs(cid):
    """(proxy 'drawn' rows, BEATs frames, FlexSED frames, audio): raw BEATs spans (peak >= 0.35, low 0.175, min 0.3) and
    FlexSED 0.8 runs (min 0.3) of depictable families"""
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import expect_screen as E
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    dep = set(E.FAMILIES)
    B = DCC.load_fr(BEATS_HELD / f"{cid}.npz")
    F = DCC.load_fr(FLEX_HELD / f"{cid}.npz")
    braw = [e for e in _extract_events(B[0], B[1], B[2], 0.35, None, 0.3, low=0.175) if canonical(e.label) in dep]
    rows = braw + [e for e in _extract_events(F[0], F[1], F[2], 0.8, None, 0.3, low=0.8) if canonical(e.label) in dep]
    return rows, braw, B, F, np.load(AUDIO / f"{cid}.npy")


def new_events(evs):
    """strong events that are NEW: no other same-family event starts before it and ends after its start - 2.0"""
    from benchmark.gold.score_per_sound import same_family
    out = []
    for e in evs:
        if not any(o is not e and same_family(o["label"], e["label"]) and o["start"] < e["start"]
                   and o["end"] > e["start"] - PAUSE for o in evs):
            out.append(e)
    return out


def score(cands, evs):
    """(correct under the new-event truth, correct under the plain Round 42 rule) for (family, onset) candidates"""
    from benchmark.gold.score_per_sound import same_family
    ne = new_events(evs)
    hit = lambda f, a, pool: any(same_family(f, e["label"]) and e["start"] - EARLY <= a <= e["start"] + LATE for e in pool)
    return sum(hit(f, a, ne) for f, a in cands), sum(hit(f, a, evs) for f, a in cands)


def run(ids, ks, ev_of, memo):
    import config
    from src.labels import canonical
    from src.stage4_audio_event_detection import perceptual_returns
    res = {k: {"n": 0, "new": 0, "plain": 0, "rows": []} for k in ks}
    base = {"n": 0, "new": 0, "plain": 0}
    for cid in ids:
        if cid not in memo:
            memo[cid] = clip_inputs(cid)
        rows, braw, B, F, y = memo[cid]
        bsp = [(canonical(e.label), float(e.start)) for e in braw]
        c = score(bsp, ev_of[cid])
        base["n"] += len(bsp); base["new"] += c[0]; base["plain"] += c[1]
        for k in ks:
            r = perceptual_returns(rows, B[0], B[1], B[2], F[0], F[1], F[2], y, k=k)
            cand = [(e.label, float(e.start)) for e in r]
            c = score(cand, ev_of[cid])
            res[k]["n"] += len(cand); res[k]["new"] += c[0]; res[k]["plain"] += c[1]
            res[k]["rows"] += [[cid, f, round(a, 2), bool(score([(f, a)], ev_of[cid])[0])] for f, a in cand]
    for d in list(res.values()) + [base]:
        d["p_new"] = round(d["new"] / d["n"], 4) if d["n"] else None
        d["p_plain"] = round(d["plain"] / d["n"], 4) if d["n"] else None
    return res, base


def cmd_guard():
    import config
    from benchmark.gold.tagens import split
    config.LABEL_FILTER = "depictable"; config.DISPLAY_THRESHOLD = 0.35
    A, B = split()
    ev_of = {c["id"]: c["events"] for c in json.loads(HELDOUT.read_text(encoding="utf-8"))["clips"]}
    memo, t0 = {}, time.time()
    ra, ba = run(A, KGRID, ev_of, memo)
    for k in KGRID:
        print(f"[half A] k {k}: {ra[k]['n']} candidates, new-event correct {ra[k]['new']} (p {ra[k]['p_new']}), "
              f"plain {ra[k]['plain']} (p {ra[k]['p_plain']})", flush=True)
    ok = [k for k in KGRID if ra[k]["n"] >= MIN_N]
    out = {"round": "Round 65 RETURN 415 guard", "kgrid": KGRID, "half_A": {str(k): {x: v for x, v in ra[k].items() if x != "rows"}
                                                                             for k in KGRID}, "half_A_beats": ba}
    if not ok:
        out["verdict"] = "STOP (no k with >= 20 half-A candidates)"
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8"); print(out["verdict"]); return 3
    k = sorted(ok, key=lambda x: (-ra[x]["p_new"], x))[0]
    rb, bb = run(B, [k], ev_of, memo)
    r = rb[k]
    go = r["n"] >= MIN_N and r["p_new"] is not None and r["p_new"] >= BAR
    sec = (time.time() - t0) / (len(A) + len(B))
    out.update({"k": k, "half_B": r, "half_B_beats": bb, "go": go, "verdict": "GO" if go else "STOP",
                "sec_per_clip_all_k": round(sec, 3)})
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[half A] chosen k {k}", flush=True)
    print(f"[half B] RET k {k}: {r['n']} candidates, new-event correct {r['new']} (p {r['p_new']}), plain {r['plain']} "
          f"(p {r['p_plain']}); raw BEATs/FlexSED spans {bb['n']}: new {bb['p_new']}, plain {bb['p_plain']}", flush=True)
    print(f"[verdict] {out['verdict']} -> {OUT}", flush=True)
    return 0 if go else 3


def cmd_diff(A, B):
    """changed pictures B vs A on merged DEV (weakwitness_dev.classify, as tagens diff)"""
    from benchmark.gold import weakwitness_dev as W
    res = {"changed": []}
    for part, (gold, P, _c, _s) in W.parts([A, B]).items():
        for st in P[A]:
            ca, cb = W.classify(gold[st], P[A][st]), W.classify(gold[st], P[B][st])
            ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
            kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
            if ka != kb:
                res["changed"].append({"part": part, "clip": st, "only_" + A: sorted(ka - kb, key=lambda x: x[1]),
                                       "only_" + B: sorted(kb - ka, key=lambda x: x[1])})
                print(part, st, "\n   -", sorted(ka - kb, key=lambda x: x[1]), "\n   +", sorted(kb - ka, key=lambda x: x[1]), flush=True)
    p = _ROOT / "benchmark" / "gold" / "perc_return_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "guard":
        sys.exit(cmd_guard())
    elif a[0] == "diff":
        cmd_diff(a[1], a[2])
