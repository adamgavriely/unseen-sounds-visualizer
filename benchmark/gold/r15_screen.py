"""Round 15 candidate screen (DEV only, offline, 2026-09-30): post-display rules on the pictures of the best DEV arm
TO1+F7F8 (data/work/r13/TO1+F7F8_proposed, read-only copy of the cluster run), rescored with the same scorer and cost.
  S1 strict gate: a picture is dropped if ANY gate vote (name / ab / desc) on ANY of its stretches said visible.
  S2 one sound, one picture: a picture is dropped if an earlier picture of the same family exists in the clip and the
     family's evidence (max of BEATs and FlexSED same-family columns) stays >= TAU at every frame from the earlier
     picture's start to this picture's start (the sound never stopped). TAU = 0.2 (the FIX-CTRL "present" level).
Candidate-level screen: its numbers are seen before any pre-registration of a pipeline arm.

    python benchmark/gold/r15_screen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S

ARM, SYS = "TO1+F7F8", "proposed"
TAU = 0.2
BEATS = _ROOT / "benchmark" / "gold" / "beats_fw"
FLEX = _ROOT / "data" / "work" / "flexsed_cache"


def pictures(arm, stems):
    with R.flags({k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}):
        return {st: S.load_pictures(R.R13 / f"{arm}_{SYS}", st, SYS) or [] for st in stems}


def votes(arm, st, label, start):
    f = R.R13 / f"{arm}_{SYS}" / st / "gate_votes.json"
    if not f.exists():
        return []
    return [v for v in json.loads(f.read_text(encoding="utf-8"))
            if S.same_family(v["label"], label) and abs(v["start"] - start) < 0.6]


def evidence(st, label):
    """(times, score) on a 0.04-s grid: max over BEATs and FlexSED columns of the picture's family"""
    zf = np.load(FLEX / f"{st}.npz", allow_pickle=True)
    fw, labs, fps = zf["fw"], [str(x) for x in zf["labels"]], float(zf["fps"])
    cols = [i for i, l in enumerate(labs) if S.same_family(l, label)]
    t = np.arange(fw.shape[1]) / fps
    e = fw[cols].max(axis=0) if cols else np.zeros_like(t)
    fb = BEATS / f"{st}.npz"
    if fb.exists():
        zb = np.load(fb, allow_pickle=True)
        bl = [str(x) for x in zb["labels"]]
        bc = [i for i, l in enumerate(bl) if S.same_family(l, label)]
        if bc:
            bt = zb["times"]; bv = zb["fw"][:, bc].max(axis=1)
            # a 2-s window covers [start, start + 2): value at t = max over windows covering t
            cov = np.array([bv[(bt <= x) & (bt + 2.0 > x)].max(initial=0.0) for x in t])
            e = np.maximum(e, cov)
    return t, e


def s1(arm, st, pics):
    return [p for p in pics if not any(v["name"] or v["ab"] or v["desc"] for v in votes(arm, st, p[0], p[1]))]


def s2(st, pics):
    keep = []
    for p in sorted(pics, key=lambda x: x[1]):
        prev = [q for q in keep if S.same_family(q[0], p[0]) and q[1] < p[1]]
        if prev:
            t, e = evidence(st, p[0])
            a = max(q[1] for q in prev)
            seg = e[(t >= a) & (t <= p[1])]
            if seg.size and seg.min() >= TAU:
                continue
        keep.append(p)
    return keep


def main():
    gold, stems = DCC.dev_stems()
    P = {"B0r": pictures("B0r", stems), ARM: pictures(ARM, stems)}
    P["S1"] = {st: s1(ARM, st, P[ARM][st]) for st in stems}
    P["S2"] = {st: s2(st, P[ARM][st]) for st in stems}
    P["S1+S2"] = {st: s2(st, P["S1"][st]) for st in stems}
    rows = {n: [S.score_clip(gold[st], P[n][st]) for st in stems] for n in P}
    cost = {n: [DCC.clip_cost(r) for r in rr] for n, rr in rows.items()}
    out = {}
    for n in P:
        m = DCC.metrics(rows[n])
        d = DCC.boot(np.subtract(cost[n], cost["B0r"])) if n != "B0r" else None
        out[n] = {"hits": m["hits"], "wrong": m["wrong"], "v/c/p": [m["visible"], m["cross"], m["phantom"]],
                  "cost": round(m["viewer_cost"], 3), "d_vs_B0r": d}
        print(n, out[n])
    for n in ("S1", "S2"):
        ch = []
        for st in stems:
            gone = [p for p in P[ARM][st] if p not in P[n][st]]
            if gone:
                h0 = DCC.needed_hit(gold[st], P[ARM][st]); h1 = DCC.needed_hit(gold[st], P[n][st])
                ch.append([st, [(p[0], round(p[1], 2)) for p in gone], "hits lost:", [x for x in h0 if x not in h1]])
        out[n + "_dropped"] = ch
        print(n, "dropped:"); [print("  ", c) for c in ch]
    (_ROOT / "benchmark" / "gold" / "r15_screen.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
