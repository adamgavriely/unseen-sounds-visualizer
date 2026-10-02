"""Round 66 NAME-ALL step 2 (docs/prereg_round13_detector_push.md "Round 66 step 2"): D' vs D' + NAME_ALL on merged DEV in one
scoring; the pre-registered pass rule; changed pictures with the spec's NAME-ALL score A and stretch margins (from the arm's
gate_votes.json `nameall` rows); per-video VLM cost. Run from ~/MscProj_tg after both parts' stage 5 (slurm/job_nameall_dev_*.sh).

    python benchmark/gold/nameall_dev.py      # CPU -> benchmark/gold/nameall_dev.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DP, NA = "SHIP8+MD3+WW5+SL", "SHIP8+MD3+WW5+SL+NA"
REF = (29, 15, 2.056)
OUT = _ROOT / "benchmark" / "gold" / "nameall_dev.json"


def main():
    from benchmark.gold import round13_dev as R
    from benchmark.gold import weakwitness_dev as W
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import btp_screen as Bt
    from src.stage5_cross_modal_analysis.nameall import sound_score
    roots = {"dev": R.R13 / f"{NA}_proposed"}                    # before W.parts re-points R13 to DEV2
    P = W.parts([DP, NA])
    from benchmark.gold import tagger_prep as T                 # after the DEV gold is read (it blocks gold reads at import)
    roots["dev2"] = T.out("dev2") / f"{NA}_proposed"
    rows = {DP: [], NA: []}
    hits = {}
    for part, (gold, Pk, _c, _s) in P.items():
        for st in Pk[DP]:
            for a in (DP, NA):
                r = S.score_clip(gold[st], Pk[a][st])
                rows[a].append(r)
                hits.setdefault((part, st), {})[a] = r["hit"]
    M = {a: Bt.summ(rows[a]) for a in rows}
    for a in (DP, NA):
        print(f"{a:22s} {Bt.fmt(M[a])}")
    assert (M[DP]["hits"], M[DP]["wrong"]) == REF[:2] and abs(M[DP]["cost"] - REF[2]) < 0.001, M[DP]
    lost = [[p, st, v[DP] - v[NA]] for (p, st), v in hits.items() if v[NA] < v[DP]]
    gained = [[p, st, v[NA] - v[DP]] for (p, st), v in hits.items() if v[NA] > v[DP]]
    ln = sum(x[2] for x in lost)
    removed = REF[1] - M[NA]["wrong"]
    cheaper = M[NA]["cost"] < REF[2]
    main_rule = M[NA]["hits"] >= 29 and not lost and cheaper
    few = cheaper and removed >= 2 * ln and M[NA]["wrong"] <= REF[1] - 3 * ln and M[NA]["hits"] >= 26
    verdict = "PASS" if (main_rule or few) else "FAIL"
    print(f"NAME-ALL vs D': hits lost {lost} gained {gained}; wrong removed {removed}; cheaper {cheaper}; main {main_rule}; "
          f"fewer-pictures {few} -> {verdict}")
    # gate rows of the arm: per spec (label, start) the stretch margins and A; per clip the VLM cost
    gate, cost = {}, {}
    for part, root in roots.items():
        for gv in sorted(root.glob("*/gate_votes.json")):
            st = gv.parent.name
            for r in json.loads(gv.read_text(encoding="utf-8")):
                na = r.get("nameall")
                if na is None:
                    continue
                gate.setdefault((part, st, r["label"], round(float(r["start"]), 2)), []).append(na.get("m"))
                c = cost.setdefault(st, {"stretches": 0, "gen": 0, "prefill": 0})
                c["stretches"] += 1; c["gen"] += na.get("gen", 0); c["prefill"] += na.get("prefill", 0)
    changed = []
    for part, (gold, Pk, _c, _s) in P.items():
        for st in Pk[NA]:
            ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in W.classify(gold[st], Pk[NA][st])}
            kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in W.classify(gold[st], Pk[DP][st])}
            if ka != kb:
                specs = {k[2:]: [m, sound_score(m)] for k, m in gate.items() if k[0] == part and k[1] == st}
                changed.append({"part": part, "clip": st, "only_NA": sorted(ka - kb, key=lambda x: x[1]),
                                "only_Dprime": sorted(kb - ka, key=lambda x: x[1]),
                                "nameall": {f"{l}|{s}": v for (l, s), v in specs.items()}})
                print(f"[NA vs D'] {part} {st}\n   + {sorted(ka - kb, key=lambda x: x[1])}\n   - {sorted(kb - ka, key=lambda x: x[1])}"
                      f"\n   A: {[(l, s, None if v[1] == float('-inf') else round(v[1], 2)) for (l, s), v in specs.items()]}")
    for nm in ("as_explosion_XJ8lc3I6", "bell_miami"):
        print("named", nm, [(k[2], k[3], [None if m is None else round(m, 2) for m in v], round(sound_score(v), 2))
                            for k, v in gate.items() if k[1] == nm])
    if cost:
        g = sorted(c["gen"] for c in cost.values()); p = sorted(c["prefill"] for c in cost.values())
        s_ = sorted(c["stretches"] for c in cost.values())
        print(f"per-video VLM cost (gated clips {len(cost)}): stretches median {statistics.median(s_)} (max {s_[-1]}); generations "
              f"median {statistics.median(g)} (max {g[-1]}); prefill passes median {statistics.median(p)} (max {p[-1]})")
    OUT.write_text(json.dumps({"rows": M, "hits_lost": lost, "hits_gained": gained, "wrong_removed": removed, "cheaper": cheaper,
                               "main_rule": main_rule, "fewer_pictures": few, "verdict": verdict, "changed": changed,
                               "cost": cost, "gate": {"|".join(map(str, k)): v for k, v in gate.items()}},
                              indent=1, default=lambda v: None if v == float("-inf") else float(v)), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
