"""Round 60L SCENE-LOGIT (docs/prereg_round13_detector_push.md "Round 60L SCENE-LOGIT"): B, D and SL (= D with
config.SCENE_FIT_LOGIT) on merged DEV in one scoring, the pass rule vs B, changed pictures SL vs D and vs B, and every logit
scene ask (from <DASM_LOCAL_SCENE map>.logit_answers.jsonl, DEV / DEV2 clips only). Run from ~/MscProj_tg after the arms'
stage 4 / stage 5 (slurm/job_scenelogit.sh).

    python benchmark/gold/scenelogit.py      # CPU -> benchmark/gold/scenelogit_dev.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

B, D, SL = "SHIP8+MD3", "SHIP8+MD3+WW5", "SHIP8+MD3+WW5+SL"
REF = {B: (29, 18, 2.141), D: (29, 14, 2.028)}
OUT = _ROOT / "benchmark" / "gold" / "scenelogit_dev.json"


def main():
    from benchmark.gold import weakwitness_dev as W
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import btp_screen as Bt
    from benchmark.gold import round13_dev as R
    P = W.parts([B, D, SL])
    rows = {a: [] for a in (B, D, SL)}
    per_clip_hits = {}
    for part, (gold, Pk, _c, _s) in P.items():
        for st in Pk[B]:
            for a in (B, D, SL):
                r = S.score_clip(gold[st], Pk[a][st])
                rows[a].append(r)
                per_clip_hits.setdefault((part, st), {})[a] = r["hit"]
    M = {a: Bt.summ(rows[a]) for a in rows}
    for a in (B, D, SL):
        print(f"{a:18s} {Bt.fmt(M[a])}")
    for a, (h, w, c) in REF.items():
        assert (M[a]["hits"], M[a]["wrong"]) == (h, w) and abs(M[a]["cost"] - c) < 0.001, (a, M[a])
    lost = [[p, st, v[B] - v[SL]] for (p, st), v in per_clip_hits.items() if v[SL] < v[B]]
    gained = M[SL]["hits"] - M[B]["hits"]
    main_rule = (M[SL]["hits"] >= 29 and not lost and M[SL]["wrong"] <= 18 + 2 * max(0, gained) and M[SL]["cost"] < 2.141)
    verdict = "PASS (ship SL in place of D's text readout)" if main_rule else "FAIL (D unchanged)"
    print(f"SL vs B main rule: hits {M[SL]['hits']} (>= 29), needed hits lost vs B {lost}, wrong {M[SL]['wrong']} "
          f"(<= {18 + 2 * max(0, gained)}), cost {M[SL]['cost']:.3f} (< 2.141) -> {verdict}")
    changed = {}
    for ref in (D, B):
        out = []
        for part, (gold, Pk, _c, _s) in P.items():
            for st in Pk[SL]:
                ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in W.classify(gold[st], Pk[SL][st])}
                kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in W.classify(gold[st], Pk[ref][st])}
                if ka != kb:
                    out.append({"part": part, "clip": st, "only_SL": sorted(ka - kb, key=lambda x: x[1]),
                                "only_" + ref: sorted(kb - ka, key=lambda x: x[1])})
                    print(f"[SL vs {ref}] {part} {st}\n   + {sorted(ka - kb, key=lambda x: x[1])}\n   - "
                          f"{sorted(kb - ka, key=lambda x: x[1])}")
        changed[ref] = out
    mp = Path(str(R.arm_cfg(SL)["DASM_LOCAL_SCENE"]))
    devclips = set(json.loads(mp.read_text(encoding="utf-8")))
    asks = []
    lp = mp.with_name(mp.name + ".logit_answers.jsonl")
    if lp.exists():
        for ln in lp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                x = json.loads(ln)
                if x["key"][0] in devclips:
                    asks.append(x)
    for x in asks:
        print("   ask", x["key"], x["verdict"], [(a.get("answer"), a.get("d")) for a in x["answers"]])
    print(f"logit scene asks (DEV / DEV2): {len(asks)}; credible {sum(x['verdict'] is True for x in asks)}, not "
          f"{sum(x['verdict'] is False for x in asks)}, None {sum(x['verdict'] is None for x in asks)}")
    OUT.write_text(json.dumps({"rows": M, "hits_lost_vs_B": lost, "main_rule_vs_B": main_rule, "verdict": verdict,
                               "changed": changed, "asks": asks}, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
