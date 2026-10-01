"""Round 60 SCENE-MARGIN (docs/prereg_round13_detector_push.md "Round 60 SCENE-MARGIN"). Run from ~/MscProj_tg (GPU for sanity / gate).

  python benchmark/gold/scenemargin.py videos    # clip -> mp4 map of DEV + DEV2 (SHIP8+MD3 media.json) = the DASM_LOCAL_SCENE value
  python benchmark/gold/scenemargin.py sanity    # step 0: 10 fixed one-ear spans, F3 frames, both option orders; exit 3 on STOP
  python benchmark/gold/scenemargin.py gate      # step 1: held-out 415 one-ear DASM < 0.35 spans, F3 verdict vs Round 42 class; exit 3 on STOP
  python benchmark/gold/scenemargin.py diff      # step 2: changed pictures of SHIP8+MD3+WW5 vs SHIP8+MD3 and vs SHIP8+MD3+WW + raw answers

One implementation of the scene question: reason._scene_fit (Round 14 F3), as called by stage 4 (_scene_margin).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3 SHIP8+MD3+WW SHIP8+MD3+WW5")

GOLD = _ROOT / "benchmark" / "gold"
WORK = Path.home() / "MscProj" / "data" / "work"
MAP = WORK / "scenemargin" / "videos.json"
ANS = MAP.with_name(MAP.name + ".answers.jsonl")
OUT = GOLD / "scenemargin_415.json"
B, C, CP = "SHIP8+MD3", "SHIP8+MD3+WW", "SHIP8+MD3+WW5"
DEV5 = [("tg_d032", "Thunder", 13.75, 15.0), ("tg_d075", "Alarm", 0.14, 9.25), ("tg_d128", "Hammer", 9.0, 10.0),
        ("tg_d107", "Screaming", 6.52, 9.0), ("mv_protest_scene_movie", "Glass", 4.75, 6.25)]
ORDER2 = "Answer no or yes."


def vlm():
    import torch
    import config
    from src.stage5_cross_modal_analysis import reason
    config.VLM_THINKING = False
    return reason, reason._load("Qwen/Qwen3.8-27B", "cuda" if torch.cuda.is_available() else "cpu")


def ask(reason, M, vid, fam, s, e, prompt=None):
    old = reason.SCENE_FIT_PROMPT
    if prompt:
        reason.SCENE_FIT_PROMPT = prompt
    try:
        log = []
        v = reason._scene_fit(SimpleNamespace(spans=[(float(s), float(e))], start=float(s), end=float(e), event_label=fam),
                              str(vid), M[0], M[1], log=log)
    finally:
        reason.SCENE_FIT_PROMPT = old
    return v, log


def pop415():
    R = json.loads((GOLD / "weakwitness2_415.json").read_text(encoding="utf-8"))["rows"]
    rows = sorted([r for r in R if r["ears"] == 1 and r["dasm_win"] < 0.35], key=lambda r: (r["clip"], r["onset"]))
    assert len(rows) == 90, len(rows)
    return rows


def mp4(c):
    return _ROOT / "data" / "input" / "audioset_heldout" / f"{c}.mp4"


def videos():
    m = {}
    for d in (WORK / "r13" / f"{B}_proposed", WORK / "r13dev2" / f"{B}_proposed"):
        for p in sorted(d.glob("*/media.json")):
            m[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))["video_path"]
    miss = [c for c, v in m.items() if not Path(v).exists()]
    MAP.parent.mkdir(parents=True, exist_ok=True)
    MAP.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(len(m), "clips; missing mp4", miss, MAP, flush=True)


def sanity():
    vids = json.loads(MAP.read_text(encoding="utf-8"))
    items = [(c, f, s, e, vids.get(c)) for c, f, s, e in DEV5] + [(r["clip"], r["family"], r["onset"], r["end"], str(mp4(r["clip"])))
                                                                     for r in pop415()[:5]]
    reason, M = vlm()
    p2 = reason.SCENE_FIT_PROMPT.replace("Answer yes or no.", ORDER2)
    assert p2 != reason.SCENE_FIT_PROMPT
    res = []
    for c, f, s, e, v in items:
        v1, l1 = ask(reason, M, v, f, s, e)
        v2, l2 = ask(reason, M, v, f, s, e, prompt=p2)
        res.append({"clip": c, "family": f, "span": [s, e], "order1": v1, "raw1": l1, "order2": v2, "raw2": l2})
        print("[sanity]", c, f, s, e, "| yes/no:", v1, l1, "| no/yes:", v2, l2, flush=True)
    from collections import Counter
    pairs = Counter((x["order1"], x["order2"]) for x in res)
    top = pairs.most_common(1)[0][1]
    agree = sum(x["order1"] == x["order2"] for x in res)
    stop = top >= 9
    out = {"items": res, "pairs": {str(k): n for k, n in pairs.items()}, "most_common_pair_n": top, "order_agreement": agree,
           "stop_untestable": stop}
    (GOLD / "scenemargin_sanity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[sanity] pairs", dict(pairs), "most common", top, "/ 10; order agreement", agree, "/ 10; STOP" if stop else "; GO", flush=True)
    sys.exit(3 if stop else 0)


def gate():
    from benchmark.gold import heldout_a4_screen as H
    rows = pop415()
    reason, M = vlm()
    for r in rows:
        r["scene"], r["scene_raw"] = ask(reason, M, mp4(r["clip"]), r["family"], r["onset"], r["end"])
        print("[gate]", r["clip"], r["family"], r["onset"], r["end"], r["class"], "->", r["scene"], r["scene_raw"], flush=True)

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    cr, nc, na = [r for r in rows if r["scene"] is True], [r for r in rows if r["scene"] is False], [r for r in rows if r["scene"] is None]
    sc, sn = S(cr), S(nc)
    go = (len(cr) >= 10 and len(nc) >= 10 and sc["precision"] is not None and sc["precision"] >= 0.375
          and sn["precision"] is not None and sn["precision"] <= 0.20)
    res = {"what": "Round 60 SCENE-MARGIN step 1 (415)", "all": S(rows), "credible": sc, "not_credible": sn, "none": len(na),
           "credible_omni_only": S([r for r in cr if r["omni"]]), "credible_afn_only": S([r for r in cr if r["afn"]]),
           "not_credible_omni_only": S([r for r in nc if r["omni"]]), "not_credible_afn_only": S([r for r in nc if r["afn"]]),
           "by_class": {k: {"credible": sum(r["class"] == k and r["scene"] is True for r in rows),
                            "not": sum(r["class"] == k and r["scene"] is False for r in rows)}
                        for k in sorted({r["class"] for r in rows})},
           "go": go, "rows": rows}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("[gate] all", res["all"], "\n credible", sc, "\n not credible", sn, "\n none", len(na), "\n by class", res["by_class"],
          "\n GO" if go else "\n STOP", flush=True)
    sys.exit(0 if go else 3)


def diff():
    from benchmark.gold import weakwitness_dev as W
    ans = {}
    if ANS.exists():
        for ln in ANS.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                x = json.loads(ln)
                ans[tuple(x["key"])] = x
    res = {"asked": len(ans), "credible": sum(x["verdict"] is True for x in ans.values()),
           "not": sum(x["verdict"] is False for x in ans.values()), "none": sum(x["verdict"] is None for x in ans.values()),
           "answers": list(ans.values()), "vs": {}}
    P = W.parts([CP, B, C])
    for ref in (B, C):
        out = []
        for part, (gold, Pk, _c, _s) in P.items():
            for st in Pk[CP]:
                ca, cb = W.classify(gold[st], Pk[CP][st]), W.classify(gold[st], Pk[ref][st])
                ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
                kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
                if ka != kb:
                    asks = [x for k, x in ans.items() if k[0] == st]
                    out.append({"part": part, "clip": st, "only_" + CP: sorted(ka - kb, key=lambda x: x[1]),
                                "only_" + ref: sorted(kb - ka, key=lambda x: x[1]), "scene_asks": asks})
                    print(f"[diff vs {ref}]", part, st, "\n   +", sorted(ka - kb, key=lambda x: x[1]), "\n   -",
                          sorted(kb - ka, key=lambda x: x[1]), "\n   asks", [(x["key"], x["verdict"], x["answers"]) for x in asks],
                          flush=True)
        res["vs"][ref] = out
    print("[diff] asked", res["asked"], "credible", res["credible"], "not", res["not"], "none", res["none"], flush=True)
    p = GOLD / "scenemargin_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    {"videos": videos, "sanity": sanity, "gate": gate, "diff": diff}[sys.argv[1]]()
