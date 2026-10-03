"""Round 53 WEAK-WITNESS (docs/history/preregistrations/prereg_round13_detector_push.md "Round 53 WEAK-WITNESS", release v1.2.0), merged DEV helpers. Run from ~/MscProj_tg.

  python benchmark/gold/weakwitness_dev.py step0            # mechanism check: every SHIP8+MD3 hit picture -> its stage-4 row(s):
                                                            # origin, rescued, DASM max over [pre_start-0.5, end+0.5], Qwen V4 /
                                                            # AF V4 on the P1 cut (the DASM_LOCAL_VETO "both" keep). Hits only.
  python benchmark/gold/weakwitness_dev.py diff A B         # per-picture classes of arm A vs arm B, changed clips only
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")

from benchmark.gold import dev_candidates_check as DCC  # noqa: E402
from benchmark.gold import merged_dev as M  # noqa: E402
from benchmark.gold import round13_dev as R  # noqa: E402
from benchmark.gold import score_per_sound as S  # noqa: E402

SYS = "proposed"


def classify(gold, pics, early=S.EARLY, late=S.LATE):
    """score_clip's loop, per picture: (label, start, end, class) with class hit / dontcare / dup / visible / cross / phantom"""
    gold = sorted(gold, key=lambda g: g["start"]); pics = sorted(pics, key=lambda p: p[1])
    taken = [False] * len(gold)
    scored = lambda g: S.OLD_RULE or g["importance"] >= S.MIN_IMPORTANCE
    out = []
    for lab, a, b in pics:
        cands = [i for i, g in enumerate(gold) if S.same_family(lab, g["label"]) and S.in_window(a, g["start"], early, late)]
        if cands:
            free = [i for i in cands if not taken[i]]
            if not free:
                out.append((lab, a, b, "dup")); continue
            i = min(free, key=lambda i: abs(a - gold[i]["start"]))
            covered = [i] + [j for j in cands if not taken[j] and S.same_family(gold[j]["label"], gold[i]["label"])]
            hit = False; dc = False
            for j in covered:
                taken[j] = True
                if gold[j]["needed"] and scored(gold[j]):
                    hit = True
                elif gold[j]["needed"]:
                    dc = True
            out.append((lab, a, b, "hit" if hit else ("visible" if any(not gold[j]["needed"] for j in covered)
                                                       else ("dontcare" if dc else "visible"))))
        else:
            any_sound = any(S.in_window(a, g["start"], early, late) or (g["start"] <= a <= g["end"]) for g in gold)
            out.append((lab, a, b, "cross" if any_sound else "phantom"))
    return out


def parts(arms):
    """{part: (gold, {arm: {stem: pictures}}, R-module, stage4 path)} for old DEV and the tagger DEV part"""
    os.environ["TG_ARMS"] = " ".join(arms)
    gold, stems = DCC.dev_stems()
    res = {}
    P = {}
    for a in arms:
        with R.flags({k: R.arm_cfg(a)[k] for k in R.DISPLAY_KEYS}):
            P[a] = {st: S.load_pictures(R.R13 / f"{a}_{SYS}", st, SYS) or [] for st in stems}
    res["dev"] = (gold, P, dict((a, R.arm_cfg(a)) for a in arms), R.R13 / "stage4.json")
    from benchmark.gold import tagger_prep as T      # imported after the DEV gold is read (it blocks gold reads at import)
    DCC2, R2, stems2 = T.configure("dev2")
    keep = set(stems2)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out("dev2") / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    gold2 = T._REAL_LOAD_GOLD([tmp])
    o = T.out("dev2")
    P2 = {}
    for a in arms:
        with R2.flags({k: R2.arm_cfg(a)[k] for k in R2.DISPLAY_KEYS}):
            P2[a] = {st: S.load_pictures(o / f"{a}_{SYS}", st, SYS) or [] for st in stems2 if st in gold2}
    res["dev2"] = (gold2, P2, dict((a, R2.arm_cfg(a)) for a in arms), o / "stage4.json")
    return res


def step0():
    import config
    from src.labels import canonical
    from src.stage4_audio_event_detection import _v4_names_qwen, _af_p1_accepts
    from src.types import AudioEvent
    arm = "SHIP8+MD3"
    out = []
    for part, (gold, P, cfgs, s4) in parts([arm]).items():
        cfg = cfgs[arm]
        rows_all = json.loads(Path(s4).read_text(encoding="utf-8"))["arms"][f"{arm}|{SYS}"]
        for st, pics in P[arm].items():
            hits = [p for p in classify(gold[st], pics) if p[3] == "hit"]
            if not hits:
                continue
            rr = rows_all.get(st, [])
            rr = rr.get("rows", rr) if isinstance(rr, dict) else rr
            f = Path(cfg["LISTENER_DASM_DIR"]) / f"{st}.npz"
            z = np.load(f, allow_pickle=True) if f.exists() else None
            for lab, a, b, _k in hits:
                fam = canonical(lab)
                cand = [r for r in rr if canonical(r["label"]) == fam and r["start"] - 1.0 <= a <= r["end"] + 0.5]
                rec = {"part": part, "clip": st, "picture": [lab, round(a, 2), round(b, 2)], "rows": []}
                for r in cand:
                    ps = float(r.get("pre_start", r["start"])); en = float(r["end"])
                    dm = None
                    if z is not None:
                        cols = [i for i, l in enumerate([str(x) for x in z["labels"]]) if canonical(l) == fam]
                        m = (z["times"] >= ps - 0.5) & (z["times"] <= en + 0.5)
                        if cols and m.any():
                            dm = round(float(z["fw"][m][:, cols].max()), 3)
                    with R.flags(cfg):
                        config._CURRENT_CLIP = st
                        e = AudioEvent(r["label"], ps, en, float(r["conf"]))
                        q, af = bool(_v4_names_qwen(e)), bool(_af_p1_accepts(e))
                    rec["rows"].append({"label": r["label"], "pre_start": round(ps, 2), "end": round(en, 2), "conf": round(float(r["conf"]), 3),
                                        "origin": r.get("origin"), "rescued": bool(r.get("rescued")), "dasm_win": dm,
                                        "qwen_v4": q, "af_v4": af,
                                        "droppable": (not r.get("rescued")) and dm is not None and dm < 0.4 and not (q and af)})
                out.append(rec)
    for x in out:
        print(x["part"], x["clip"], x["picture"], x["rows"], flush=True)
    p = _ROOT / "benchmark" / "gold" / "weakwitness_step0.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(p)


def diff(A, B):
    res = []
    for part, (gold, P, _c, _s) in parts([A, B]).items():
        for st in P[A]:
            ca, cb = classify(gold[st], P[A][st]), classify(gold[st], P[B][st])
            ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
            kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
            if ka != kb:
                res.append({"part": part, "clip": st, "only_" + A: sorted(ka - kb, key=lambda x: x[1]),
                            "only_" + B: sorted(kb - ka, key=lambda x: x[1])})
                print(part, st, "\n   -", sorted(ka - kb, key=lambda x: x[1]), "\n   +", sorted(kb - ka, key=lambda x: x[1]), flush=True)
    p = _ROOT / "benchmark" / "gold" / "weakwitness_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    if sys.argv[1] == "step0":
        step0()
    else:
        diff(sys.argv[2], sys.argv[3])
