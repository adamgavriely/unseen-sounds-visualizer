"""Round 43c HUMAN-ADD (docs/prereg_round13_detector_push.md): HUMAN-2 as an ADD-seen rule, CPU only, from the cached
Round 43b replies. Sound seen iff the shipped majority says seen (every stretch) OR HUMAN-2 (b) is yes in both orders on
more than half of its stretches. Then: which SHIP8 placed pictures (merged DEV) would the added "seen" silence?

    TG_ARMS=SHIP8 python benchmark/gold/human_add_screen.py   -> benchmark/gold/human_add_screen.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import human_gate as H
from benchmark.gold import human2_gate as H2
from benchmark.gold.box_gate import gold_index, majority

OUT = _ROOT / "benchmark" / "gold" / "human_add_screen.json"


def seen_add(stretches) -> bool:
    return all(majority(st) for st in stretches) or H2.sound_majority(stretches, H.seen_human)


def gate_screen():
    gold = gold_index()
    files = sorted(H2.OUT.glob("*.json"))
    assert len(files) == 49, len(files)
    base = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0}
    c = dict(base); flips = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None or g["importance"] < 2:
                continue
            b = all(majority(st) for st in s["stretches"]); p = seen_add(s["stretches"])
            for tab, pred in ((base, b), (c, p)):
                if g["seen"]:
                    tab["seen"] += 1; tab["seen_sil"] += pred
                else:
                    tab["needed"] += 1; tab["needed_kept"] += not pred
            if p != b:
                flips.append([f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED", "silenced" if p else "kept",
                              [[(st.get("human") or {}).get("raw"), (st.get("human") or {}).get("act_replies")] for st in s["stretches"]]])
    assert base == H.BASE, base
    go = (c["seen_sil"] >= 19 and c["needed_kept"] >= 32) or (c["needed_kept"] >= 35 and c["seen_sil"] >= 15)
    print(f"[base     ] seen silenced {base['seen_sil']}/{base['seen']} | needed kept {base['needed_kept']}/{base['needed']}")
    print(f"[HUMAN-ADD] seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']} -> {'GO' if go else 'STOP'}")
    for x in flips:
        print("   ", x)
    return {"base": base, "human_add": c, "go": go, "flips": flips}


def pictures():
    """SHIP8 placed pictures of merged DEV: silenced iff a cached HUMAN-2 sound of the same family has a stretch whose
    onset is within +/- 0.5 s of the picture start and HUMAN-2 says seen on more than half of that sound's stretches."""
    from benchmark.gold import btp_screen as B
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    B.ARM = "SHIP8"
    cache = {}
    for f in H2.OUT.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        cache[f.stem] = [(s["label"], [(float(st["start"]), float(st["end"])) for st in s["stretches"]],
                          H2.sound_majority(s["stretches"], H.seen_human)) for s in d["sounds"]]
    rows, cnt = [], {"pictures": 0, "no cached clip": 0, "no matching sound": 0, "matched, not seen": 0, "silenced": 0}
    for pt, st, g, pics in B.parts():
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        for l, a, b, resc in pics:
            cnt["pictures"] += 1
            if st not in cache:
                cnt["no cached clip"] += 1; continue
            m = [(lab, seen) for lab, strs, seen in cache[st] if S.same_family(l, lab) and any(abs(s0 - a) <= 0.5 for s0, _ in strs)]
            if not m:
                cnt["no matching sound"] += 1; continue
            sil = any(seen for _, seen in m)
            cnt["silenced" if sil else "matched, not seen"] += 1
            if sil:
                rows.append({"part": pt, "clip": st, "picture": l, "start": round(a, 2), "class": before[(l, round(a, 3))],
                             "matched": [lab for lab, _ in m]})
    print("SHIP8 pictures:", cnt)
    for r in rows:
        print("   ", r)
    return {"counts": cnt, "silenced": rows}


if __name__ == "__main__":
    res = gate_screen()
    res["pictures"] = pictures()
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
