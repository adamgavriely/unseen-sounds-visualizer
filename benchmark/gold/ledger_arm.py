"""Per-item ledger of the shipped pipeline (SHIP8) on merged DEV: every missed needed sound and every wrong picture,
with the scorer's own matching (score_per_sound.score_clip, replicated item by item and checked against its counts).

    TG_ARMS=SHIP8 python benchmark/gold/ledger_ship8.py      (from ~/MscProj_tg)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import merged_dev as M
from benchmark.gold import score_per_sound as S

ARM = sys.argv[1] if len(sys.argv) > 1 else "SHIP8"
LOG = []
_orig = S.score_clip


def items(st, gold, pics):
    gold = sorted(gold, key=lambda g: g["start"]); pics = sorted(pics, key=lambda p: p[1])
    taken, matched, out = [False] * len(gold), set(), []
    scored = lambda g: S.OLD_RULE or g["importance"] >= S.MIN_IMPORTANCE
    for lab, a, b in pics:
        cands = [i for i, g in enumerate(gold) if S.same_family(lab, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE)]
        if cands:
            free = [i for i in cands if not taken[i]]
            if not free:
                continue
            i = min(free, key=lambda i: abs(a - gold[i]["start"]))
            covered = [i] + [j for j in cands if not taken[j] and S.same_family(gold[j]["label"], gold[i]["label"])]
            hit = False
            for j in covered:
                taken[j] = True
                if gold[j]["needed"]:
                    matched.add(j); hit = hit or scored(gold[j])
            if any(not gold[j]["needed"] for j in covered) and not hit:
                out.append({"clip": st, "kind": "wrong", "type": "visible", "picture": lab, "at": round(a, 2),
                            "gold": gold[i]["label"]})
        else:
            near = [g for g in gold if S.in_window(a, g["start"], S.EARLY, S.LATE) or g["start"] <= a <= g["end"]]
            out.append({"clip": st, "kind": "wrong", "type": "cross" if near else "phantom", "picture": lab, "at": round(a, 2),
                        "gold": ", ".join(f"{g['label']} {g['start']:.1f}-{g['end']:.1f}{'' if g['needed'] else ' (seen)'}" for g in near)})
    for i, g in enumerate(gold):
        if g["needed"] and scored(g) and i not in matched:
            same = [(lab, round(a, 2)) for lab, a, b in pics if S.same_family(lab, g["label"])]
            out.append({"clip": st, "kind": "miss", "sound": g["label"], "at": round(g["start"], 2), "end": round(g["end"], 2),
                        "importance": g["importance"], "same_family_pictures": same})
    return out


def wrapped(gold, pics, *a, **k):
    LOG.append((gold, pics))
    return _orig(gold, pics, *a, **k)


def main():
    S.score_clip = wrapped
    dev = M.rows_dev([ARM]); n1 = len(LOG)
    dev2 = M.rows_dev2([ARM])
    S.score_clip = _orig
    stems = [st for st, _ in dev[ARM]] + [st for st, _ in dev2[ARM]]
    parts = ["dev"] * n1 + ["dev2"] * (len(LOG) - n1)
    rows = []
    for (gold, pics), st, part in zip(LOG, stems, parts):
        its = items(st, gold, pics)
        r = _orig(gold, pics)
        assert sum(x["kind"] == "miss" for x in its) == r["miss"], st
        assert sum(x["kind"] == "wrong" for x in its) == r["visible"] + r["cross"] + r["phantom"], st
        for x in its:
            x["part"] = part
        rows += its
    miss = [x for x in rows if x["kind"] == "miss"]; wrong = [x for x in rows if x["kind"] == "wrong"]
    print(f"{ARM}: misses {len(miss)}, wrong {len(wrong)}")
    Path(_ROOT / "benchmark" / "gold" / f"ledger_{ARM.replace('+', '_')}.json").write_text(json.dumps({"arm": ARM, "misses": miss, "wrong": wrong}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
