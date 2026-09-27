"""Data for the inspector page (docs/inspector/): every clip, gold sound and shown picture of the scored sets, with the
official scorer's classes, the gate's votes and the detector's events. Re-presents existing renders; decides nothing
(docs/prereg_v4.md, 2026-09-28, TEST exposure 11: inspection).

    python benchmark/gold/inspector_data.py        # cluster (needs the renders) -> docs/inspector/data.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

WORK = _ROOT / "data" / "work"
SETS = {"DEV": ("dev", "dev_monocap_v31"), "TEST": ("test_bench", "test_final_v33"), "sliceB": ("sliceB", "sliceB_v32")}
SYSTEMS = {"ours": "proposed", "blind": "blind_a2i"}
OUT = _ROOT / "docs" / "inspector" / "data.json"


def classify(gold, pics, early=S.EARLY, late=S.LATE):
    """the scorer's matching (score_per_sound.score_clip), keeping per-sound outcomes and per-picture classes"""
    order = sorted(range(len(gold)), key=lambda i: gold[i]["start"])
    g = [gold[i] for i in order]
    pics = sorted(pics, key=lambda p: p[1])
    taken = [False] * len(g)
    scored = lambda x: x["importance"] >= S.MIN_IMPORTANCE
    snd = [None] * len(g)
    pout = []
    for lab, a, b in pics:
        cands = [i for i, x in enumerate(g) if S.same_family(lab, x["label"]) and S.in_window(a, x["start"], early, late)]
        if cands:
            free = [i for i in cands if not taken[i]]
            if not free:
                pout.append({"label": lab, "start": a, "end": b, "class": "duplicate"}); continue
            i = min(free, key=lambda i: abs(a - g[i]["start"]))
            covered = [i] + [j for j in cands if not taken[j] and S.same_family(g[j]["label"], g[i]["label"])]
            hit_here = False
            for j in covered:
                taken[j] = True
                if g[j]["needed"] and scored(g[j]):
                    snd[j] = {"outcome": "hit", "late": round(a - g[j]["start"], 2)}; hit_here = True
                elif g[j]["needed"]:
                    snd[j] = {"outcome": "don't care (importance 1)"}
            if any(not g[j]["needed"] for j in covered):
                cls = "hit (+ a visible sound of the same family)" if hit_here else "wrong: source visible or obvious"
            else:
                cls = "hit" if hit_here else "don't care"
            pout.append({"label": lab, "start": a, "end": b, "class": cls,
                         "sound": [order[j] for j in covered]})
        else:
            any_sound = any(S.in_window(a, x["start"], early, late) or (x["start"] <= a <= x["end"]) for x in g)
            pout.append({"label": lab, "start": a, "end": b,
                         "class": "wrong: a different sound" if any_sound else "wrong: no such sound"})
    res = [None] * len(gold)
    for k, i in enumerate(order):
        x = g[k]
        if snd[k] is not None:
            res[i] = snd[k]
        elif x["needed"] and scored(x):
            res[i] = {"outcome": "miss"}
        elif x["needed"]:
            res[i] = {"outcome": "don't care (importance 1)"}
        else:
            res[i] = {"outcome": "not needed (" + ("visible" if x["visible"] else "obvious") + ")"}
    return res, pout


def rd(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    subs = S.subsets_of(gold)
    out = {"sets": {}, "clips": [], "note": "Scorer classes as benchmark/gold/score_per_sound.py; window [-0.5, +1.0] s; "
           "needed = not visible and not obvious; only needed sounds of importance 2-3 are scored."}
    for split, (sub, tag) in SETS.items():
        stems = sorted(subs[sub])
        rows = {k: [] for k in SYSTEMS}
        rows["silence"] = []
        for stem in stems:
            gs = gold[stem]
            clip = {"clip": stem, "split": split, "tag": tag, "category": S.category(gs),
                    "sounds": [dict(x) for x in gs], "systems": {}}
            for name, sysname in SYSTEMS.items():
                wd = WORK / f"protocol_{sysname}_{tag}" / stem
                pics = S.load_pictures(WORK / f"protocol_{sysname}_{tag}", stem, sysname)
                if pics is None:
                    continue
                per_sound, per_pic = classify(gs, pics)
                rows[name].append(S.score_clip(gs, pics))
                entry = {"sounds": per_sound, "pictures": per_pic}
                if name == "ours":
                    entry["gate"] = rd(wd / "gate_votes.json")
                    entry["events"] = [{k: e[k] for k in ("label", "start", "end", "confidence")} for e in (rd(wd / "events.json") or [])]
                    entry["augmentations"] = [{k: a.get(k) for k in ("event_label", "start", "end", "augment", "reason", "subject", "source")}
                                              for a in (rd(wd / "augmentations.json") or [])]
                    entry["media"] = rd(wd / "media.json")
                clip["systems"][name] = entry
            rows["silence"].append(S.score_clip(gs, []))
            out["clips"].append(clip)
        stats = {}
        for name, rr in rows.items():
            if not rr:
                continue
            agg = S.aggregate(rr)
            stats[name] = {k: v for k, v in agg.items() if isinstance(v, (int, float))}
        needed = sum(1 for st in stems for x in gold[st] if x["needed"] and x["importance"] >= 2)
        out["sets"][split] = {"tag": tag, "clips": len(stems), "sounds": sum(len(gold[st]) for st in stems),
                              "needed_2_3": needed,
                              "visible": sum(1 for st in stems for x in gold[st] if x["visible"]),
                              "obvious_not_visible": sum(1 for st in stems for x in gold[st] if x["obvious"] and not x["visible"]),
                              "categories": {c: sum(1 for st in stems if S.category(gold[st]) == c) for c in ("unseen", "mixed", "seen", "no_ambient")},
                              "stats": stats}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["sets"].items():
        print(k, v["clips"], "clips", v["needed_2_3"], "needed;", {n: (s.get("hits"), s.get("misses"), round(s.get("viewer_cost", 0), 2)) for n, s in v["stats"].items()})
    print("->", OUT)


if __name__ == "__main__":
    main()
