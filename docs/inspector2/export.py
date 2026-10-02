"""Build docs/inspector2/data.js from the pipeline's per-clip decision trails (src/trail.py -> trail.json).

Run on the cluster after the shipped arm has been run with the trail hooks on (see HOOKS.md), once per shipped version:

    python docs/inspector2/export.py --version "D′" --arm SHIP8+MD3+WW5+SL \
        --root DEV=data/work/r13/SHIP8+MD3+WW5+SL_proposed --root DEV=../MscProj_tg/data/work/r13dev2/SHIP8+MD3+WW5+SL_proposed \
        --root TEST=../MscProj_tg/data/work/r16final/SHIP8+MD3+WW5+SL_proposed --root TEST=../MscProj_tg/data/work/r13test2/SHIP8+MD3+WW5+SL_proposed

Each root holds one folder per clip with trail.json, augmentations.json and media.json. Gold: the one gold file. The
scorer is the shipped one (benchmark/gold/score_per_sound + inspector_data.classify), so hits, misses and wrong pictures
here are the same as in the reported tables; the script checks the totals against --expect if given.
Decides nothing and changes no score.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.gold import score_per_sound as S          # noqa: E402
from benchmark.gold.inspector_data import classify       # noqa: E402

GOLD = ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
VERDICT = {"hit": "hit", "hit (+ a visible sound of the same family)": "hit", "wrong: source visible or obvious": "visible",
           "wrong: a different sound": "cross", "duplicate": "cross", "wrong: no such sound": "phantom", "don't care": "hit"}


def load_steps():
    """the step catalogue (names, bars, questions) lives in sample_data.py so both files share it"""
    src = (HERE / "sample_data.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(src.split("P = lambda")[0], ns)           # only the STEPS block, not the sample cases or the write
    steps = [dict(id=a, stage=b, name=c, plain=d, bar=e, model=f, question=g) for a, b, c, d, e, f, g in ns["STEPS"]]
    content = {"build": ns["BUILD"], "lesson": ns["LESSON"], "build_steps": [dict(zip(("change", "dev", "test", "note"), r)) for r in ns["BUILD_STEPS"]],
               "tried": [dict(zip(("part", "idea", "why", "shipped"), r)) for r in ns["TRIED"]]}
    return steps, content


def key(r):
    return (r["label"], round(r["start"], 2), round(r["end"], 2))


def build_cands(trail):
    """group the flat decision records into candidate spans; a move/merge/relabel carries the id to its new span"""
    by_key, cands = {}, []
    for r in trail:
        k = key(r)
        cid = by_key.get(k)
        if cid is None:
            cid = f"c{len(cands) + 1}"
            cands.append({"id": cid, "label": r["label"], "start": r["start"], "end": r["end"],
                          "origin": r.get("extra", {}).get("origin", ""), "fate": "drawn", "trail": []})
            by_key[k] = cid
        c = cands[int(cid[1:]) - 1]
        step = {x: r[x] for x in ("step", "res", "value", "bar", "note", "asks") if x in r}
        c["trail"].append(step)
        if r["res"] == "drop":
            c["fate"], c["at"] = "dropped", r["step"]
        elif r["res"] == "merge":
            c["fate"], c["at"] = "merged", r["step"]
        if "to" in r and r["res"] in ("move", "relabel", "rescue", "pass"):
            to = r["to"]
            by_key[key(to)] = cid
            c.update(label=to["label"], start=to["start"], end=to["end"])
    return cands


def why_text(c, steps):
    last = next((s for s in reversed(c["trail"]) if s["res"] in ("drop", "merge")), None)
    if not last:
        return ""
    name = steps.get(last["step"], {}).get("name", last["step"])
    bits = [f"{name}: {last.get('value', '')}".rstrip(": ")]
    if last.get("bar"):
        bits.append(f"bar {last['bar']}")
    for a in last.get("asks", [])[:3]:
        bits.append(f"{a.get('who', 'model')} answered '{a.get('a', '')}'")
    return "; ".join(bits)


def export(roots, version, arm, out, expect=None):
    steps, content = load_steps()
    sidx = {s["id"]: s for s in steps}
    order = {s["id"]: i for i, s in enumerate(steps)}
    gold_all = S.load_gold([GOLD])
    clips, sets = [], defaultdict(lambda: defaultdict(int))
    for split, root in roots:
        root = Path(root)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            stem = d.name
            if stem not in gold_all:
                continue
            tf = d / "trail.json"
            trail = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else []
            cands = build_cands(trail)
            pics = S.load_pictures(root, stem, "proposed") or []
            gold = gold_all[stem]
            res, pout = classify(gold, pics)
            media = json.loads((d / "media.json").read_text(encoding="utf-8")) if (d / "media.json").exists() else {}
            g_out = []
            for i, (g, r) in enumerate(zip(gold, res)):
                gid = f"g{i + 1}"
                near = [c for c in cands if S.same_family(c["label"], g["label"])
                        and c["start"] <= max(g["end"], g["start"] + S.LATE) and c["end"] >= g["start"] + S.EARLY]
                row = {"id": gid, "label": g["label"], "start": g["start"], "end": g["end"], "needed": g["needed"],
                       "visible": g["visible"], "cands": [c["id"] for c in near]}
                if g["needed"]:
                    row["outcome"] = "hit" if r["outcome"] == "hit" else "miss"
                    sets[split]["needed"] += 1
                    if row["outcome"] == "hit":
                        sets[split]["hits"] += 1
                    else:
                        dead = [c for c in near if c["fate"] != "drawn"]
                        if not near:
                            row["lost_at"], row["why"] = "never_heard", "No detector produced a span of this sound type near its start."
                        elif dead and len(dead) == len(near):
                            far = max(dead, key=lambda c: order.get(c.get("at"), -1))
                            row["lost_at"], row["why"] = far.get("at"), why_text(far, sidx)
                        else:
                            row["lost_at"], row["why"] = "scorer", "A picture of this type was drawn, but not inside the −0.5 … +1.0 s window (or it was matched to another sound)."
                g_out.append(row)
            p_out = []
            for j, p in enumerate(pout):
                v = VERDICT.get(p["class"], "cross")
                link = [c for c in cands if c["fate"] == "drawn" and S.same_family(c["label"], p["label"])
                        and c["start"] <= p["end"] and c["end"] >= p["start"]]
                c0 = min(link, key=lambda c: abs(c["start"] - p["start"]), default=None)
                gi = (p.get("sound") or [None])[0]
                p_out.append({"id": f"p{j + 1}", "label": p["label"], "start": p["start"], "end": p["end"], "verdict": v,
                              "gold": f"g{gi + 1}" if gi is not None else None, "cand": c0["id"] if c0 else None})
                if v != "hit":
                    sets[split]["wrong"] += 1
                    sets[split][v] += 1
            sets[split]["clips"] += 1
            clips.append({"clip": stem, "split": split, "dur": media.get("duration"), "video": f"media/{split}/{stem}.mp4",
                          "gold": g_out, "pictures": p_out, "cands": cands, "has_trail": tf.exists()})
    for k, s in sets.items():
        s["misses"] = s["needed"] - s["hits"]
        s["cost"] = round((4 * s["misses"] + 2 * s["wrong"]) / max(1, s["clips"]), 3)
    data = {"meta": {"version": version, "arm": arm, "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     "sample": False, "window": [S.EARLY, S.LATE], "cost": "(4·miss + 2·wrong)/clips",
                     "clips_without_trail": sum(1 for c in clips if not c["has_trail"])},
            "sets": {k: dict(v) for k, v in sets.items()}, "steps": steps, "clips": clips, **content}
    if expect:
        for k, (h, w) in expect.items():
            got = sets[k]
            assert (got["hits"], got["wrong"]) == (h, w), f"{k}: export {got['hits']}/{got['wrong']} != reported {h}/{w}"
    Path(out).write_text("window.INSPECTOR2 = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {out}: " + ", ".join(f"{k} {v['hits']}/{v['needed']} hits, {v['wrong']} wrong, cost {v['cost']}" for k, v in sets.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--root", action="append", required=True, help="SPLIT=path (repeatable)")
    ap.add_argument("--expect", action="append", default=[], help="SPLIT=hits/wrong, checked against the export")
    ap.add_argument("--out", default=str(HERE / "data.js"))
    a = ap.parse_args()
    roots = [tuple(r.split("=", 1)) for r in a.root]
    exp = {e.split("=")[0]: tuple(int(x) for x in e.split("=")[1].split("/")) for e in a.expect}
    export(roots, a.version, a.arm, a.out, exp or None)
