"""Fresh AudioSet set (340 of the 422 clips of audioset_fresh.json, see "subset_rule" in the output): frozen D' (SHIP8+MD3+WW5+SL, "proposed") vs the
proposal's direct audio-to-image baseline ("blind_a2i", the same run) vs show nothing. REPORT ONLY: nothing is tuned here.

Gold rule (fixed 4 Oct before any score was read): detector_round12.gold2 with its pre-registered v2 defaults (names by
MID, consequential AND depictable AND not music, same-family runs merged when the pause <= 2.0 s) -> needed sounds
(importance 2). Every other labelled non-speech, non-music event (merged the same way) -> importance 1 = "don't care" in
score_per_sound (a picture of it is neither a hit nor wrong; it is never a miss), as in the v2 false-span rule.
Scored with score_per_sound.score_clip (window -0.5/+1.0 s of onset) through its own loader. Brief's counts:
wrong = visible + cross + phantom + dup (second picture of a sound counts as wrong); cost = (4 miss + 2 wrong) / clips.
Paired clip bootstrap, 100,000 draws, seed 0, two-sided p (final_vs_baselines.boot2).

Ran on 4 Oct 2026 in the cluster scoring checkout (v1.2.0 file names: tagger_prep.py = clip_prep.py, and
benchmark/detector_round12.py, benchmark/setup_audit.py, benchmark/gold/audioset_fresh.json from v1.2.0), after each
shard was run with slurm/run_best.sh steps (frozen arm, unchanged):
    TG_EXTRA_SPLITS="freshf00 ... freshf08" TG_ARMS="SHIP8+MD3+WW5+SL" python benchmark/gold/fresh_side_by_side.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import detector_round12 as R12
from benchmark.gold import tagger_prep as T
from benchmark.gold import score_per_sound as S

ARM = "SHIP8+MD3+WW5+SL"
SYSTEMS = ("proposed", "blind_a2i")
SRC = _ROOT / "benchmark" / "gold" / "audioset_fresh.json"
WORK = _ROOT / "data" / "work" / "fresh_side_by_side"
OUT = _ROOT / "benchmark" / "gold" / "fresh_side_by_side.json"


def gold_export():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    clips, counts = [], {"events": 0, "needed": 0, "dontcare": 0}
    for c in d["clips"]:
        stem = Path(c["src"]).stem
        sc, allg = R12.gold2(c)                       # v2 defaults
        others = [g for g in allg if g not in sc]
        others = R12.merge_runs([{"label": g["label"], "start": g["start"], "end": g["end"]} for g in others])
        snds = [{"label": g["label"], "start": g["start"], "end": g["end"], "importance": 2} for g in sc]
        snds += [{"label": g["label"], "start": g["start"], "end": g["end"], "importance": 1} for g in others]
        counts["events"] += len(c["events"]); counts["needed"] += len(sc)
        clips.append({"clip": f"{stem}.mp4", "done": True, "sounds": snds})
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / "fresh_gold_export.json"
    p.write_text(json.dumps({"clips": clips}, indent=1), encoding="utf-8")
    gold = T._REAL_LOAD_GOLD([p])
    counts["clips"] = len(gold)
    counts["needed_loaded"] = sum(1 for g in gold.values() for s in g if s["importance"] >= 2)
    counts["dontcare_loaded"] = sum(1 for g in gold.values() for s in g if s["importance"] < 2)
    counts["unresolved"] = len(S.UNRESOLVED)
    counts["needed_depth0"] = sum(1 for g in gold.values() for s in g if s["importance"] >= 2 and not S._specific(s["label"]))
    return gold, counts


def rows_of(splits, gold):
    rows = {s: [] for s in SYSTEMS}
    stems_all, missing = [], []
    for split in splits:
        DCC, R, stems = T.configure(split)
        o = T.out(split)
        for sysn in SYSTEMS:
            with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
                for st in stems:
                    P = S.load_pictures(o / f"{ARM}_{sysn}", st, sysn)
                    if P is None:
                        missing.append(f"{split}/{sysn}/{st}"); P = []
                    rows[sysn].append(S.score_clip(gold[st], P))
        stems_all += stems
    return rows, stems_all, missing


def boot2(d, n=100000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means >= 0).mean(), (means <= 0).mean()))
    return [float(d.mean()), float(lo), float(hi), float(p)]


def clip_cost(r, dup=True):
    return 4.0 * r["miss"] + 2.0 * (r["visible"] + r["cross"] + r["phantom"] + (r["dup"] if dup else 0))


def summ(rr):
    H = sum(r["hit"] for r in rr); M = sum(r["miss"] for r in rr)
    W = sum(r["visible"] + r["cross"] + r["phantom"] + r["dup"] for r in rr)
    return {"hits": H, "needed": H + M, "wrong": W, "dup": sum(r["dup"] for r in rr),
            "cost": float(np.mean([clip_cost(r) for r in rr])),
            "cost_no_dup": float(np.mean([clip_cost(r, False) for r in rr])),
            "F1": 2 * H / (2 * H + W + M) if H + W + M else 0.0, "clips": len(rr)}


def main():
    splits = os.environ["TG_EXTRA_SPLITS"].split()
    gold, counts = gold_export()
    rows, stems, missing = rows_of(splits, gold)
    assert len(stems) == len(set(stems)) == 340, len(stems)
    res = {"subset_rule": "340 of the 422: shards f00-f08 (sorted clip-id chunks); f09-f10 (82 clips) dropped before any score was read because they were last in the A100 queue (Adam 4 Oct, time only); strata 195 complex / 145 random", "arm": ARM, "gold": counts, "missing_outputs": missing, "rows": {s: summ(rows[s]) for s in SYSTEMS}}
    nothing = np.array([4.0 * (r["hit"] + r["miss"]) for r in rows["proposed"]])
    res["rows"]["show_nothing"] = {"hits": 0, "needed": int(nothing.sum() / 4), "wrong": 0, "cost": float(nothing.mean()),
                                   "F1": 0.0}
    c = {s: np.array([clip_cost(r) for r in rows[s]]) for s in SYSTEMS}
    res["diff"] = {"final_vs_audio_to_image": boot2(c["proposed"] - c["blind_a2i"]),
                   "final_vs_show_nothing": boot2(c["proposed"] - nothing),
                   "audio_to_image_vs_show_nothing": boot2(c["blind_a2i"] - nothing)}
    res["per_clip"] = {st: {s: rows[s][i] for s in SYSTEMS} for i, st in enumerate(stems)}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("gold", counts, "missing outputs", len(missing))
    for s, x in res["rows"].items():
        print(f"  {s:14s}", x)
    for k, v in res["diff"].items():
        print(f"  {k:32s} d %+.3f [%+.3f, %+.3f] p %.5f" % tuple(v))
    print("->", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gold":
        print(gold_export()[1])
    else:
        main()
