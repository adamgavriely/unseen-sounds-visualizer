"""Final system vs the proposal's baselines, on the merged development set (71 clips) and the merged test set (88 clips).

Baselines (project proposal, Section 7 "Baseline Systems"):
  * direct audio-to-image generation  = the same detector and picture model, every heard sound drawn, no look at the
                                         video ("blind_a2i" system of the harness, the final arm's own run);
  * audio captioning                  = the same decisions shown as text; per sound it is scored exactly like the
                                         audio-to-image baseline (a caption is a hit or wrong at the same moments);
  * show nothing                      = no augmentation at all (cost 4 x needed sounds per clip).
Paired clip bootstrap, 100,000 draws, seed 0, two-sided p. Run from the scoring checkout (needs the stage-5 outputs):
    python benchmark/gold/final_vs_baselines.py            -> benchmark/gold/final_vs_baselines.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S

ARM = "SHIP8+MD3+WW5+SL"
SYSTEMS = ("proposed", "blind_a2i")
OUT = _ROOT / "benchmark" / "gold" / "final_vs_baselines.json"


def _part(R, out_dir, gold, stems, sysn):
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        P = {st: S.load_pictures(out_dir / f"{ARM}_{sysn}", st, sysn) or [] for st in stems}
    return [S.score_clip(gold[st], P[st]) for st in stems if st in gold]


def _batch2(split, sysns):
    from benchmark.gold import clip_prep as T
    DCC2, R2, stems = T.configure(split)
    keep = set(stems)
    d = json.loads(T.GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out(split) / f"{split}_gold_only.json"
    DCC2.dump(tmp, d)
    gold = T._REAL_LOAD_GOLD([tmp])
    return {s: _part(R2, T.out(split), gold, stems, s) for s in sysns}


def dev_rows():
    from benchmark.gold import dev_harness as R
    gold, stems = DCC.dev_stems()
    a = {s: _part(R, R.R13, gold, stems, s) for s in SYSTEMS}
    b = _batch2("dev2", SYSTEMS)
    return {s: a[s] + b[s] for s in SYSTEMS}


def test_rows():
    from benchmark.gold import final_test as FT
    F, T, R, _a, fin = FT.old_setup(ARM)
    from benchmark.gold import clip_prep as TP0
    gold1 = TP0._REAL_LOAD_GOLD([F.GOLD])     # the real loader (clip_prep blocks S.load_gold in its own steps)
    stems1 = list(T.STEMS)
    a = {s: _part(R, fin, gold1, stems1, s) for s in SYSTEMS}
    from benchmark.gold import clip_prep as TP
    for k, v in FT._ARMS0.items():
        R.ARMS[k] = dict(v)
    TP._ORIG.clear()
    b = _batch2("test2", SYSTEMS)
    return {s: a[s] + b[s] for s in SYSTEMS}


def boot2(d, n=100000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means >= 0).mean(), (means <= 0).mean()))
    return [float(d.mean()), float(lo), float(hi), float(p)]


def summarise(rows):
    res = {"rows": {}, "diff": {}}
    cost = {}
    for s in SYSTEMS:
        m = DCC.metrics(rows[s])
        res["rows"][s] = {k: m[k] for k in ("hits", "misses", "wrong", "visible", "cross", "phantom", "viewer_cost")}
        cost[s] = np.array([DCC.clip_cost(r) for r in rows[s]])
    nothing = np.array([4.0 * S_hits_misses(r) for r in rows["proposed"]])
    res["rows"]["show_nothing"] = {"viewer_cost": float(nothing.mean())}
    res["diff"]["final_vs_audio_to_image"] = boot2(cost["proposed"] - cost["blind_a2i"])
    res["diff"]["final_vs_show_nothing"] = boot2(cost["proposed"] - nothing)
    res["diff"]["audio_to_image_vs_show_nothing"] = boot2(cost["blind_a2i"] - nothing)
    res["clips"] = len(cost["proposed"])
    return res


def S_hits_misses(r):
    m = DCC.metrics([r])
    return m["hits"] + m["misses"]


def main():
    out = {"arm": ARM, "development": summarise(dev_rows()), "test": summarise(test_rows())}
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for part in ("development", "test"):
        r = out[part]
        print(part, r["clips"], "clips")
        for s, x in r["rows"].items():
            print("  ", s, x)
        for k, v in r["diff"].items():
            print("  ", k, "d %+.3f [%+.3f, %+.3f] p %.4f" % tuple(v))
    print("->", OUT)


if __name__ == "__main__":
    main()
