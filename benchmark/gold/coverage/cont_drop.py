"""Continuation drop (Adam 10 Oct: keep trying). Many remaining "different sound" wrong pictures start in the middle of a
sound of the same family that is already going on (a bird call 3.8 s into 28 s of birds, a siren restarting at 11 s).
Rule (display level, online): a picture is dropped when its family's FlexSED curve stayed >= theta for the whole
d seconds before its start (the sound was already there; this is not a new onset). theta in {0.2, 0.3, 0.5},
d in {1.0, 1.5, 2.5}, or off; chosen by clip-grouped 5-fold CV on all 158 clips (onset cost), scored out of fold,
on top of v1.4 (benchmark/gold/v14_score.py inputs).

    python benchmark/gold/coverage/cont_drop.py -> cont_drop.md
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
GRID = [None] + list(itertools.product((0.2, 0.3, 0.5), (1.0, 1.5, 2.5)))


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    curves = json.loads((IN / "flexsed_curves.json").read_text())
    allc = stems("dev") + stems("test")
    pics = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}

    def apply(st, prm):
        if prm is None:
            return pics[st]
        th, d = prm
        out = []
        for lab, a, b in pics[st]:
            c = ((curves.get(st) or {}).get("labels", {}).get(lab) or {}).get("flex")
            if c and c.get("t") and a - d >= 0:
                t, v = np.asarray(c["t"]), np.asarray(c["v"])
                m = (t >= a - d) & (t < a)
                if m.any() and v[m].min() >= th:
                    continue
            out.append((lab, a, b))
        return out

    memo = {}

    def row(st, prm):
        if (st, prm) not in memo:
            memo[(st, prm)] = S.score_clip(gold[st], apply(st, prm))
        return memo[(st, prm)]

    cost = lambda rr: S.viewer_cost(rr)
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh)
    oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: cost([row(st, g) for st in tr]))
        ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    def summ(rr):
        a = S.aggregate(rr); return a["hits"], a["visible"] + a["cross"] + a["phantom"], a["viewer_cost"]
    L = ["# Continuation drop, all 158 clips, clip-grouped 5-fold CV", "", f"choices per fold: {ch}",
         f"v1.4: hits / wrong / cost = {summ([row(st, None) for st in allc])}",
         f"out of fold: {summ([oof[st] for st in allc])}", "", "| theta, d (all clips) | hits | wrong | cost |", "|---|---|---|---|"]
    for g in GRID[1:]:
        h, w, c = summ([row(st, g) for st in allc]); L.append(f"| {g} | {h} | {w} | {c:.3f} |")
    (HERE / "cont_drop.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
