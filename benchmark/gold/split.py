"""The DEV / TEST split of Adam's gold set (declared 2026-09-22, amendment 7).

Adam's instruction: "only work with the gold set I annotated — you can split train/dev but only on
what I did, and keep variety". So every model choice from here on is made on DEV only and confirmed
once on TEST, and both halves carry the same mix of the four clip categories, of the two
populations (his benchmark clips and the AudioSet-Strong slice) and of the sourcing waves.

Rule (fixed here, before any number was read from it): clips are grouped by
(category, population, wave), each group is shuffled with seed 7 and dealt alternately into DEV and
TEST, so both halves hold ~half of every group. The split is written to benchmark/gold/split.json
and never recomputed — later code reads the file.

    python benchmark/gold/split.py            # write and print the table
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
OUT = _ROOT / "benchmark" / "gold" / "split.json"
SEED = 7


def wave(stem: str) -> str:
    """where the clip came from, so neither half is all one source"""
    for p, w in (("w8_", "wave8"), ("m5_", "movies"), ("m4_", "structured"), ("t1_", "trailers"),
                 ("b3_", "batch3"), ("mv_", "movie_scene"), ("mc_", "movie_scene"), ("ambient_", "ambient"),
                 ("as_", "audioset_event"), ("un_", "unsplash"), ("ly_", "library"), ("lx_", "library"),
                 ("oc_", "outdoor"), ("ev_", "everyday"), ("movie_", "movie_scene"), ("pdfilm_", "movie_scene")):
        if stem.startswith(p):
            return w
    return "other"


def build():
    import numpy as np
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for stem, snds in gold.items():
        pop = "sliceB" if stem in subs["sliceB"] else "bench"
        groups[(S.category(snds), pop, wave(stem))].append(stem)
    dev, test = [], []
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        for i, stem in enumerate(members):
            (dev if i % 2 == 0 else test).append(stem)
    out = {"seed": SEED, "rule": "stratified by (category, population, wave), alternate deal",
           "dev": sorted(dev), "test": sorted(test)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return gold, subs, out


def table(gold, subs, sp):
    print(f"{'':16s} {'DEV':>14s} {'TEST':>14s}")
    for cat in ("mixed", "unseen", "seen", "no_ambient"):
        d = [s for s in sp["dev"] if S.category(gold[s]) == cat]
        t = [s for s in sp["test"] if S.category(gold[s]) == cat]
        nd = sum(1 for s in d for x in gold[s] if x["needed"] and x["importance"] >= 2)
        nt = sum(1 for s in t for x in gold[s] if x["needed"] and x["importance"] >= 2)
        print(f"{cat:16s} {len(d):4d} clips {nd:3d} nd {len(t):4d} clips {nt:3d} nd")
    for pop in ("bench", "sliceB"):
        d = [s for s in sp["dev"] if (s in subs[pop])]
        t = [s for s in sp["test"] if (s in subs[pop])]
        print(f"{pop:16s} {len(d):4d} clips        {len(t):4d} clips")
    print("waves dev :", dict(Counter(wave(s) for s in sp["dev"])))
    print("waves test:", dict(Counter(wave(s) for s in sp["test"])))
    print("->", OUT)


def load():
    """(dev stems, test stems) -- the file is authoritative"""
    d = json.loads(OUT.read_text(encoding="utf-8"))
    return set(d["dev"]), set(d["test"])


if __name__ == "__main__":
    table(*build())
