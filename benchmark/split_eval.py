"""Tune on one half of the benchmark, report on the other.

Every threshold in this project was chosen by sweeping it over the same 274 clips the
result was then quoted on. That inflates the number by an unknown amount, and it is the
first thing an examiner will ask about, because the sweep has five display values and
five augment values and the benchmark has a few hundred clips: there is real room to fit
noise.

So: split the labelled clips 50/50, stratified by the annotator's tag so both halves
contain the same mix of scenarios, with a fixed seed. Sweep on the training half only,
freeze the winning thresholds, and report them on the held-out half. Three numbers come
out, and the gap between them is the finding:

  train (best)   what the sweep achieves where it was allowed to look -- optimistic
  test (frozen)  the same thresholds on clips the sweep never saw -- honest
  test (config)  the thresholds currently in config.py, on the same held-out half

Usage:
    python -m benchmark.split_eval
    python -m benchmark.split_eval --seed 11      # a different split
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from benchmark.evaluate import SHOULD_AUGMENT, _score, CACHE, TAGS

DISPLAY_GRID = (0.08, 0.10, 0.12, 0.15)
AUGMENT_GRID = (0.12, 0.15, 0.20, 0.25, 0.30)


def split(tags: dict, seed: int):
    """Stratified 50/50 by human tag, so neither half is scenario-skewed."""
    rng = random.Random(seed)
    by_tag = defaultdict(list)
    for k, v in tags.items():
        by_tag[v["tag"]].append(k)
    train, test = {}, {}
    for tag, keys in sorted(by_tag.items()):
        keys.sort()
        rng.shuffle(keys)
        half = len(keys) // 2
        for k in keys[:half]:
            train[k] = tags[k]
        for k in keys[half:]:
            test[k] = tags[k]
    return train, test


def best_on(tags: dict, cache: dict):
    """The (display, augment) pair maximising F1, as the original sweep chose them."""
    best, best_f1 = None, -1.0
    for d in DISPLAY_GRID:
        for a in AUGMENT_GRID:
            if a < d:
                continue
            s = _score(tags, cache, d, a)
            if s["f1"] > best_f1:
                best, best_f1 = (d, a), s["f1"]
    return best


def line(label, s):
    print(f"  {label:22}{s['n_clips']:>5}{s['accuracy']:>10.1%}{s['precision']:>11.1%}"
          f"{s['recall']:>9.1%}{s['f1']:>8.1%}")


def main():
    seed = 7
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    tags = json.loads(TAGS.read_text(encoding="utf-8"))
    usable = {k: v for k, v in tags.items() if v.get("tag") in SHOULD_AUGMENT}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if not cache:
        sys.exit("no eval cache -- run benchmark.evaluate first")

    train, test = split(usable, seed)
    d, a = best_on(train, cache)
    print(f"[split_eval] seed={seed}  train={len(train)} test={len(test)} clips")
    print(f"[split_eval] thresholds chosen on TRAIN only: display={d} augment={a}")
    print(f"[split_eval] config.py currently has: display={config.DISPLAY_THRESHOLD} "
          f"augment={config.AUGMENT_THRESHOLD}\n")

    print(f"  {'set':22}{'n':>5}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>8}")
    tr = _score(train, cache, d, a)
    te = _score(test, cache, d, a)
    cf = _score(test, cache, config.DISPLAY_THRESHOLD, config.AUGMENT_THRESHOLD)
    line("train (best, tuned)", tr)
    line("test (frozen)", te)
    line("test (config.py)", cf)

    gap = tr["f1"] - te["f1"]
    print(f"\n  optimism from tuning on the test set: {gap:+.1%} F1")
    print("  (train minus held-out, same thresholds -- how much the swept number "
          "flattered itself)")


if __name__ == "__main__":
    main()
