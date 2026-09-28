"""DEV check of the display rules (Adam, 28 Sept): a picture stays at most 1 s past its sound's real end, and repeats are
joined only if the gap is <= 1 s; plus the confidence floor 0.40. Scored renders, official scorer, onset rule.

    python benchmark/gold/display_rule_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S


def main():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    dev = sorted(S.subsets_of(gold)["dev"])
    steps = (("scored", {}), ("display: 1 s after end, join <= 1 s", {"MAX_AFTER_END": 1.0, "MERGE_GAP": 1.0}),
             ("+ confidence floor 0.40", {"PICTURE_MIN_CONF": 0.40}))
    for name, sets in steps:
        for k, v in sets.items():
            setattr(config, k, v)
        for sysn in ("proposed", "blind_a2i"):
            root = _ROOT / "data" / "work" / f"protocol_{sysn}_dev_monocap_v31"
            a = S.aggregate([S.score_clip(gold[st], S.load_pictures(root, st, sysn) or []) for st in dev])
            wrong = a["visible"] + a["cross"] + a["phantom"]
            print(f"{name:38s} {sysn:9s} hits {a['hits']}/36 wrong {wrong} dup {a['dup']} F1 {a['F1']:.3f} cost {a['viewer_cost']:.2f}")


if __name__ == "__main__":
    main()
