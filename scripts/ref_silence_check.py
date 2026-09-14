"""Does the independent reference's silence decision agree with the human tag?

The reference says "nothing beyond the picture" when every verified sound is visible (or
nothing was heard); the human tag says silence is right on seen_ambient and no_ambient
clips. Agreement between the two is the cheapest check that the reference can judge the
gate at all. The list-and-match visibility step scored 55 % -- chance -- and a review
set the bar for running a judge pass at 70 %.

    python scripts/ref_silence_check.py v2            # exit 1 below the bar
    python scripts/ref_silence_check.py v2 --bar 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SILENT_RIGHT = {"seen_ambient", "no_ambient"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--bar", type=float, default=0.70)
    a = ap.parse_args()
    ref = json.loads((_ROOT / "benchmark" / f"protocol_reference_indep_{a.tag}.json").read_text("utf-8"))
    refs = ref["references"]
    desc = json.loads((_ROOT / "benchmark" / f"protocol_descriptions_{a.tag}.json").read_text("utf-8"))
    tag = {r["clip"]: r.get("human_tag") for r in desc}
    per = {}
    agree = n = 0
    for clip, r in refs.items():
        t = tag.get(clip)
        if t is None:
            continue
        silent = "nothing beyond" in r["reference"].lower()
        right = silent == (t in SILENT_RIGHT)
        per.setdefault(t, [0, 0]); per[t][0] += right; per[t][1] += 1
        agree += right; n += 1
    acc = agree / max(1, n)
    print(f"[silence-check] {a.tag}: reference silent == human silent on {agree}/{n} = {acc:.0%}")
    for t, (k, m) in sorted(per.items()):
        print(f"    {t:15s} {k:3d}/{m:<3d}")
    if acc < a.bar:
        sys.exit(f"[silence-check] below the {a.bar:.0%} bar; not judging")


if __name__ == "__main__":
    main()
