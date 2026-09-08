"""Report on the blind re-adjudication: is the labelling criterion stable?

Three questions, in order of how much they change the project:

  1. SELF-AGREEMENT. Does Adam give the same clip the same label twice? Reported as
     raw agreement and Cohen's kappa. If kappa is low, every number computed against
     these labels -- including the gate's 49.3% accuracy -- is measuring label noise
     as much as model error, and the honest thesis contribution shifts from "the gate
     is weak" to "the task is not yet well-defined, here is the evidence".
  2. DROP CONVERSION. With the Drop button removed, what do the old drops become? If
     a large share become unseen/mixed, the benchmark can be completed from clips
     already on disk and no further sourcing is needed.
  3. THE REASONS. Free text is the only record of the criterion actually being
     applied; 107 of 113 original drops had no reason at all.

Usage: python scripts/readjudicate_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SET_FILE = ROOT / "benchmark" / "readjudicate_set.json"
NEW = ROOT / "benchmark" / "readjudication.json"
CATS = ["unseen_ambient", "mixed", "seen_ambient", "no_ambient"]


def kappa(pairs):
    """Cohen's kappa between the original and the re-judged label."""
    if not pairs:
        return 0.0
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    if not NEW.exists():
        sys.exit("no benchmark/readjudication.json yet -- run readjudicate.bat first")
    items = {x["clip"]: x["original"] for x in json.loads(SET_FILE.read_text(encoding="utf-8"))}
    new = json.loads(NEW.read_text(encoding="utf-8"))
    print(f"re-judged {len(new)} of {len(items)} clips\n")

    # ---- 1. self-agreement on clips whose original label was a real category ----
    pairs = [(items[c], v["tag"]) for c, v in new.items()
             if c in items and items[c] in CATS]
    if pairs:
        agree = sum(1 for a, b in pairs if a == b)
        print(f"1. SELF-AGREEMENT (excluding old drops), n={len(pairs)}")
        print(f"   raw agreement {agree}/{len(pairs)} = {100*agree/len(pairs):.0f}%"
              f"   Cohen's kappa = {kappa(pairs):.2f}")
        conf = defaultdict(Counter)
        for a, b in pairs:
            conf[a][b] += 1
        print(f"   {'original':16}" + "".join(f"{c[:12]:>14}" for c in CATS))
        for a in CATS:
            if sum(conf[a].values()):
                print(f"   {a:16}" + "".join(f"{conf[a][b]:>14}" for b in CATS))
        k = kappa(pairs)
        verdict = ("labels are stable -- the gate's poor score is a MODEL problem"
                   if k >= 0.7 else
                   "moderate -- some label noise is inflating the gate's error" if k >= 0.4
                   else "labels are NOT stable -- the task definition is the problem, "
                        "and no gating metric against them is interpretable")
        print(f"   -> {verdict}")

    # ---- 2. what did the old DROPs become? ----
    drops = [(c, v) for c, v in new.items() if items.get(c) == "drop"]
    if drops:
        c = Counter(v["tag"] for _, v in drops)
        useful = c["unseen_ambient"] + c["mixed"]
        print(f"\n2. OLD DROPS re-judged, n={len(drops)}")
        print(f"   {dict(c)}")
        print(f"   became unseen/mixed: {useful}/{len(drops)} = {100*useful/len(drops):.0f}%")
        est = round(113 * useful / len(drops))
        print(f"   -> extrapolating to all 113 drops: ~{est} recoverable positives "
              f"already on disk")

    # ---- 3. the criterion, in Adam's own words ----
    print("\n3. REASONS GIVEN (the actual task definition)")
    for tag in CATS:
        rs = [v["reason"] for c, v in new.items() if v["tag"] == tag]
        if rs:
            print(f"\n   -- {tag} ({len(rs)}) --")
            for r in rs[:8]:
                print(f"      {r[:88]}")
    words = Counter()
    for v in new.values():
        for w in v["reason"].lower().split():
            if len(w) > 3:
                words[w] += 1
    print(f"\n   most frequent words: "
          f"{', '.join(f'{w}({n})' for w, n in words.most_common(12))}")


if __name__ == "__main__":
    main()
