"""Free sweep of the gate's decision rule on the cached votes (amendment 7, 2026-09-22).

The gate is a veto stack: three votes per stretch, majority wins, and a sound is silenced only if it
is judged visible in EVERY stretch. Each layer can only lower sensitivity, which is exactly the
measured 0.43 / 0.88 fingerprint. The raw votes are cached for every gold sound
(benchmark/gold/gate_gold), so every rule below is re-decided on CPU at no GPU cost.

Grid (declared before reading any number): stretch rule {all, majority, any} x vote rule
{majority of the 3 shipped votes, obvious only, obvious OR majority, obvious AND majority,
any of the 4} x the same-family-overlap silencing {on, off -- off is not decidable from these
votes, so it is reported separately}. Selection rule, also declared: the highest sensitivity
(share of visible/obvious sounds silenced) subject to specificity >= 0.88 (the shipped system's
own false-silence floor), measured on DEV only; ties go to the simpler rule.

    python benchmark/gold/gate_sweep.py --arm Qwen38-27B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.gate_gold import OUT_DIR
from benchmark.gold.split import load as load_split

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
STRETCH_RULES = ("all", "majority", "any")
VOTE_RULES = ("maj3", "obvious", "obvious_or_maj3", "obvious_and_maj3", "any4")
SPEC_FLOOR = 0.88


def stretch_visible(st, vote_rule: str) -> bool:
    votes = [st.get("name"), st.get("ab"), st.get("desc")]
    yes = sum(1 for v in votes if v is True); no = sum(1 for v in votes if v is False)
    maj3 = yes > no
    ob = st.get("obvious") is True
    return {"maj3": maj3, "obvious": ob, "obvious_or_maj3": ob or maj3,
            "obvious_and_maj3": ob and maj3, "any4": maj3 or ob or yes > 0}[vote_rule]


def silenced(stretches, stretch_rule: str, vote_rule: str) -> bool:
    v = [stretch_visible(st, vote_rule) for st in stretches]
    if not v:
        return False
    return {"all": all(v), "any": any(v), "majority": sum(v) > len(v) / 2}[stretch_rule]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="Qwen38-27B")
    ap.add_argument("--half", default="dev", choices=["dev", "test", "all"])
    a = ap.parse_args()
    gold = S.load_gold([GOLD])
    dev, test = load_split()
    keep = {"dev": dev, "test": test, "all": set(gold)}[a.half]
    files = {f.stem: f for f in (OUT_DIR / a.arm).glob("*.json")}
    rows = []
    for stretch_rule in STRETCH_RULES:
        for vote_rule in VOTE_RULES:
            tp = fn = tn = fp = 0
            for stem in sorted(s for s in gold if s in keep and s in files):
                for c in json.loads(files[stem].read_text(encoding="utf-8"))["sounds"]:
                    if c["importance"] < 2:
                        continue
                    sil = silenced(c["stretches"], stretch_rule, vote_rule)
                    if c["seen"]:
                        tp += int(sil); fn += int(not sil)          # visible/obvious: silencing is right
                    else:
                        fp += int(sil); tn += int(not sil)          # needed: silencing is a lost picture
            sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
            rows.append({"stretch": stretch_rule, "vote": vote_rule, "sens": sens, "spec": spec,
                         "balanced": 0.5 * (sens + spec), "seen": tp + fn, "needed": tn + fp, "lost": fp})
    rows.sort(key=lambda r: (-r["sens"], -r["spec"]))
    print(f"{a.half.upper()}  (seen {rows[0]['seen']}, needed {rows[0]['needed']} sounds rated >= 2)")
    print(f"{'stretch':9s} {'vote':18s} {'sens':>6s} {'spec':>6s} {'bal':>6s}  lost pictures")
    for r in sorted(rows, key=lambda r: (-r["balanced"],)):
        mark = " <= shipped" if (r["stretch"], r["vote"]) == ("all", "maj3") else ""
        ok = "*" if r["spec"] >= SPEC_FLOOR else " "
        print(f"{ok}{r['stretch']:8s} {r['vote']:18s} {r['sens']:6.2f} {r['spec']:6.2f} {r['balanced']:6.2f}  {r['lost']:3d}{mark}")
    elig = [r for r in rows if r["spec"] >= SPEC_FLOOR]
    if elig:
        best = max(elig, key=lambda r: (r["sens"], r["spec"]))
        print(f"\nselected (max sensitivity at specificity >= {SPEC_FLOOR}): {best['stretch']} / {best['vote']} "
              f"-- sens {best['sens']:.2f} spec {best['spec']:.2f} balanced {best['balanced']:.2f}")
    (_ROOT / "benchmark" / "gold" / f"gate_sweep_{a.arm}_{a.half}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
