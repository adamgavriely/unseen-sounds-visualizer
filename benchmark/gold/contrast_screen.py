"""Round 39 CONTRAST screen (docs/prereg_round13_detector_push.md): on merged-DEV P2/PV candidates (clips with gold), how
many needed-class (gold hit_needed) accepts does the forced choice add on TIER-rejected items, and how many other-class
(other_gold + none). Nothing can be lost (only rejected items are asked). GO iff needed added >= 3 and other added <= 2x.
Base check: TIER accepts must reproduce moss_screen.json (tier_needed 24 / tier_other 94 on 1004 items).

    python benchmark/gold/contrast_screen.py            # from ~/MscProj_tg on the cluster (conda msproj)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_listener as L
from benchmark.gold.moss_screen import KNOWN

G = _ROOT / "benchmark" / "gold"
PARTS = {"dev": (G / "dev_listener_contrast.json", G / "annotations" / "gold_AG.json"),
         "dev2": (G / "dev2_listener_contrast.json", G / "annotations" / "gold_AG.json")}


def main():
    tot = {"items": 0, "tier_needed": 0, "tier_other": 0, "asked": 0, "no_B": 0, "needed_added": 0, "other_added": 0}
    adds, known = [], []
    for part, (cp, gp) in PARTS.items():
        d = json.loads(cp.read_text(encoding="utf-8"))
        gold = S.load_gold([gp])
        for x in d["items"]:
            if x["clip"] not in gold:
                continue
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            needed = cls == "hit_needed"
            tot["items"] += 1
            if x["tier"]:
                tot["tier_needed" if needed else "tier_other"] += 1
                continue
            desc = (f"[{part}] {x['clip']} {x['label']!r} {x['start']:.2f}-{x['end']:.2f} peak {x['peak']:.2f} [{cls}] "
                    f"Q {int(x['qwen_v4'])} A {int(x['af_v4'])} | B {x.get('B')!r} ({x.get('B_model')} {x.get('B_score', 0) or 0:.2f}) "
                    f"AB {x.get('text_AB', '')[:30]!r} BA {x.get('text_BA', '')[:30]!r}")
            if x.get("no_B"):
                tot["no_B"] += 1
            else:
                tot["asked"] += 1
            if x["accept"].get("CONTRAST"):
                tot["needed_added" if needed else "other_added"] += 1
                adds.append(("needed" if needed else "other", desc))
            if any(c in x["clip"] and x["family"].startswith(lab) for c, lab in KNOWN):
                known.append(desc)
    assert (tot["items"], tot["tier_needed"], tot["tier_other"]) == (1004, 24, 94), tot      # moss_screen.json base
    go = tot["needed_added"] >= 3 and tot["other_added"] <= 2 * tot["needed_added"]
    for c, s in adds:
        print(f"+ {c}: {s}")
    print("known hit-side misses (TIER-rejected items):")
    for s in known:
        print("   ", s)
    print("CONTRAST screen:", json.dumps(tot), "GO" if go else "STOP")
    (G / "contrast_screen.json").write_text(json.dumps({**tot, "go": go, "added": adds, "known": known}, indent=1),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
