"""How far each system is from a perfect gate, by scenario group.

The oracle uses the human label as the gate: on clips where no picture is due it takes the
best score among systems that showed nothing; where one is due, the best among systems
that showed something (scripts/paired_stats.py's definition). The gap to the oracle is
paired per clip with a bootstrap 95% CI. This is a description of the same judge scores,
not a new result.

    python scripts/oracle_gap.py v3_grounded       -> benchmark/oracle_gap_<tag>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from scripts.paired_stats import load, boot_ci, SHOULD_SHOW, SILENT_RIGHT

SYSTEMS = ("proposed", "blind_a2i", "audio_caption")
NAME = {"proposed": "gated", "blind_a2i": "blind", "audio_caption": "caption"}


def oracle_score(rec_by_sys: dict) -> float:
    """Perfect gate: correct silence scores 4 by the rubric's rule where no picture is due;
    the best system output where one is due."""
    t = rec_by_sys["proposed"].get("human_tag")
    if t in SILENT_RIGHT:
        return 4.0
    return max(rec_by_sys[s]["score"] for s in SYSTEMS)


def main(tag: str):
    data, _f = load(tag); n = len(data)
    groups = {"picture due": [c for c in data if data[c]["proposed"].get("human_tag") in SHOULD_SHOW],
              "no picture due": [c for c in data if data[c]["proposed"].get("human_tag") in SILENT_RIGHT]}
    groups["all"] = list(data)
    out = {"tag": tag, "n": n, "groups": {}}
    print(f"gap to the oracle gate ({tag}, {n} clips)")
    print(f"  {'group':15s} {'n':>3s}  {'oracle':>6s}  " + "  ".join(f"{NAME[s]:>18s}" for s in SYSTEMS))
    for g, cs in groups.items():
        orc = [oracle_score(data[c]) for c in cs]
        row = {"n": len(cs), "oracle": float(np.mean(orc))}
        cells = []
        for s in SYSTEMS:
            sc = [data[c][s]["score"] for c in cs]
            gap = [o - x for o, x in zip(orc, sc)]
            lo, hi = boot_ci(gap)
            row[NAME[s]] = {"mean": float(np.mean(sc)), "gap": float(np.mean(gap)), "gap_ci": [lo, hi]}
            cells.append(f"{np.mean(sc):.2f} (-{np.mean(gap):.2f} [{lo:.2f},{hi:.2f}])")
        out["groups"][g] = row
        print(f"  {g:15s} {len(cs):3d}  {np.mean(orc):6.2f}  " + "  ".join(f"{c:>18s}" for c in cells))
    p = _ROOT / "benchmark" / f"oracle_gap_{tag}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v3_grounded")
