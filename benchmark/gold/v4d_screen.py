"""Round 29 V4D screen (docs/prereg_round13_detector_push.md): on merged-DEV P2/PV candidates, how many needed-class
accepts does V4D add beyond Qwen V4, and how many other-class accepts. GO iff added needed >= 2 and added other <= 2x.

    python benchmark/gold/v4d_screen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_listener as L

G = _ROOT / "benchmark" / "gold"
PARTS = {"dev": (G / "dev_listener_v.json", G / "dev_listener_v4d.json", G / "annotations" / "gold_AG.json"),
         "dev2": (G / "dev2_listener_v.json", G / "dev2_listener_v4d.json", G / "annotations" / "gold_AG.json")}


def key(x):
    return (x["clip"], x["pool"], x["label"], round(x["start"], 2), round(x["end"], 2))


def main():
    tot = {"needed_added": 0, "other_added": 0, "needed_lost": 0}
    for part, (vp, dp, gp) in PARTS.items():
        v = {key(x): x for x in json.loads(vp.read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")}
        d = {key(x): x for x in json.loads(dp.read_text(encoding="utf-8"))["items"]}
        gold = S.load_gold([gp])
        add_n, add_o, nd = [], [], 0
        for k, x in d.items():
            if k not in v or x["clip"] not in gold:
                continue
            q4 = bool(v[k]["accept"].get("V4"))
            dd = bool(x["accept"].get("V4D"))
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            nd += 1
            if dd and not q4:
                (add_n if cls == "hit_needed" else add_o).append(f"{x['clip']} {x['label']} {x['start']:.1f} [{cls}]")
        print(f"[{part}] candidates {nd}: V4D adds needed {len(add_n)}, other {len(add_o)}")
        for s in add_n + add_o:
            print("    ", s)
        tot["needed_added"] += len(add_n); tot["other_added"] += len(add_o)
    go = tot["needed_added"] >= 2 and tot["other_added"] <= 2 * tot["needed_added"]
    print("V4D screen:", tot, "GO" if go else "STOP")
    (G / "v4d_screen.json").write_text(json.dumps({**tot, "go": go}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
