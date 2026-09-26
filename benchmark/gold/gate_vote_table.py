"""Week plan B.4: each of the gate's three questions voting alone vs the shipped majority, on the DEV gold sounds
(importance >= 2). Report only. A vote alone = the clip-level rule 'silent only if that vote says visible in every
stretch'. Columns as in gate_gold.score: seen sounds silenced / needed sounds kept / balanced accuracy.

    python benchmark/gold/gate_vote_table.py --arm Qwen38-27B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmark.gold.gate_gold import OUT_DIR, JUDGE100, decide


def alone(stretches, key):
    return all(st.get(key) is True for st in stretches) if stretches else False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="Qwen38-27B"); a = ap.parse_args()
    judge = set(JUDGE100.read_text().split())
    rules = {"name (open naming)": lambda s: alone(s, "name"), "a/b (both orders)": lambda s: alone(s, "ab"),
             "desc (description)": lambda s: alone(s, "desc"), "majority of 3 (shipped)": lambda s: decide(s, "majority")}
    rows = {}
    for rn, fn in rules.items():
        ss = sn = nk = nn = 0
        for f in (OUT_DIR / a.arm).glob("*.json"):
            if f.stem not in judge:
                continue
            for s in json.loads(f.read_text(encoding="utf-8"))["sounds"]:
                if s["importance"] < 2:
                    continue
                v = fn(s["stretches"])
                if s["seen"]:
                    sn += 1; ss += int(v)
                else:
                    nn += 1; nk += int(not v)
        rows[rn] = {"seen": sn, "needed": nn, "seen_silenced": ss / sn, "needed_kept": nk / nn, "balanced": (ss / sn + nk / nn) / 2}
        r = rows[rn]
        print(f"{rn:26s} seen silenced {r['seen_silenced']:.2f} ({ss}/{sn}) | needed kept {r['needed_kept']:.2f} ({nk}/{nn}) | balanced {r['balanced']:.2f}")
    (OUT_DIR / f"vote_table_{a.arm}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
