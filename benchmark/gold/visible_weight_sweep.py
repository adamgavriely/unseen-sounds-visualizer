"""Declared secondary (docs/prereg_round13_detector_push.md "visible-weighted cost"): cost = (4 miss + w visible + 2 cross
+ 2 phantom) / clips for w in {0, 1, 2}, for the shipped chain on merged DEV (benchmark/gold/merged_dev.json rows) and TEST
(benchmark/gold/final_test*.json rows).

    python benchmark/gold/visible_weight_sweep.py
"""
import glob
import json
from pathlib import Path

G = Path(__file__).resolve().parent
W = (0, 1, 2)


def cost(r, n, w):
    return (4 * r["misses"] + w * r["visible"] + 2 * r["cross"] + 2 * r["phantom"]) / n


def table(rows, n, title):
    out = [f"## {title} ({n} clips)", "", "| system | hits | wrong (v/c/p) | " + " | ".join(f"cost w={w}" for w in W) + " |",
           "|---|---|---|" + "---|" * len(W)]
    for name, r in rows:
        out.append(f"| {name} | {r['hits']} | {r['visible'] + r['cross'] + r['phantom']} ({r['visible']}/{r['cross']}/{r['phantom']}) | "
                   + " | ".join(f"{cost(r, n, w):.3f}" for w in W) + " |")
    return out


def main():
    lines = ["# Visible-weighted cost sweep (secondary; w = weight of a picture whose source is on screen; w = 2 = primary)", ""]
    md = G / "merged_dev.json"
    if md.exists():
        d = json.loads(md.read_text(encoding="utf-8"))
        n = sum(d["parts"].values())
        lines += table(list(d["rows"].items()), n, "merged DEV") + [""]
    rows, seen = [], set()
    for f in sorted(glob.glob(str(G / "final_test*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for name, r in d["rows"].items():
            if name not in seen:
                seen.add(name); rows.append((name, r)); n = sum(d["clips"].values())
    lines += table(rows, n, "TEST (reported reads)")
    (G / "visible_weight_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
