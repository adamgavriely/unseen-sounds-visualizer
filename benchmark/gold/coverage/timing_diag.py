"""Fable's ideas 1 and 3 (10 Oct): (1) for needed sounds that a candidate overlaps but none starts in the onset window,
the start offset of the nearest overlapping candidate, by detector origin; (3) the current system with every picture
start shifted earlier by d (a fixed display delay), all 158 clips.   python benchmark/gold/coverage/timing_diag.py"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.step11_policies import policy, TEXTURE
from benchmark.gold.coverage.pooled_hold import hold, J

HERE = Path(__file__).resolve().parent


def main():
    gold = S.load_gold([V.GOLD]); dev, test = V.stems("dev"), V.stems("test"); allc = dev + test
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    cands = {c["clip"]: c["cands"] for c in dj["clips"]}
    L = ["# Timing diagnostics, all 158 clips", "", "## (1) heard-but-mistimed needed sounds (importance 2-3)", ""]
    offs = collections.defaultdict(list)
    n = 0
    for st in allc:
        for g in gold[st]:
            if not (g["needed"] and g["importance"] >= 2):
                continue
            cs = [c for c in cands[st] if S.same_family(c["label"], g["label"])]
            if any(S.in_window(c["start"], g["start"], S.EARLY, S.LATE) for c in cs):
                continue
            ov = [c for c in cs if c["end"] > g["start"] - 0.5 and c["start"] < g["end"] + 0.5]
            if not ov:
                continue
            n += 1
            c = min(ov, key=lambda c: abs(c["start"] - g["start"]))
            offs[c["origin"]].append(round(c["start"] - g["start"], 2))
            offs["all"].append(round(c["start"] - g["start"], 2))
    bins = [-99, -5, -2, -1, -0.5, 1, 1.5, 2.5, 5, 99]
    L += [f"{n} sounds. Offset = nearest overlapping candidate start - gold onset (s).", "",
          "| origin | n | " + " | ".join(f"{bins[i]}..{bins[i + 1]}" for i in range(len(bins) - 1)) + " |",
          "|---|---|" + "---|" * (len(bins) - 1)]
    for o, v in sorted(offs.items(), key=lambda kv: -len(kv[1])):
        h = np.histogram(v, bins=bins)[0]
        L.append(f"| {o} | {len(v)} | " + " | ".join(str(x) for x in h) + " |")
    # (3) fixed shift of the current system
    Dc = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    Tc = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    dur = {}
    for st in allc:
        ts = [float(m["t"][-1]) for c in ev.get(st, {}).get("labels", {}).values() for m in (c or {}).values() if m and m.get("t")]
        dur[st] = max(ts) if ts else 10.0
    L += ["", "## (3) current system with every start shifted earlier by d", "", "| d (s) | hits | wrong | onset cost | cost_cov | start err median |", "|---|---|---|---|---|---|"]
    for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        rr = []
        for st in allc:
            p = [x for x in policy(st, [tuple(q) for q in (Dc if st in dev else Tc)[st]["pics_none"]], "F", {}, {}, fl) if x[0] not in TEXTURE]
            p = hold(p, ev.get(st, {"labels": {}}), (0.5, None, False, 0.0, "extend"), dur[st])
            p = [(lab, max(0.0, a - d), b) for lab, a, b in p]
            rr.append(V.score_clip_v2(gold[st], p))
        A = V.aggregate(rr)
        L.append(f"| {d} | {A['hits']} | {A['wrong']} | {A['onset_cost']:.3f} | {A['cost_cov']:.3f} | {A['start_med']:+.2f} |")
    (HERE / "timing_diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
