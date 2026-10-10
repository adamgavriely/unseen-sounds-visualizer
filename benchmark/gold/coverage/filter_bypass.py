"""Which hand filter drops too many right candidates? (Fable idea "per-filter attribution", Adam 10 Oct: keep trying.)
All 158 clips. For every rule step that drops candidates (decision trail `at`), the dropped candidates (Speech / Music out)
are scored as if that step had let them through: added as pictures (family, start, end) when no picture of their family
overlaps, same-family adds within 2.5 s joined, then the adopted hold. Attribution table on all clips, then a
clip-grouped 5-fold CV (seed 0): on the training clips, greedy forward selection of (step, peak bar q in {0, 0.5, 0.7})
pairs by onset cost; scored on the held-out clips.

    python benchmark/gold/coverage/filter_bypass.py   (cluster CPU from ~/wt_slice) -> filter_bypass.md
"""
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.pooled_hold import hold, J
from benchmark.gold.coverage.step11_policies import policy, TEXTURE
from benchmark.gold.coverage.fable_ideas import peak_of
from src.labels import canonical, is_music

HERE = Path(__file__).resolve().parent
HOLD = (0.5, None, False, 0.0, "extend")
QS = (0.0, 0.5, 0.7)


def main():
    gold = S.load_gold([V.GOLD]); dev, test = V.stems("dev"), V.stems("test"); allc = dev + test
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    cands = {c["clip"]: [x for x in c["cands"] if x.get("fate") != "drawn" and canonical(x["label"]) not in ("Speech", "Music")
                         and not is_music(x["label"]) and canonical(x["label"]) not in TEXTURE] for c in dj["clips"]}
    Dc = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    Tc = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    dur = {}
    for st in allc:
        ts = [float(m["t"][-1]) for c in ev.get(st, {}).get("labels", {}).values() for m in (c or {}).values() if m and m.get("t")]
        dur[st] = max(ts) if ts else 10.0
    base = {st: [x for x in policy(st, [tuple(p) for p in (Dc if st in dev else Tc)[st]["pics_none"]], "F", {}, {}, fl) if x[0] not in TEXTURE]
            for st in allc}
    steps = sorted({x.get("at") or "?" for st in allc for x in cands[st]})

    def build(st, sel):
        pics = list(base[st])
        add = [x for x in cands[st] if any((x.get("at") or "?") == f and peak_of(x) >= q for f, q in sel)]
        add = sorted(add, key=lambda x: (canonical(x["label"]), x["start"]))
        joined, cur = [], None
        for x in add:
            fam = canonical(x["label"])
            if any(S.same_family(p[0], fam) and p[2] > x["start"] - 0.5 and p[1] < x["end"] + 0.5 for p in pics):
                continue
            if cur and cur[0] == fam and x["start"] - cur[2] <= 2.5:
                cur[2] = max(cur[2], x["end"])
            else:
                cur = [fam, x["start"], x["end"]]; joined.append(cur)
        pics = sorted(pics + [tuple(c) for c in joined], key=lambda p: p[1])
        return V.score_clip_v2(gold[st], hold(pics, ev.get(st, {"labels": {}}), HOLD, dur[st]))

    memo = {}

    def rows(sel, clips):
        k = tuple(sorted(sel))
        out = []
        for st in clips:
            if (k, st) not in memo:
                memo[(k, st)] = build(st, sel)
            out.append(memo[(k, st)])
        return out

    ref = rows([], allc); A0 = V.aggregate(ref)
    L = ["# Which filter drops too many right candidates? (all 158 clips)", "",
         f"Current system: {A0['hits']} hits / {A0['wrong']} wrong, onset cost {A0['onset_cost']:.3f}.", "",
         "## Attribution: let one step's dropped candidates through (on top of the current system)", "",
         "| step | dropped cands | peak >= q | hits / wrong if bypassed | d hits | d wrong | onset cost |", "|---|---|---|---|---|---|---|"]
    single = []
    for f in steps:
        n = sum(1 for st in allc for x in cands[st] if (x.get("at") or "?") == f)
        for q in QS:
            a = V.aggregate(rows([(f, q)], allc))
            dh, dw = a["hits"] - A0["hits"], a["wrong"] - A0["wrong"]
            if dh or dw:
                single.append((a["onset_cost"], f, q))
                L.append(f"| {f} | {n} | {q} | {a['hits']} / {a['wrong']} | {dh:+d} | {dw:+d} | {a['onset_cost']:.3f} |")
    pool = [(f, q) for _, f, q in sorted(single)]
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh)
    oof, chosen = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        sel, cur = [], V.aggregate(rows([], tr))["onset_cost"]
        while True:
            best = None
            for pq in pool:
                if any(pq[0] == f for f, _ in sel):
                    continue
                c = V.aggregate(rows(sel + [pq], tr))["onset_cost"]
                if best is None or c < best[0]:
                    best = (c, pq)
            if best is None or cur - best[0] < 0.005:
                break
            sel.append(best[1]); cur = best[0]
        chosen.append(sel)
        for st, r in zip(sorted(te), rows(sel, sorted(te))):
            oof[st] = r
    a = V.aggregate([oof[st] for st in allc]); j = J([oof[st] for st in allc])[0]
    L += ["", "## Clip-grouped 5-fold CV (selection on training clips only)", ""] + [f"- fold {i + 1}: {c}" for i, c in enumerate(chosen)] + \
         ["", f"Out of fold, all 158 clips: {a['hits']} hits / {a['wrong']} wrong, onset cost {a['onset_cost']:.3f} (current {A0['onset_cost']:.3f}), J {j:.3f}"]
    (HERE / "filter_bypass.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
