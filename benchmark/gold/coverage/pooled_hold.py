"""All 158 clips (Adam 9 Oct): picture end times from the ears' evidence, tuned by clip-wise cross-validation.
Base pictures = a/b gate + flash + plain texture ban (pooled best, pooled_dev_test.md). Starts are never moved.

Rule (picture level, online): from the picture's start, follow the family's evidence (FlexSED >= F, BEATs >= B, DASM
>= 0.35 if D) bridging gaps < 1 s; new end = last evidence + tau, at least start + 1.5 s, never into the next picture of
the same label (minus the 2.5-s merge gap) or past the clip. mode "extend": the end may only grow; "both": it may also
shrink. "extend" walks on from the current end (as Step 1), "both" re-derives the end from the start. The grid also
holds "no change". Objective per clip set: J = cost_cov + 0.5 x (wrong seconds + stale seconds) per clip (a wrong picture costs 2
and lasts ~4 s). Settings chosen on the training folds, scored on the held-out fold: 2 folds = DEV <-> TEST, and
5 random clip folds (seed 0). Reported: held-out J, hits / wrong, cost_cov, start / end errors, coverage.

    python benchmark/gold/coverage/pooled_hold.py   (cluster CPU from ~/wt_slice) -> pooled_hold.md
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.step11_policies import policy, TEXTURE
from benchmark.gold.coverage.hold_sweep import _at

HERE = Path(__file__).resolve().parent
GRID = [None] + list(itertools.product((0.3, 0.4, 0.5), (None, 0.1, 0.175), (False, True), (0.0, 0.5, 1.0), ("extend", "both")))
STEP, BRIDGE, DWELL, GAP = 0.02, 1.0, 1.5, 2.5


def ev_at(c, t, F, B, D):
    return (_at(c.get("flex"), t) >= F or (B is not None and _at(c.get("beats"), t) >= B)
            or (D and _at(c.get("dasm"), t) >= 0.35))


def hold(pics, ev, setting, dur):
    if setting is None:                                  # no change (the base)
        return list(pics)
    F, B, D, tau, mode = setting
    out = []
    for k, (lab, a, b) in enumerate(pics):
        c = ev["labels"].get(lab)
        if not c:
            out.append((lab, a, b)); continue
        nxt = [p[1] for p in pics if p[0] == lab and p[1] > a + 1e-9]
        cap = min([dur] + [n - GAP for n in nxt])
        t, last = (a if mode == "both" else b), None     # "extend": walk on from the current end (Step 1); "both": from the start
        while t < cap:
            if ev_at(c, t, F, B, D):
                last = t
            elif (last is None and t - a >= BRIDGE) or (last is not None and t - last >= BRIDGE):
                break
            t += STEP
        if last is None:
            out.append((lab, a, b)); continue
        e = min(cap, max(a + DWELL, last + tau))
        e = max(b, e) if mode == "extend" else e
        if last is None and mode == "extend":
            e = b
        out.append((lab, a, max(a + 0.1, min(e, max(cap, b)))))
    return out


def J(rows):
    A = V.aggregate(rows)
    return A["cost_cov"] + 0.5 * (A["wrong_s_per_clip"] + A["stale_s_per_clip"]), A


def main():
    gold = S.load_gold([V.GOLD])
    dev, test = V.stems("dev"), V.stems("test")
    allc = dev + test
    Dp = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    Tp = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    base = {}
    for st in allc:
        p = [tuple(x) for x in (Dp if st in Dp else Tp)[st]["pics_none"]]
        p = [x for x in policy(st, p, "F", {}, {}, fl) if x[0] not in TEXTURE]
        base[st] = p
    dur = {st: max([x[2] for x in base[st]] + [10.0]) for st in allc}
    for st in allc:
        for c in ev.get(st, {}).get("labels", {}).values():
            for m in (c or {}).values():
                if m and m.get("t"):
                    dur[st] = max(dur[st], float(m["t"][-1]))
    rows = {}
    for si, s in enumerate(GRID):
        for st in allc:
            rows[(si, st)] = V.score_clip_v2(gold[st], hold(base[st], ev.get(st, {"labels": {}}), s, dur[st]))
    base_rows = {st: V.score_clip_v2(gold[st], base[st]) for st in allc}
    rng = random.Random(0)
    sh = allc[:]; rng.shuffle(sh)
    folds = [("DEV->TEST", dev, test), ("TEST->DEV", test, dev)] + \
            [(f"random {k + 1}/5", [c for c in sh if c not in sh[k::5]], sh[k::5]) for k in range(5)]
    L = ["# All 158 clips: picture ends from the ears' evidence, clip-wise cross-validation", "",
         "Base = a/b gate + flash + plain texture ban. Starts never moved. J = cost_cov + 0.5 x (wrong + stale seconds) per clip.", "",
         "| fold | chosen setting (F, B, D, tau, mode) | held-out J: base -> hold | cost_cov | hits / wrong | |end err| median | hit cover |",
         "|---|---|---|---|---|---|---|"]
    held = []
    for name, tr, te in folds:
        best = min(range(len(GRID)), key=lambda si: J([rows[(si, st)] for st in tr])[0])
        jb, ab = J([base_rows[st] for st in te]); jh, ah = J([rows[(best, st)] for st in te])
        held += [(st, best) for st in te]
        L.append(f"| {name} | {GRID[best]} | {jb:.3f} -> {jh:.3f} | {ab['cost_cov']:.3f} -> {ah['cost_cov']:.3f} | "
                 f"{ab['hits']}/{ab['wrong']} -> {ah['hits']}/{ah['wrong']} | {ab['end_abs_med']:.2f} -> {ah['end_abs_med']:.2f} | "
                 f"{ab['hit_cov']:.2f} -> {ah['hit_cov']:.2f} |")
    # the 5 random folds together = one out-of-sample prediction per clip
    oos = [(st, si) for st, si in held[-len(allc):]]
    jb, ab = J([base_rows[st] for st, _ in oos]); jh, ah = J([rows[(si, st)] for st, si in oos])
    L += ["", f"5-fold out-of-sample, all 158 clips: J {jb:.3f} -> {jh:.3f}; cost_cov {ab['cost_cov']:.3f} -> {ah['cost_cov']:.3f}; "
          f"hits/wrong {ab['hits']}/{ab['wrong']} -> {ah['hits']}/{ah['wrong']}; start err median {ab['start_med']:+.2f} s (|{ab['start_abs_med']:.2f}|); "
          f"end err median {ab['end_med']:+.2f} -> {ah['end_med']:+.2f} s (|{ab['end_abs_med']:.2f}| -> |{ah['end_abs_med']:.2f}|); "
          f"hit cover {ab['hit_cov']:.2f} -> {ah['hit_cov']:.2f}; wrong s/clip {ab['wrong_s_per_clip']:.2f} -> {ah['wrong_s_per_clip']:.2f}; "
          f"stale s/clip {ab['stale_s_per_clip']:.2f} -> {ah['stale_s_per_clip']:.2f}"]
    best_all = min(range(len(GRID)), key=lambda si: J([rows[(si, st)] for st in allc])[0])
    L.append(f"\nSetting chosen on all 158 clips (for the shipped rule, not a score): {GRID[best_all]}")
    (HERE / "pooled_hold.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
