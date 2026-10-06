"""Step 3c (PREREG_step3c_start_move.md): move each picture's start back to the earliest unbroken same-family evidence,
at most W s. DEV only, CPU, local.   python benchmark/gold/coverage/start_move.py  -> start_move_dev.json / .md"""
import itertools
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.hold_sweep import _at

HERE = Path(__file__).resolve().parent
STEP = 0.02
E = {"F0.5": lambda c, t: _at(c.get("flex"), t) >= 0.5,
     "F0.3": lambda c, t: _at(c.get("flex"), t) >= 0.3,
     "FB": lambda c, t: _at(c.get("flex"), t) >= 0.5 or _at(c.get("beats"), t) >= 0.175,
     "ANY": lambda c, t: _at(c.get("flex"), t) >= 0.3 or _at(c.get("beats"), t) >= 0.10 or _at(c.get("dasm"), t) >= 0.35}
W = (0.5, 1.0, 1.5)
BASES = {"D'": "SHIP8+MD3+WW5+SL|M", "AB-m": "SHIP8+MD3+WW5+SL|AB-m"}


def move(pics, ev, rule, w):
    out = []
    for k, (lab, a, b) in enumerate(pics):
        c = ev["labels"].get(lab, {})
        prev = max([p[2] for p in pics[:k] if p[0] == lab and p[1] < a] + [0.0])
        t, new = a - STEP, a
        while t >= max(prev, a - w) - 1e-9 and t >= 0 and E[rule](c, t):
            new = t; t -= STEP
        out.append((lab, round(new, 3), b))
    return out


def main():
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    ev = json.loads((HERE / "evidence_dev_all.json").read_text(encoding="utf-8"))
    d = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]
    miss = sorted({(st, p[0]) for c in BASES.values() for st in dev for p in d[c]["clips"][st]["pics_none"] if p[0] not in ev[st]["labels"]})
    res, L = {"no_evidence_labels": miss}, []
    for bn, cell in BASES.items():
        base = {st: [tuple(p) for p in d[cell]["clips"][st]["pics_none"]] for st in dev}
        rb = {st: V.score_clip_v2(gold[st], base[st]) for st in dev}
        ab = V.aggregate(list(rb.values()))
        res[bn] = {"base": ab, "cells": {}}
        L += [f"## base {bn}", "", "| cell | hits | wrong | onset cost | cost_cov | hit cover | changed clips | pass |", "|---|---|---|---|---|---|---|---|",
              f"| base | {ab['hits']} | {ab['wrong']} | {ab['onset_cost']:.3f} | {ab['cost_cov']:.3f} | {ab['hit_cov']:.2f} | - | - |"]
        for rule, w in itertools.product(E, W):
            rr = {st: V.score_clip_v2(gold[st], move(base[st], ev[st], rule, w)) for st in dev}
            a = V.aggregate(list(rr.values()))
            wr = lambda r: r["visible"] + r["cross"] + r["phantom"]
            ch = [[st, rb[st]["hit"], rr[st]["hit"], wr(rb[st]), wr(rr[st])] for st in dev if (rr[st]["hit"], wr(rr[st])) != (rb[st]["hit"], wr(rb[st]))]
            a["changed"] = ch
            a["pass"] = a["hits"] >= ab["hits"] and a["wrong"] <= ab["wrong"] and a["onset_cost"] < ab["onset_cost"] - 1e-9
            res[bn]["cells"][f"{rule}_W{w}"] = a
            L.append(f"| {rule} W{w} | {a['hits']} | {a['wrong']} | {a['onset_cost']:.3f} | {a['cost_cov']:.3f} | {a['hit_cov']:.2f} | "
                     + ("; ".join(f"{st} {h0}->{h1}/{w0}->{w1}" for st, h0, h1, w0, w1 in ch) or "-") + f" | {'yes' if a['pass'] else 'no'} |")
        ok = [k for k, a in res[bn]["cells"].items() if a["pass"]]
        sel = min(ok, key=lambda k: (round(res[bn]["cells"][k]["onset_cost"], 2), res[bn]["cells"][k]["cost_cov"])) if ok else None
        res[bn]["selected"] = sel
        L += ["", f"passing: {ok or 'none'}; selected: {sel}", ""]
    L.append(f"pictures with no evidence curve (never moved): {miss}")
    (HERE / "start_move_dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (HERE / "start_move_dev.md").write_text("# Step 3c: start at the earliest heard evidence, DEV (71 clips)\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
