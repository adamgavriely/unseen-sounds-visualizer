"""Step 1 sweep (PREREG_step1_hold.md): hold each picture burst while its family is still heard. DEV only.

    python benchmark/gold/coverage/hold_sweep.py build   # cluster (checkout of the scored runs), CPU: pictures per cell
    python benchmark/gold/coverage/hold_sweep.py score   # local: scorer v2 per cell, pass bars, selection

build reads evidence_dev.json (dump_evidence.py) and the frozen arm's augmentations.json, extends burst ends (starts
untouched), then draws the pictures with the real display code under the arm's display flags (MAX_AFTER_END None).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

HERE = Path(__file__).resolve().parent
ARM = "SHIP8+MD3+WW5+SL"
STEP, BRIDGE, TAIL, DASM_BAR = 0.02, 1.0, 0.5, 0.35
GRID = {"F": (0.3, 0.4, 0.5), "B": (None, 0.10, 0.175), "D": (False, True), "G": ("strict", "lenient")}


def cells():
    out = {}
    for f, b, d, g in itertools.product(GRID["F"], GRID["B"], GRID["D"], GRID["G"]):
        out[f"F{f}_B{'off' if b is None else b}_D{'on' if d else 'off'}_{g}"] = {"F": f, "B": b, "D": d, "G": g}
    return out


# ----------------------------------------------------------------------------- the rule
def _at(curve, t):
    """an ear's family score at time t (nearest frame); 0 outside the curve"""
    if curve is None or not curve["t"]:
        return 0.0
    ts = curve["_t"] if "_t" in curve else np.asarray(curve["t"])
    i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
    if i > 0 and abs(ts[i - 1] - t) <= abs(ts[i] - t):
        i -= 1
    return float(curve["v"][i])


def evidence(curves, t, cell):
    if _at(curves.get("flex"), t) >= cell["F"]:
        return True
    if cell["B"] is not None and _at(curves.get("beats"), t) >= cell["B"]:
        return True
    return bool(cell["D"]) and _at(curves.get("dasm"), t) >= DASM_BAR


def gate_ok(gate, label, t, mode):
    """strict: t inside a stretch judged not seen for the family; lenient: t not inside a stretch judged seen"""
    st = [s for g in gate if S.same_family(g["label"], label) or g["label"] == label for s in g["stretches"]]
    if mode == "strict":
        return any(a <= t <= b and not seen for a, b, seen in st)
    return not any(a <= t <= b and seen for a, b, seen in st)


def hold_end(b, cap, curves, gate, label, cell):
    """new end of a burst that now ends at b; never shorter, never past cap"""
    t, last = b, None
    while t < cap:
        if not gate_ok(gate, label, t, cell["G"]):
            break
        if evidence(curves, t, cell):
            last = t
        elif (last is None and t - b >= BRIDGE) or (last is not None and t - last >= BRIDGE):
            break
        t += STEP
    if last is None:
        return b
    e = min(last + TAIL, cap)
    u = last                                   # the tail itself must stay where the gate allows it
    while u < e and gate_ok(gate, label, u, cell["G"]):
        u += STEP
    return max(b, min(e, u))


def extend_specs(specs, ev, cell, frozen_pics, dur, merge_gap):
    """copy of the specs with burst ends held; frozen_pics = the frozen display's (label, start, end) for the no-new-join cap"""
    specs = copy.deepcopy(specs)
    gate = ev["gate"]
    shown = {}
    for lab, a, _ in frozen_pics:
        shown.setdefault(lab, []).append(a)
    for s in specs:
        for c in ev["labels"].get(s["event_label"], {}).values():
            if c is not None and "_t" not in c:
                c["_t"] = np.asarray(c["t"])
    bursts = {}                                 # label -> sorted burst starts over every drawn spec of that label
    for s in specs:
        if s.get("augment"):
            for a, _ in (s.get("spans") or [(s["start"], s["end"])]):
                bursts.setdefault(s["event_label"], []).append(float(a))
    for v in bursts.values():
        v.sort()
    for s in specs:
        if not s.get("augment"):
            continue
        lab = s["event_label"]
        curves = ev["labels"].get(lab, {})
        sp = [list(x) for x in (s.get("spans") or [(s["start"], s["end"])])]
        for x in sp:
            a, b = float(x[0]), float(x[1])
            nxt = [n for n in bursts[lab] if n > a + 1e-9]
            cap = float(dur)
            if nxt:
                n = nxt[0]
                separate = any(abs(max(0.0, n) - p) < 1e-6 for p in shown.get(lab, []))
                cap = min(cap, (n - merge_gap - STEP) if separate else n)
            if cap <= b:
                continue
            x[1] = hold_end(b, cap, curves, gate, lab, cell)
        s["spans"] = sp
        s["end"] = max([s["end"]] + [x[1] for x in sp])
    return specs


# ----------------------------------------------------------------------------- build (cluster)
def pictures(specs, dur, stem):
    """S.load_pictures on in-memory specs (same AugmentationSpec construction and display calls)"""
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])]) for s in specs]
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    placed, _ = _assign_rows(_display_spans(objs, d, require_image=True, clip=stem))
    return [[lab, float(a), float(b)] for _, lab, a, b, _ in placed]


def build():
    from benchmark.gold import inspector_trail_export as X
    try:
        from benchmark.gold import dev_harness as R
    except ImportError:
        from benchmark.gold import round13_dev as R
    import config
    ev_all = json.loads((_ROOT / "scratch_cov" / "evidence_dev.json").read_text(encoding="utf-8"))
    disp = {k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}
    out = {"arm": ARM, "cells": {c: {"clips": {}} for c in ["frozen"] + list(cells())}}
    with R.flags({**disp, "MAX_AFTER_END": None}):
        gap = max(float(config.MERGE_GAP), float(getattr(config, "MIN_DWELL", 1.5)))
        for split, part, base, stems in X.parts(ARM):
            if split != "DEV":
                continue
            root = base / f"{ARM}_proposed"
            for st in stems:
                specs = json.loads((root / st / "augmentations.json").read_text(encoding="utf-8"))
                dur = float(json.loads((root / st / "media.json").read_text(encoding="utf-8")).get("duration") or 0) or None
                ref = S.load_pictures(root, st, "proposed") or []
                froz = pictures(specs, dur, st)
                assert [tuple(p) for p in froz] == [tuple(p) for p in ref], (st, froz, ref)
                out["cells"]["frozen"]["clips"][st] = {"pics_none": froz}
                for name, cell in cells().items():
                    ext = extend_specs(specs, ev_all[st], cell, froz, dur or 1e9, gap)
                    out["cells"][name]["clips"][st] = {"pics_none": pictures(ext, dur, st)}
            print(split, part, len(stems), flush=True)
    p = _ROOT / "scratch_cov" / "hold_pics_dev.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    print("->", p)


# ----------------------------------------------------------------------------- score (local)
def score():
    from benchmark.gold.coverage import score_coverage as V
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    d = json.loads((HERE / "hold_pics_dev.json").read_text(encoding="utf-8"))["cells"]
    froz = d["frozen"]["clips"]
    base = V.aggregate([V.score_clip_v2(gold[st], [tuple(x) for x in froz[st]["pics_none"]]) for st in dev])
    bars = {"hit_cov": 0.90, "wrong_s": base["wrong_s_per_clip"] * 1.15, "stale_s": base["stale_s_per_clip"] + 0.5}
    res = {"baseline": base, "bars": bars, "cells": {}}
    for name, c in d.items():
        rows = [V.score_clip_v2(gold[st], [tuple(x) for x in c["clips"][st]["pics_none"]]) for st in dev]
        a = V.aggregate(rows)
        starts_same = sum(1 for st in dev if sorted((p[0], round(p[1], 6)) for p in c["clips"][st]["pics_none"]) ==
                          sorted((p[0], round(p[1], 6)) for p in froz[st]["pics_none"]))
        a["clips_same_starts"] = starts_same
        a["pass"] = {"hit_cov": a["hit_cov"] >= bars["hit_cov"],
                     "counts": a["hits"] == base["hits"] and a["wrong"] == base["wrong"] and starts_same == len(dev),
                     "wrong_s": a["wrong_s_per_clip"] <= bars["wrong_s"] + 1e-9,
                     "stale_s": a["stale_s_per_clip"] <= bars["stale_s"] + 1e-9}
        res["cells"][name] = a
    ok = [n for n, a in res["cells"].items() if n != "frozen" and all(a["pass"].values())]
    safe = [n for n, a in res["cells"].items() if n != "frozen" and a["pass"]["counts"] and a["pass"]["wrong_s"] and a["pass"]["stale_s"]]
    key = lambda n: (round(res["cells"][n]["cost_cov"], 3), res["cells"][n]["wrong_s_per_clip"] + res["cells"][n]["stale_s_per_clip"])
    pick = min(ok, key=key) if ok else None
    res["passing_all"] = ok
    res["selected"] = pick
    res["best_safe"] = min(safe, key=key) if safe else None
    (HERE / "hold_sweep_dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = ["| cell | hits | wrong | cost_cov | hit cover | need-time cover | wrong s/clip | stale s/clip | same starts | bars 1-4 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ["frozen"] + sorted((n for n in res["cells"] if n != "frozen"), key=key):
        a = res["cells"][n]
        p = "".join("y" if a["pass"][k] else "-" for k in ("hit_cov", "counts", "wrong_s", "stale_s"))
        lines.append(f"| {n} | {a['hits']} | {a['wrong']} | {a['cost_cov']:.3f} | {a['hit_cov']:.3f} | {a['needed_time_cov']:.3f} | "
                     f"{a['wrong_s_per_clip']:.2f} | {a['stale_s_per_clip']:.2f} | {a['clips_same_starts']}/{len(dev)} | {p} |")
    lines += ["", f"bars: hit cover >= 0.90; counts and starts unchanged; wrong <= {bars['wrong_s']:.2f} s/clip; stale <= {bars['stale_s']:.2f} s/clip",
              f"passing all four: {ok or 'none'}", f"selected: {pick}", f"best passing bars 2-4: {res['best_safe']}"]
    (HERE / "hold_sweep_dev.md").write_text("# Step 1 hold sweep, DEV (71 clips)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "score"))
    {"build": build, "score": score}[ap.parse_args().step]()
