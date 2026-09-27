"""Build data.js for the inspector page from data.json.

The page (index.html) must open by double-click (file://), where the browser blocks fetch(); so the data is loaded as a
script: data.js = "window.INSPECTOR = {...}". This script copies data.json unchanged and adds a "derived" block per
clip with the facts that need the AudioSet family rule (benchmark/gold/score_per_sound.same_family), which the page
cannot compute in JavaScript:

  - why each missed needed sound (importance 2-3) was missed, per system
    (reasons in the order of benchmark/gold/dev_miss_table.py, plus one bookkeeping case)
  - which gate stretch each shown picture came from (and the gate votes there)
  - which augmentation (index) each of ours' pictures belongs to (for the picture thumbnail)
  - gate errors: a needed sound where the VLM voted "visible" at the onset

Decides nothing and changes no score. Re-run after data.json changes:

    python docs/inspector/build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / "data.json"
OUT = HERE / "data.js"
EARLY, LATE = 0.5, 1.0
NOTE = ""

try:
    sys.path.insert(0, str(ROOT))
    from benchmark.gold import score_per_sound as S
    same_family = S.same_family
    EARLY, LATE = S.EARLY, S.LATE
except Exception as e:  # the page still works, but family matches are exact-name only
    NOTE = f"WARNING: could not import the scorer ({e!r}); family match = exact label only. Reasons may be wrong."
    print(NOTE)

    def same_family(a, b):
        return a == b


def in_window(pic_start, onset):
    return onset - EARLY <= pic_start <= onset + LATE


def gate_of_picture(pic, gate):
    """index of the gate entry (stretch) a picture was drawn from, and whether the match is approximate"""
    for k, g in enumerate(gate):
        if g["label"] == pic["label"] and abs(g["stretch"][0] - pic["start"]) < 0.05:
            return k, False
    # display spans can merge or relabel stretches: fall back to a same-family stretch holding the start
    best = None
    for k, g in enumerate(gate):
        a, b = g["stretch"]
        if same_family(g["label"], pic["label"]) and a - 0.05 <= pic["start"] <= b + 0.05:
            d = abs(a - pic["start"])
            if best is None or d < best[0]:
                best = (d, k)
    return (best[1], True) if best else (None, True)


def aug_of_gate(g, augs):
    """augmentation index of a gate entry: same event (label + start). The thumbnail file is
    media/<SPLIT>/<clip>_pics/<index>.png with this index (position in augmentations.json)."""
    for i, a in enumerate(augs):
        if a["event_label"] == g["label"] and abs(float(a["start"]) - float(g["start"])) < 0.01:
            return i
    return None


def aug_of_picture(pic, gate, augs, gk):
    if gk is not None:
        i = aug_of_gate(gate[gk], augs)
        if i is not None:
            return i, False
    # fallback: a shown augmentation of the same family, nearest start
    cands = [(abs(float(a["start"]) - pic["start"]), i) for i, a in enumerate(augs)
             if a.get("augment") and same_family(a["event_label"], pic["label"])]
    return (min(cands)[1], True) if cands else (None, True)


def miss_reason(snd, own_pics, blind_pics, events, is_ours):
    """why a needed sound (scorer outcome: miss) got no picture; same order as dev_miss_table.py"""
    lab, on, end = snd["label"], snd["start"], snd["end"]
    fam = [p for p in own_pics if same_family(p["label"], lab)]
    if any(in_window(p["start"], on) for p in fam):
        p = min((p for p in fam if in_window(p["start"], on)), key=lambda p: abs(p["start"] - on))
        return {"reason": "taken by another sound", "pic_start": p["start"], "pic_label": p["label"]}
    bfam = [p for p in blind_pics if same_family(p["label"], lab)]
    if is_ours and any(in_window(p["start"], on) for p in bfam):
        p = min((p for p in bfam if in_window(p["start"], on)), key=lambda p: abs(p["start"] - on))
        return {"reason": "removed by the gate", "pic_start": p["start"], "pic_label": p["label"]}
    over = [(p, s) for s, ps in (("ours" if is_ours else "blind", fam), ("blind", bfam if is_ours else [])) for p in ps
            if p["end"] > on and p["start"] < end]
    if over:
        p, s = min(over, key=lambda x: abs(x[0]["start"] - on))
        return {"reason": "timing", "pic_start": p["start"], "pic_label": p["label"], "pic_system": s,
                "late": round(p["start"] - on, 2)}
    ev = [e for e in events if same_family(e["label"], lab) and e["end"] >= on - EARLY and e["start"] <= on + LATE]
    if ev:
        e = max(ev, key=lambda e: e.get("confidence") or 0)
        return {"reason": "detected, not drawn", "event_label": e["label"], "event_start": e["start"],
                "confidence": e.get("confidence")}
    return {"reason": "never detected"}


def near_gate(snd, gate):
    """gate entries (stretches) of the sound's family that overlap its onset window or its span"""
    lab, on, end = snd["label"], snd["start"], max(snd["end"], snd["start"] + LATE)
    return [k for k, g in enumerate(gate)
            if same_family(g["label"], lab) and g["stretch"][1] >= on - EARLY and g["stretch"][0] <= end]


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    counts = {}
    for c in d["clips"]:
        ours = c["systems"].get("ours") or {}
        blind = c["systems"].get("blind") or {}
        gate = ours.get("gate") or []
        augs = ours.get("augmentations") or []
        events = ours.get("events") or []
        der = {"ours_pics": [], "blind_pics": [], "miss": {"ours": {}, "blind": {}}, "near_gate": {}}
        for p in ours.get("pictures") or []:
            gk, ga = gate_of_picture(p, gate)
            ai, aa = aug_of_picture(p, gate, augs, gk)
            der["ours_pics"].append({"gate": gk, "gate_approx": ga, "aug": ai, "aug_approx": aa})
        for p in blind.get("pictures") or []:
            gk, ga = gate_of_picture(p, gate)
            # was the same stretch also drawn by ours? (if not, the gate removed it)
            in_ours = any(q["label"] == p["label"] and abs(q["start"] - p["start"]) < 0.05 for q in ours.get("pictures") or [])
            der["blind_pics"].append({"gate": gk, "gate_approx": ga, "in_ours": in_ours})
        for i, s in enumerate(c["sounds"]):
            ng = near_gate(s, gate)
            if ng:
                der["near_gate"][str(i)] = ng
            for name, sysd, is_ours in (("ours", ours, True), ("blind", blind, False)):
                out = (sysd.get("sounds") or [None] * len(c["sounds"]))[i]
                if out and out.get("outcome") == "miss":
                    r = miss_reason(s, sysd.get("pictures") or [], blind.get("pictures") or [], events, is_ours)
                    if r["reason"] == "removed by the gate":
                        r["gate"] = ng
                    der["miss"][name][str(i)] = r
                    key = (c["split"], name, r["reason"])
                    counts[key] = counts.get(key, 0) + 1
        c["derived"] = der
    d["build_note"] = NOTE
    d["window"] = [-EARLY, LATE]
    OUT.write_text("window.INSPECTOR = " + json.dumps(d, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("miss reasons (split, system, reason): count")
    for k in sorted(counts):
        print("  ", k, counts[k])
    print("->", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
