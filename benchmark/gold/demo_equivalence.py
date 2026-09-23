"""Does the demo show the system that was measured? (2026-09-23)

ComfyUI needs a newer torch than the pipeline was validated on, so the demo runs under a second,
isolated torch build. That is only acceptable if it makes the SAME decisions. Comparing rendered
videos would prove nothing -- two diffusion passes are never identical -- so this compares what the
thesis actually measures: which sounds were detected, with what spans, and which of them the gate
chose to draw.

    python benchmark/gold/demo_equivalence.py --a data/work/torch_251/<stem> --b data/work/torch_280/<stem>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p, name):
    f = Path(p) / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="work dir of the validated build")
    ap.add_argument("--b", required=True, help="work dir of the demo build")
    a = ap.parse_args()
    same = True

    ea, eb = load(a.a, "events.json"), load(a.b, "events.json")
    if ea is None or eb is None:
        print("events.json missing on one side"); raise SystemExit(2)
    ka = sorted((e["label"], round(e["start"], 2), round(e["end"], 2)) for e in ea)
    kb = sorted((e["label"], round(e["start"], 2), round(e["end"], 2)) for e in eb)
    print(f"events: {len(ka)} vs {len(kb)}  {'IDENTICAL' if ka == kb else 'DIFFERENT'}")
    if ka != kb:
        same = False
        for x in sorted(set(ka) ^ set(kb))[:10]:
            print("   only on one side:", x)

    ga, gb = load(a.a, "augmentations.json"), load(a.b, "augmentations.json")
    if ga is not None and gb is not None:
        da = sorted((s["event_label"], round(s["start"], 2), bool(s.get("augment"))) for s in ga)
        db = sorted((s["event_label"], round(s["start"], 2), bool(s.get("augment"))) for s in gb)
        print(f"gate decisions: {len(da)} vs {len(db)}  {'IDENTICAL' if da == db else 'DIFFERENT'}")
        if da != db:
            same = False
            for x in sorted(set(da) ^ set(db))[:10]:
                print("   only on one side:", x)

    va, vb = load(a.a, "gate_votes.json"), load(a.b, "gate_votes.json")
    if va is not None and vb is not None:
        wa = sorted((v["label"], round(v["stretch"][0], 2), bool(v.get("seen"))) for v in va)
        wb = sorted((v["label"], round(v["stretch"][0], 2), bool(v.get("seen"))) for v in vb)
        print(f"visibility votes: {len(wa)} vs {len(wb)}  {'IDENTICAL' if wa == wb else 'DIFFERENT'}")
        if wa != wb:
            same = False

    print("\n" + ("EQUIVALENT: the demo shows the measured system."
                  if same else
                  "NOT EQUIVALENT: the demo must run from cached stage outputs instead."))


if __name__ == "__main__":
    main()
