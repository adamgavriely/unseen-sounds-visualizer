"""Step 2 (PREREG_step2_gate_mirror.md): re-decide each arm's gate under the a/b rule and draw the pictures. Cluster, CPU.

For every DEV clip of each arm: the gate records of the arm's trail (votes per stretch), the arm's proposed specs and its
blind_a2i specs (= the same stage 5 without the gate). Rule M (majority) must reproduce the arm's own pictures; AB-s / AB-m
re-decide every gated sound in both directions (a shown sound whose stretches are all a/b-seen is silenced; a silenced
one with an a/b-not-seen stretch is drawn as its blind_a2i spec), and a "kind of X" sound follows a restored parent.

    python benchmark/gold/coverage/step2_gate_build.py [arm ...]  -> scratch_cov/step2_pics_dev.json
"""
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import inspector_trail_export as X
from benchmark.gold.coverage.dump_gate import gate_votes
from benchmark.gold.coverage.hold_sweep import pictures
try:
    from benchmark.gold import dev_harness as R
except ImportError:
    from benchmark.gold import round13_dev as R

FROZEN = "SHIP8+MD3+WW5+SL"
TOL = 0.011


def seen(v, rule):
    if rule == "M":
        return v["seen"]
    if v["ab"] is None:                                     # split a/b answer
        return False if rule == "AB-s" else v["seen"]
    return bool(v["ab"])


def redecide(P, B, gates, rule, log):
    P = copy.deepcopy(P)
    restored = set()
    for i, (s, t) in enumerate(zip(P, B)):
        if not t.get("augment") or t["event_label"] != s["event_label"]:
            continue
        g = [x for x in gates if x["label"] == s["event_label"] and abs(x["start"] - s["start"]) < TOL and abs(x["end"] - s["end"]) < TOL]
        gated = (not s.get("augment")) and str(s.get("reason", "")).startswith("source visible on screen")
        if not g or not g[0]["stretches"]:
            if gated:
                log["no_votes"] += 1
            continue
        silence = all(seen(v, rule) for v in g[0]["stretches"])
        if gated and not silence:
            P[i] = dict(t); restored.add(s["event_label"]); log["restored"] += 1
        elif s.get("augment") and g[0]["res"] == "pass" and silence:
            P[i] = dict(s); P[i]["augment"] = False; P[i]["reason"] = "source visible on screen (a/b rule)"; log["silenced"] += 1
    for i, (s, t) in enumerate(zip(P, B)):                 # family rule: a kind of X follows a restored X
        if not s.get("augment") and t.get("augment") and str(s.get("reason", "")).startswith("a kind of "):
            par = s["reason"][len("a kind of "):].split(",")[0]
            if par in restored:
                P[i] = dict(t); log["restored_kind"] += 1
    return P


def main():
    arms = sys.argv[1:] or [FROZEN]
    disp = {k: R.arm_cfg(FROZEN)[k] for k in R.DISPLAY_KEYS}
    out = {"cells": {}, "log": {}}
    with R.flags({**disp, "MAX_AFTER_END": None}):
        for arm in arms:
            tarm = FROZEN + "_trail" if arm == FROZEN else arm
            for rule in ("M", "AB-s", "AB-m"):
                cell = f"{arm}|{rule}"
                out["cells"][cell] = {"clips": {}}
                log = out["log"][cell] = {"restored": 0, "silenced": 0, "restored_kind": 0, "no_votes": 0, "M_differs": []}
                for split, part, base, stems in X.parts(FROZEN):
                    if split != "DEV":
                        continue
                    for st in stems:
                        rp = base / f"{arm}_proposed" / st
                        P = json.loads((rp / "augmentations.json").read_text(encoding="utf-8"))
                        B = json.loads((base / f"{arm}_blind_a2i" / st / "augmentations.json").read_text(encoding="utf-8"))
                        tf = base / f"{tarm}_proposed" / st / "trail.json"
                        gates = gate_votes(json.loads(tf.read_text(encoding="utf-8"))) if tf.exists() else []
                        dur = float(json.loads((rp / "media.json").read_text(encoding="utf-8")).get("duration") or 0) or None
                        pics = pictures(redecide(P, B, gates, rule, log), dur, st)
                        if rule == "M" and pics != pictures(P, dur, st):
                            log["M_differs"].append(st)
                        out["cells"][cell]["clips"][st] = {"pics_none": pics}
                print(cell, log, flush=True)
    p = _ROOT / "scratch_cov" / "step2_pics_dev.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    main()
