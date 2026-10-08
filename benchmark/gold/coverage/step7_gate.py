"""Step 7 G (PREREG_step7_add_and_gate_combo.md): two-opinion gate (a/b share + Qwen3-Omni onset look). Cluster, CPU.
Same replay as step2_gate_build (both directions, gate-free spec when drawn), silence iff score >= s.
    python benchmark/gold/coverage/step7_gate.py -> scratch_cov/step7_gate_pics_dev.json"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import inspector_trail_export as X
from benchmark.gold.coverage.dump_gate import gate_votes
from benchmark.gold.coverage.hold_sweep import pictures
from benchmark.gold.coverage.step2_gate_build import FROZEN, TOL
try:
    from benchmark.gold import dev_harness as R
except ImportError:
    from benchmark.gold import round13_dev as R

HERE = Path(__file__).resolve().parent
SS = (0.4, 0.5, 0.6, 0.7)


def main():
    it = json.loads((HERE / "verify_items_dev.json").read_text(encoding="utf-8"))
    om = json.loads((HERE / "omni_verify_dev.json").read_text(encoding="utf-8"))["v2"]
    bursts = {}
    for b in it["bursts"]:
        bursts.setdefault(b["clip"], []).append(b)
    disp = {k: R.arm_cfg(FROZEN)[k] for k in R.DISPLAY_KEYS}
    out = {"cells": {f"G{s}": {"clips": {}} for s in SS}, "log": {f"G{s}": {"restored": 0, "silenced": 0, "no_omni": 0} for s in SS}}
    with R.flags({**disp, "MAX_AFTER_END": None}):
        for split, part, base, stems in X.parts(FROZEN):
            if split != "DEV":
                continue
            for st in stems:
                rp = base / f"{FROZEN}_proposed" / st
                P0 = json.loads((rp / "augmentations.json").read_text(encoding="utf-8"))
                B = json.loads((base / f"{FROZEN}_blind_a2i" / st / "augmentations.json").read_text(encoding="utf-8"))
                gates = gate_votes(json.loads((base / f"{FROZEN}_trail_proposed" / st / "trail.json").read_text(encoding="utf-8")))
                dur = float(json.loads((rp / "media.json").read_text(encoding="utf-8")).get("duration") or 0) or None
                for s_ in SS:
                    log = out["log"][f"G{s_}"]
                    P = copy.deepcopy(P0)
                    restored = set()
                    for i, (s, t) in enumerate(zip(P, B)):
                        if not t.get("augment") or t["event_label"] != s["event_label"]:
                            continue
                        g = [x for x in gates if x["label"] == s["event_label"] and abs(x["start"] - s["start"]) < TOL and abs(x["end"] - s["end"]) < TOL]
                        if not g or not g[0]["stretches"]:
                            continue
                        ab = float(np.mean([(v["ab"] if v["ab"] is not None else v["seen"]) for v in g[0]["stretches"]]))
                        ps = [om[b["id"]] for b in bursts.get(st, []) if b["id"] in om and S.same_family(b["family"], s["event_label"])
                              and b["end"] > s["start"] and b["start"] < s["end"]]
                        if not ps:
                            log["no_omni"] += 1
                        score = (ab + max(ps)) / 2 if ps else ab
                        silence = score >= s_
                        gated = (not s.get("augment")) and str(s.get("reason", "")).startswith("source visible on screen")
                        if gated and not silence:
                            P[i] = dict(t); restored.add(s["event_label"]); log["restored"] += 1
                        elif s.get("augment") and g[0]["res"] == "pass" and silence:
                            P[i] = dict(s); P[i]["augment"] = False; log["silenced"] += 1
                    for i, (s, t) in enumerate(zip(P, B)):
                        if not s.get("augment") and t.get("augment") and str(s.get("reason", "")).startswith("a kind of "):
                            if s["reason"][len("a kind of "):].split(",")[0] in restored:
                                P[i] = dict(t)
                    out["cells"][f"G{s_}"]["clips"][st] = {"pics_none": pictures(P, dur, st)}
    for k, v in out["log"].items():
        print(k, v)
    p = _ROOT / "scratch_cov" / "step7_gate_pics_dev.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    main()
