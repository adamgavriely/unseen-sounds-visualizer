"""DEV only: per clip and per drawn label of the frozen arm, the family's frame scores of BEATs, FlexSED and DASM
(max over the model's classes in the label's family, score_per_sound.same_family) and the gate's per-stretch verdicts
(the _trail arm's "gate" records). Input of hold_sweep.py. Runs on the cluster checkout of the scored runs, CPU only.

    python benchmark/gold/coverage/dump_evidence.py [out.json]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import inspector_trail_export as X
from benchmark.gold import dev_candidates_check as DCC

ARM = "SHIP8+MD3+WW5+SL"
WORK = DCC.WORK
CACHES = {"dev": {"beats": WORK / "j2_dev_beats", "flex": WORK / "flexsed_cache", "dasm": WORK / "devcand" / "dasm_cache"},
          "dev2": {"beats": WORK / "j2_dev2_beats", "flex": WORK / "flexsed_cache", "dasm": WORK / "dasm_dev2"}}
STRETCH = re.compile(r"stretch (\d+) \(([\d.]+)-([\d.]+) s")
VERDICT = re.compile(r"stretch (\d+): (seen|not seen)")


def gate_records(trail):
    out = []
    for r in trail:
        if r.get("step") != "gate":
            continue
        times = {}
        for a in r.get("asks", []) or []:
            m = STRETCH.search(a.get("who", ""))
            if m:
                times[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        verd = {int(k): v for k, v in VERDICT.findall(r.get("note", ""))}
        if not times and len(verd) == 1:                      # one stretch = the whole sound
            times[1] = (float(r["start"]), float(r["end"]))
        out.append({"label": r["label"], "start": r["start"], "end": r["end"], "res": r.get("res"),
                    "stretches": [[times[k][0], times[k][1], verd[k] == "seen"] for k in sorted(verd) if k in times],
                    "unparsed": sorted(set(verd) - set(times))})
    return out


def fam_curve(fr, label):
    fw, t, labs = fr
    idx = [i for i, c in enumerate(labs) if S.same_family(c, label)]
    if not idx:
        return None
    return {"t": np.round(t, 3).tolist(), "v": np.round(fw[:, idx].max(axis=1), 4).tolist(),
            "classes": [labs[i] for i in idx][:20], "n_classes": len(idx)}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch_cov/evidence_dev.json")
    res = {}
    for split, part, base, stems in X.parts(ARM):
        if split != "DEV":
            continue
        root = base / f"{ARM}_proposed"
        troot = base / f"{ARM}_trail_proposed"
        for st in stems:
            aug = json.loads((root / st / "augmentations.json").read_text(encoding="utf-8"))
            blind = base / f"{ARM}_blind_a2i" / st / "augmentations.json"      # the gate-free specs: every label any gate rule could draw
            aug_b = json.loads(blind.read_text(encoding="utf-8")) if blind.exists() else []
            labels = sorted({s["event_label"] for s in aug + aug_b if s.get("augment")})
            tf = troot / st / "trail.json"
            trail = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else []
            rec = {"part": part, "labels": {}, "gate": gate_records(trail), "has_trail": tf.exists()}
            frs = {}
            for m, d in CACHES[part].items():
                f = d / f"{st}.npz"
                frs[m] = DCC.load_fr(f) if f.exists() else None
            for lab in labels:
                rec["labels"][lab] = {m: (fam_curve(fr, lab) if fr is not None else None) for m, fr in frs.items()}
            res[st] = rec
        print(split, part, len(stems), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
