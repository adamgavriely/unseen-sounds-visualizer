"""DEV only: per clip of the frozen arm, the proposed and blind_a2i augmentations and the gate's per-stretch votes
(name / a-b / describe) from the _trail arm's "gate" records. Input of step2_gate.py. Cluster, CPU only.

    python benchmark/gold/coverage/dump_gate.py [out.json]
"""
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import inspector_trail_export as X

ARM = "SHIP8+MD3+WW5+SL"
VOTES = re.compile(r"stretch (\d+): (seen|not seen) \(([^)]*)\)")
FIELD = re.compile(r"(name|a/b|desc) (yes|no|none|n/a|\?)")


def gate_votes(trail):
    out = []
    for r in trail:
        if r.get("step") != "gate":
            continue
        st = []
        for k, verdict, inner in VOTES.findall(r.get("note", "")):
            f = {a: b for a, b in FIELD.findall(inner)}
            st.append({"k": int(k), "seen": verdict == "seen",
                       "name": {"yes": True, "no": False}.get(f.get("name")), "ab": {"yes": True, "no": False}.get(f.get("a/b")),
                       "desc": {"yes": True, "no": False}.get(f.get("desc")), "raw": inner})
        out.append({"label": r["label"], "start": r["start"], "end": r["end"], "res": r.get("res"), "value": r.get("value"),
                    "stretches": st, "note": r.get("note", "")[:400]})
    return out


def main():
    res = {}
    for split, part, base, stems in X.parts(ARM):
        if split != "DEV":
            continue
        for st in stems:
            P = json.loads((base / f"{ARM}_proposed" / st / "augmentations.json").read_text(encoding="utf-8"))
            B = json.loads((base / f"{ARM}_blind_a2i" / st / "augmentations.json").read_text(encoding="utf-8"))
            tf = base / f"{ARM}_trail_proposed" / st / "trail.json"
            trail = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else []
            res[st] = {"part": part, "P": P, "B": B, "gate": gate_votes(trail)}
        print(split, part, len(stems), flush=True)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch_cov/gate_dev.json")
    out.write_text(json.dumps(res), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
