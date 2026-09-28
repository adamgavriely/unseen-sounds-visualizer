"""Adds to docs/inspector/data.json, for every gold sound, what each detector heard around its onset
([onset - 0.5, onset + 1.0] s): the top-5 labels by peak score of BEATs (527 AudioSet classes), PANNs (527) and FlexSED
(215 drawable families asked by name), plus the peak score of the annotated sound's own family in each. Local caches only.

    python docs/inspector/add_heard.py && python docs/inspector/build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.audioset_stage4_report import load
from src.labels import canonical

DATA = _ROOT / "docs" / "inspector" / "data.json"
CACHES = {"BEATs": _ROOT / "benchmark" / "gold" / "beats_fw", "PANNs": _ROOT / "benchmark" / "gold" / "panns_fw",
          "FlexSED": _ROOT / "data" / "work" / "flexsed_cache"}
BARS = {"BEATs": 0.35, "PANNs": None, "FlexSED": 0.8}


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    cache = {}
    for cl in d["clips"]:
        for name, root in CACHES.items():
            p = root / f"{cl['clip']}.npz"
            cache[name] = load(p) if p.exists() else None
        for s in cl["sounds"]:
            a, b = s["start"] - 0.5, s["start"] + 1.0
            heard = {}
            for name, fr in cache.items():
                if fr is None:
                    continue
                fw, ts, labs = fr
                m = (ts >= a) & (ts <= b)
                if not m.any():
                    continue
                pk = fw[m].max(axis=0)
                top = np.argsort(-pk)[:5]
                fam = canonical(s["label"])
                own = [float(pk[i]) for i, l in enumerate(labs) if canonical(l) == fam]
                heard[name] = {"top": [[labs[i], round(float(pk[i]), 3)] for i in top],
                               "own": round(max(own), 3) if own else None, "bar": BARS[name]}
            s["heard"] = heard
    DATA.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print("added 'heard' to", sum(len(c["sounds"]) for c in d["clips"]), "sounds")


if __name__ == "__main__":
    main()
