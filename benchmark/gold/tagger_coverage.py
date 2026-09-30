"""Every tagged clip (DEV and TEST parts) must have audio-LLM answers and DASM scores before any arm is scored: per clip,
item counts in <split>_listener{,_v,_afn}.json and the DASM file. Exits 1 if a clip has no yes/no items or no DASM file.

    python benchmark/gold/tagger_coverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
G, W = _ROOT / "benchmark" / "gold", _ROOT / "data" / "work"
bad = []
for split in ("dev2", "test2"):
    stems = [x for x in (G / f"{split}_stems.txt").read_text(encoding="utf-8").split() if x]
    cnt = {}
    for k in ("", "_v", "_afn"):
        its = json.loads((G / f"{split}_listener{k}.json").read_text(encoding="utf-8"))["items"]
        for x in its:
            cnt.setdefault(x["clip"], {}).setdefault(k or "yn", 0)
            cnt[x["clip"]][k or "yn"] += 1
    for s in stems:
        c = cnt.get(s, {})
        dasm = (W / f"dasm_{split}" / f"{s}.npz").exists()
        ok = c.get("yn", 0) > 0 and dasm
        bad += [] if ok else [s]
        print(f"[{split}] {s}: yes/no {c.get('yn', 0)}  variants {c.get('_v', 0)}  AF {c.get('_afn', 0)}  DASM {dasm}"
              + ("" if ok else "  <-- MISSING"))
print("coverage:", "all clips covered" if not bad else f"MISSING {bad}")
sys.exit(1 if bad else 0)
