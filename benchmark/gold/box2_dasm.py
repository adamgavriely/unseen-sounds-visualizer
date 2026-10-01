"""Diagnostic for a "more hits" combination (1 Oct): the pictures the BOX-2 gate arm adds over SHIP8 (both at the shipped
MERGE_GAP 2.5), each with its family's DASM max at onset +- 0.5 s (K4A-D's bar 0.575) and its class. CPU, caches only.
    TG_ARMS=SHIP8 python benchmark/gold/box2_dasm.py        (from ~/MscProj_tg)"""
import json
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold.expect_a4_screen import dasm_max
from src.labels import canonical

arm = sys.argv[1]
B.ARM = arm
rows = []
for pt, st, g, pics in B.parts():
    pics = [tuple(p[:3]) for p in pics]
    cls = dict(zip(sorted(pics, key=lambda p: p[1]), CG.classify(g, pics)))
    f = CG.PARTS[pt]["dasm"] / f"{st}.npz"
    fr = DCC.load_fr(f) if f.exists() else None
    for p in pics:
        rows.append({"part": pt, "clip": st, "label": p[0], "start": round(p[1], 2), "class": cls.get(p),
                     "dasm": dasm_max(fr, canonical(p[0]), p[1] - 0.5, p[1] + 0.5)})
(_ROOT / "benchmark" / "gold" / f"box2_dasm_{arm.replace('+', '_')}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(arm, len(rows))
