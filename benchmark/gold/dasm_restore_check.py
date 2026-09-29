"""DASM restore check (2026-09-30): ~/Transformer4SED was deleted on 28 Sept and re-fetched (GitHub cai525/Transformer4SED
HEAD c3e883d, HF CPF2/detect_any_sound htast_mga.pt + text_query/). Re-scores 4 DEV clips with dev_candidates_check.dasm()
into a scratch folder and compares them with the existing DEV cache (data/work/devcand/dasm_cache, job 31330562).
Pass = same frame grid and max |diff| < 1e-3 on every clip; exit code 1 otherwise. No gold is read.

    python benchmark/gold/dasm_restore_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC

STEMS = ["ambient_citywalk_nyc_1689", "as_explosion_XJ8lc3I6", "b3_laundromat", "birds_forest"]
TOL = 1e-3


def main():
    ref = DCC.DASM_DIR.resolve()
    g, stems = DCC.dev_stems()
    assert all(s in stems for s in STEMS), [s for s in STEMS if s not in stems]
    scratch = (DCC.WORK / "dasm_restore_check").resolve()
    for f in scratch.glob("*.npz"):
        f.unlink()
    DCC.dev_stems = lambda: (g, list(STEMS))
    DCC.DASM_DIR = scratch
    DCC.dasm()
    res = {}
    for st in STEMS:
        a, b = np.load(ref / f"{st}.npz"), np.load(scratch / f"{st}.npz")
        same_grid = a["fw"].shape == b["fw"].shape and np.allclose(a["times"], b["times"])
        d = float(np.abs(a["fw"] - b["fw"]).max()) if same_grid else None
        res[st] = {"shape_old": list(a["fw"].shape), "shape_new": list(b["fw"].shape), "max_abs_diff": d,
                   "pass": bool(same_grid and d is not None and d < TOL)}
    ok = all(r["pass"] for r in res.values())
    print(json.dumps(res, indent=1))
    print(f"DASM restore check: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
