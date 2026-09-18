"""The v4 cumulative table (docs/prereg_v4.md): one row per configuration, the same judge
and describer per row, gated vs blind with a paired bootstrap CI, the oracle, and the
panel-on cost when cost_metrics has been run for that tag.

    python scripts/v4_table.py                       # every tag it can find
    python scripts/v4_table.py v3_grounded v3_q38_grounded v4a_grounded
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from scripts.paired_stats import load, boot_ci, SILENT_RIGHT

ROWS = [("v3_grounded", "v3 (Mistral judge, Qwen2.5-VL-7B describer) — the thesis row"),
        ("v3_grounded_rubric", "v3, rubric-enforced judge"),
        ("v3_q38_grounded_rubric", "v3 panels, v4 pair, rubric-enforced"),
        ("v4b_grounded_rubric", "v4b, rubric-enforced"),
        ("v4ab_bar035_grounded_rubric", "v4ab at the WRONG bar 0.35 (discarded), rubric"),
        ("v4ab_grounded_rubric", "v4ab (PSED at 0.20 + Qwen3.8), rubric-enforced"),
        ("v3_q38_grounded", "v3 panels, v4 evaluation pair"),
        ("v4a_grounded", "detector arm: PSED + Qwen2.5-VL-7B"),
        ("v4b_grounded", "+ Qwen3.8-27B gate (stage 5); BEATs stays"),
        ("v4ab_grounded", "detector arm: PSED + Qwen3.8-27B"),
        ("v4c_grounded", "+ Qwen-Image-2512 (stage 6)"),
        ("v4_grounded", "+ Granite, SAM 3 (stages 3, 2)")]


def row(tag: str):
    try:
        data, f = load(tag)
    except FileNotFoundError:
        return None
    if not data:
        return None
    systems = ["proposed", "blind_a2i", "audio_caption"]
    means = {s: float(np.mean([data[c][s]["score"] for c in data])) for s in systems}
    d = [data[c]["proposed"]["score"] - data[c]["blind_a2i"]["score"] for c in data]
    lo, hi = boot_ci(d)
    oracle = [4.0 if data[c]["proposed"].get("human_tag") in SILENT_RIGHT else max(data[c][s]["score"] for s in systems) for c in data]
    judge = next(iter(data.values()))["proposed"].get("judge_model", "?")
    cost = None
    cf = _ROOT / "benchmark" / f"cost_metrics_{tag.replace('_grounded', '')}.json"
    if cf.exists():
        try:
            cm = json.loads(cf.read_text(encoding="utf-8"))
            cost = {k: cm[k].get("panel_on_frac") for k in ("proposed", "blind_a2i") if k in cm}
        except Exception:
            cost = None
    return {"tag": tag, "n": len(data), "judge": judge, "means": means, "gated_minus_blind": float(np.mean(d)),
            "ci": [lo, hi], "oracle": float(np.mean(oracle)), "cost": cost}


def main():
    tags = sys.argv[1:] or [t for t, _ in ROWS]
    desc = dict(ROWS)
    out = []
    print(f"{'row':44s} {'n':>3} {'gated':>6} {'blind':>6} {'capt.':>6} {'gated-blind [95% CI]':>24} {'oracle':>6}  judge")
    for t in tags:
        r = row(t)
        if r is None:
            print(f"{desc.get(t, t):44s}   — (no results yet)"); continue
        m = r["means"]; lo, hi = r["ci"]
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{desc.get(t, t)[:44]:44s} {r['n']:3d} {m['proposed']:6.2f} {m['blind_a2i']:6.2f} {m['audio_caption']:6.2f} "
              f"{r['gated_minus_blind']:+6.2f} [{lo:+.2f}, {hi:+.2f}]{sig} {r['oracle']:6.2f}  {r['judge'].split('/')[-1]}")
        out.append(r)
    (_ROOT / "benchmark" / "v4_table.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
