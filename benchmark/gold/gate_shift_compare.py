"""Week plan B.3 (docs/WEEK_PLAN_2026-09-26.md): is the gate's verdict stable when its frames move by 0.5 s?

Report only; nothing is adopted. Compares, per DEV gold sound (importance >= 2), the clip-level verdict (majority,
silent only if visible in every stretch) of three cached runs of the same model: the original run, a same-frames
repeat (the noise floor: card class, sampling) and a +0.5 s shift. Rule written before the runs: if the shift flips
more than 10 % of verdicts, the thesis states that the gate is frame-sensitive.

    python benchmark/gold/gate_shift_compare.py --base Qwen38-27B --repeat Qwen38-27B_repeat --shift Qwen38-27B_shift0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmark.gold.gate_gold import OUT_DIR, decide


def verdicts(arm):
    out = {}
    for f in (OUT_DIR / arm).glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for k, s in enumerate(d["sounds"]):
            if s["importance"] >= 2:
                out[(f.stem, k)] = (decide(s["stretches"], "majority"), s["seen"], s["label"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--repeat", required=True); ap.add_argument("--shift", required=True)
    a = ap.parse_args()
    B, R, S = verdicts(a.base), verdicts(a.repeat), verdicts(a.shift)
    keys = sorted(set(B) & set(R) & set(S))
    fr = sum(B[k][0] != R[k][0] for k in keys); fs = sum(B[k][0] != S[k][0] for k in keys)
    print(f"{len(keys)} DEV sounds (importance >= 2)")
    print(f"flipped by a same-frames repeat (noise floor): {fr} ({fr / len(keys):.1%})")
    print(f"flipped by a +0.5 s frame shift:             {fs} ({fs / len(keys):.1%})")
    for k in keys:
        if B[k][0] != S[k][0]:
            print(f"   shift flip: {k[0]} {B[k][2]!r} annotator visible={B[k][1]}  base silent={B[k][0]} -> shifted silent={S[k][0]}")
    verdict = "FRAME-SENSITIVE (> 10 %)" if fs / len(keys) > 0.10 else "stable within 10 %"
    print("verdict:", verdict)
    json.dump({"n": len(keys), "repeat_flips": fr, "shift_flips": fs, "verdict": verdict},
              open(OUT_DIR / "shift_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
