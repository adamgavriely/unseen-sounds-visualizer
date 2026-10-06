"""Holm correction over the report's five tests of the final system (Section 6.4 of the report):

  test-set cost vs direct audio-to-image, test-set cost vs showing nothing, test-set F1 vs direct audio-to-image,
  development-set cost vs showing nothing, development-set cost vs direct audio-to-image.

The two-sided p-values are read from the result files (100,000-draw paired clip bootstrap); nothing is re-sampled.
Holm's step-down rule: sort the p-values, multiply the k-th smallest by (5 - k + 1), and keep the running maximum.
    python benchmark/gold/holm_five.py
"""
from __future__ import annotations

import json
from pathlib import Path

GOLD = Path(__file__).resolve().parent


def holm(ps):
    """dict name -> p  ->  dict name -> Holm-adjusted p"""
    out, run = {}, 0.0
    for k, (name, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, p * (len(ps) - k)))
        out[name] = run
    return out


def main():
    base = json.loads((GOLD / "final_vs_baselines.json").read_text(encoding="utf-8"))
    extra = json.loads((GOLD / "final_vs_baselines_extra.json").read_text(encoding="utf-8"))
    ps = {"test, cost vs direct audio-to-image": base["test"]["diff"]["final_vs_audio_to_image"][3],
          "test, cost vs showing nothing": base["test"]["diff"]["final_vs_show_nothing"][3],
          "test, F1 vs direct audio-to-image": extra["test"]["f1_diff_final_vs_audio_to_image"][3],
          "development, cost vs showing nothing": base["development"]["diff"]["final_vs_show_nothing"][3],
          "development, cost vs direct audio-to-image": base["development"]["diff"]["final_vs_audio_to_image"][3]}
    adj = holm(ps)
    for name, p in ps.items():
        print(f"{name:44s} p {p:.5f}   Holm {adj[name]:.5f}")


if __name__ == "__main__":
    main()
