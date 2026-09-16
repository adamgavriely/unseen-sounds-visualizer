"""Paired per-clip comparison of the three systems, with bootstrap CIs and an oracle bound.

Why this exists. The headline table reports means over 100 clips of a 0-4 score whose
distribution is bimodal, and at 25 clips per category a gap of 0.1 is noise. A review
(Fable, 2026-09-14) asked for what a reader can actually trust: per-clip paired
win/tie/loss between systems, a bootstrap confidence interval on the mean difference,
and the ceiling a perfect gate would reach on this benchmark -- the ORACLE-GATE bound,
which uses the human tag as the gate (show for unseen/mixed, silent for seen/none) and
takes the best-scoring system's output in each case.

    python scripts/paired_stats.py v2_grounded          # benchmark/protocol_results_v2_grounded.json
    python scripts/paired_stats.py v2_indep v3_indep    # several tags, one table each
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
SHOULD_SHOW = {"unseen_ambient", "mixed"}
SILENT_RIGHT = {"seen_ambient", "no_ambient"}


def load(tag: str):
    f = _ROOT / "benchmark" / (f"protocol_results_{tag}.json" if tag else "protocol_results.json")
    rows = json.loads(f.read_text(encoding="utf-8"))
    by = defaultdict(dict)
    for r in rows:
        if isinstance(r.get("score"), (int, float)):
            by[r["clip"]][r["system"]] = r
    return {c: v for c, v in by.items() if len(v) == 3}, f


def boot_ci(diffs, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    means = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def report(tag: str):
    data, f = load(tag)
    systems = ["proposed", "blind_a2i", "audio_caption"]
    n = len(data)
    print(f"\n=== {f.name}: {n} clips with all three systems ===")
    for s in systems:
        sc = [data[c][s]["score"] for c in data]
        lo, hi = boot_ci(sc)
        print(f"  {s:14s} mean {np.mean(sc):.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    for a, b in (("proposed", "blind_a2i"), ("proposed", "audio_caption")):
        d = [data[c][a]["score"] - data[c][b]["score"] for c in data]
        w = sum(1 for x in d if x > 0); t = sum(1 for x in d if x == 0); l = sum(1 for x in d if x < 0)
        lo, hi = boot_ci(d)
        print(f"  {a} vs {b}: win {w} / tie {t} / loss {l}   mean diff {np.mean(d):+.2f}  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]{'  (CI excludes 0)' if lo > 0 or hi < 0 else ''}")
    # per human tag
    print("  per human tag (mean score; n):")
    tags = sorted({data[c]["proposed"].get("human_tag") or "?" for c in data})
    for t in tags:
        cs = [c for c in data if (data[c]["proposed"].get("human_tag") or "?") == t]
        line = "  ".join(f"{s.split('_')[0]:9s}{np.mean([data[c][s]['score'] for c in cs]):.2f}" for s in systems)
        print(f"    {t:15s} ({len(cs):3d})  {line}")
    # by scenario group -- the proposal's own split, defined before any system ran:
    # a picture is due on unseen/mixed clips and not due on seen/no_ambient clips.
    # Same judge score in both strata (a per-stratum metric would be post-hoc).
    for label, group in (("picture needed", SHOULD_SHOW), ("no picture needed", SILENT_RIGHT)):
        cs = [c for c in data if (data[c]["proposed"].get("human_tag") or "?") in group]
        if not cs:
            continue
        means = "  ".join(f"{s.split('_')[0]:9s}{np.mean([data[c][s]['score'] for c in cs]):.2f}" for s in systems)
        d = [data[c]["proposed"]["score"] - data[c]["blind_a2i"]["score"] for c in cs]
        lo, hi = boot_ci(d)
        print(f"  {label:18s} (n={len(cs):3d})  {means}   proposed-blind {np.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]")
    # oracle gate: perfect show/silent decision from the human tag; the best system's
    # output where a picture is due, the abstention score where silence is right
    oracle = []
    for c in data:
        t = data[c]["proposed"].get("human_tag")
        if t in SILENT_RIGHT:
            # correct silence scores 4 by the rubric's own rule (nothing shown, nothing missing).
            # (Until 2026-09-16 this took the best score among systems with no images, which
            # let the captioning baseline's text score stand in for silence and put the oracle
            # below a system that showed a redundant picture the judge liked, on 12 clips.)
            oracle.append(4.0)
        else:
            oracle.append(max(data[c][s]["score"] for s in systems))
    lo, hi = boot_ci(oracle)
    print(f"  ORACLE gate + best output: mean {np.mean(oracle):.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
          f"(ceiling for any gating on this benchmark)")
    return {"tag": tag, "n": n,
            "means": {s: float(np.mean([data[c][s]["score"] for c in data])) for s in systems},
            "oracle": float(np.mean(oracle))}


if __name__ == "__main__":
    tags = sys.argv[1:] or ["v2", "v2_grounded"]
    out = [report(t) for t in tags]
    (_ROOT / "benchmark" / "paired_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
