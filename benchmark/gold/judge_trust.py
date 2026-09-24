"""Before we report what the judge says about us, does the judge agree with the annotator? (2026-09-24)

Adam asked for the VLM judge, and separately asked how we know the judge does a good job. Those are
one question: the judge's ranking of the three systems is only reportable if the judge sees what the
annotator sees. So this runs the checks written down in docs/judge_plan.md BEFORE the ranking.

  B1  does the judge's score fall as the annotator-derived viewer cost rises?
      Spearman rank correlation over the DEV clips, bootstrap CI. Pass: rho <= -0.4, CI excludes 0.
  B2  does the judge notice a picture we already know is wrong?
      Clips carrying at least one wrong picture vs clips carrying none. Pass: clean clips score
      higher and the gap's CI excludes 0.
  B3  the ranking itself, reported only if B1 and B2 pass.

    python benchmark/gold/judge_trust.py --results benchmark/protocol_results_dev_symgen_v30_grounded_rubric.json \\
                                         --specs data/work/protocol_proposed_dev_symgen_v30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD


def spearman(x, y):
    def rank(v):
        o = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
        # average ties, or the correlation is wrong wherever the judge repeats a score
        for val in set(v):
            m = np.array([a == val for a in v])
            if m.sum() > 1:
                o[m] = o[m].mean()
        return o
    a, b = rank(list(x)), rank(list(y))
    a, b = (a - a.mean()) / (a.std() + 1e-12), (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())


def boot(fn, pairs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(pairs), len(pairs))
        s = [pairs[i] for i in idx]
        try:
            vals.append(fn(s))
        except Exception:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def clip_rows(specs_root: Path, system: str, gold):
    """per clip: the annotator-derived cost, and how many pictures were wrong"""
    out = {}
    for stem, snds in gold.items():
        pics = S.load_pictures(specs_root, stem, system)
        if pics is None:
            continue
        r = S.score_clip(snds, pics)
        wrong = r["visible"] + r["cross"] + r["phantom"]
        out[stem] = {"cost": S.COST_MISS * r["miss"] + S.COST_FA * wrong,
                     "wrong": wrong, "miss": r["miss"], "hit": r["hit"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--system", default="proposed")
    ap.add_argument("--score-key", default="score")
    a = ap.parse_args()

    rows = json.loads(Path(a.results).read_text(encoding="utf-8"))
    gold = S.load_gold([GOLD])
    ours = clip_rows(Path(a.specs), a.system, gold)

    judged = {}
    for r in rows:
        if r.get("system") != a.system:
            continue
        stem = Path(r["clip"]).stem
        if r.get(a.score_key) is not None:
            judged[stem] = float(r[a.score_key])
    shared = sorted(set(judged) & set(ours))
    print(f"{len(rows)} judged rows, {len(judged)} for '{a.system}', "
          f"{len(ours)} scored clips, {len(shared)} in both\n")
    if len(shared) < 10:
        print("too few clips in common to say anything")
        return

    # ---- B1: does the judge's score track the annotator's cost?
    pairs = [(ours[s]["cost"], judged[s]) for s in shared]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    lo, hi = boot(lambda ps: spearman([p[0] for p in ps], [p[1] for p in ps]), pairs)
    ok1 = rho <= -0.4 and hi < 0
    print("B1  judge score vs the annotator's viewer cost")
    print(f"      Spearman rho {rho:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"{'PASS' if ok1 else 'FAIL'}  (pass: rho <= -0.40 and the CI excludes 0)")

    # ---- B2: does it notice a picture we already know is wrong?
    clean = [judged[s] for s in shared if ours[s]["wrong"] == 0]
    dirty = [judged[s] for s in shared if ours[s]["wrong"] > 0]
    print("\nB2  clips with a wrong picture vs clips without")
    if clean and dirty:
        gap = float(np.mean(clean) - np.mean(dirty))
        ps = [(1, v) for v in clean] + [(0, v) for v in dirty]

        def g(sample):
            c = [v for k, v in sample if k == 1]
            d = [v for k, v in sample if k == 0]
            if not c or not d:
                raise ValueError
            return float(np.mean(c) - np.mean(d))
        glo, ghi = boot(g, ps)
        ok2 = gap > 0 and glo > 0
        print(f"      clean {np.mean(clean):.2f} (n={len(clean)})   with a wrong picture "
              f"{np.mean(dirty):.2f} (n={len(dirty)})")
        print(f"      gap {gap:+.2f}   95% CI [{glo:+.2f}, {ghi:+.2f}]   "
              f"{'PASS' if ok2 else 'FAIL'}  (pass: gap > 0 and the CI excludes 0)")
    else:
        ok2 = False
        print("      one of the two groups is empty; cannot test")

    # ---- B3: the ranking, reported only if the judge earned it
    print("\nB3  what the judge says about the three systems")
    by_sys = {}
    for r in rows:
        if r.get(a.score_key) is None:
            continue
        by_sys.setdefault(r.get("system", "?"), []).append(float(r[a.score_key]))
    for sysname, vals in sorted(by_sys.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"      {sysname:16s} mean {np.mean(vals):.2f}  median {np.median(vals):.1f}  n={len(vals)}")
    if ok1 and ok2:
        print("\n      Both checks passed: this ranking is reportable as a second opinion.")
    else:
        print("\n      A check FAILED, so the ranking above is NOT reportable. The judge does not")
        print("      agree with the annotator on these clips, and that disagreement is the finding.")


if __name__ == "__main__":
    main()
