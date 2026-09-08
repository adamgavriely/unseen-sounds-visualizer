"""Compare two protocol runs that scored the SAME clips, pair by pair.

Two experiments need exactly this, and they only differ in what was held constant:

  judge reliability   same descriptions, different judge model. Agreement bounds how
                      far any LLM-judged number in this thesis can be trusted -- the
                      protocol's softest point, and the one an examiner will press.
                        python scripts/compare_runs.py protocol_results.json \
                                                       protocol_results_judge2.json
  generator ablation  same judge, images generated instead of retrieved. The proposal
                      names diffusion; nobody has checked that generating beats
                      retrieving, and retrieval is far cheaper if it ties.
                        python scripts/compare_runs.py protocol_results.json \
                                                       protocol_results_retrieve.json

Paired throughout: every statistic is computed over clips both runs actually scored,
so a run that crashed halfway cannot flatter itself by being compared on its easy half.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"


def load(name):
    p = Path(name)
    if not p.exists():
        p = BENCH / name
    if not p.exists():
        sys.exit(f"not found: {name}")
    return {(r["clip"], r["system"]): r for r in
            json.loads(p.read_text(encoding="utf-8"))}, p


def spearman(a, b):
    """Rank correlation, average ranks for ties. Hand-rolled to avoid a scipy dep."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def quadratic_kappa(a, b, k=5):
    """Weighted kappa: near-misses on an ordinal 0-4 scale should not count as full
    disagreement, so raw exact-match understates agreement badly here."""
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ha = [a.count(i) for i in range(k)]
    hb = [b.count(i) for i in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) / (k - 1)) ** 2
            num += w * obs[i][j]
            den += w * ha[i] * hb[j] / n
    return 1 - num / den if den else 1.0


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, pa = load(sys.argv[1])
    b, pb = load(sys.argv[2])
    keys = sorted(set(a) & set(b))
    if not keys:
        sys.exit("the two runs share no (clip, system) pairs")
    sa = [int(a[k]["score"]) for k in keys]
    sb = [int(b[k]["score"]) for k in keys]

    print("=" * 72)
    print(f"A = {pa.name}   mean {sum(sa)/len(sa):.2f}")
    print(f"B = {pb.name}   mean {sum(sb)/len(sb):.2f}")
    print(f"paired on {len(keys)} (clip, system) records "
          f"[A had {len(a)}, B had {len(b)}]")
    exact = sum(1 for x, y in zip(sa, sb) if x == y)
    within1 = sum(1 for x, y in zip(sa, sb) if abs(x - y) <= 1)
    print()
    print(f"  mean difference B-A   {(sum(sb)-sum(sa))/len(sa):+.2f}")
    print(f"  exact agreement       {100*exact/len(keys):.0f}%")
    print(f"  within 1 point        {100*within1/len(keys):.0f}%")
    print(f"  Spearman rho          {spearman(sa, sb):.3f}")
    print(f"  quadratic kappa       {quadratic_kappa(sa, sb):.3f}")

    print()
    print(f"  {'system':16}{'A':>7}{'B':>7}{'B-A':>8}{'n':>6}")
    per = defaultdict(lambda: ([], []))
    for k in keys:
        per[k[1]][0].append(int(a[k]["score"]))
        per[k[1]][1].append(int(b[k]["score"]))
    for sysname, (xa, xb) in sorted(per.items()):
        ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
        print(f"  {sysname:16}{ma:>7.2f}{mb:>7.2f}{mb-ma:>+8.2f}{len(xa):>6}")

    # The records where the two runs disagree most are the ones worth reading by hand;
    # an aggregate correlation says nothing about WHY they differ.
    print()
    print("  largest disagreements:")
    worst = sorted(keys, key=lambda k: -abs(int(a[k]["score"]) - int(b[k]["score"])))
    for k in worst[:6]:
        d = abs(int(a[k]["score"]) - int(b[k]["score"]))
        if d == 0:
            break
        print(f"   {k[0][:26]:26} {k[1]:14} A={a[k]['score']} B={b[k]['score']}")
        print(f"      ref : {a[k]['reference'][:88]}")
        print(f"      desc: {a[k]['description'][:88]}")


if __name__ == "__main__":
    main()
