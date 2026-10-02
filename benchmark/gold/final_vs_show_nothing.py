"""Paired clip bootstrap: final system vs show nothing, from the Decision Inspector per-clip export.

Show nothing costs 4 x needed sounds per clip (every needed sound is a miss, no wrong pictures), so this test
needs only the final system's per-clip rows. Run from the repository root:
    python benchmark/gold/final_vs_show_nothing.py
Output (2 Oct 2026) is in final_vs_show_nothing.txt. 10,000 draws, seed 0, two-sided p.
"""
import json, random
s = open("docs/inspector2/data.js", encoding="utf-8").read()
d = json.loads(s[s.index("{"):s.rindex("}") + 1])
for split in ("DEV", "TEST"):
    rows = []
    H = M = W = 0
    for c in d["clips"]:
        if c["split"] != split:
            continue
        h = sum(g["outcome"] == "hit" for g in c["gold"])
        m = sum(g["outcome"] == "miss" for g in c["gold"])
        w = sum(p["verdict"] in ("visible", "cross", "phantom") for p in c["pictures"])
        H += h; M += m; W += w
        rows.append(((4 * m + 2 * w), 4 * (h + m)))  # (final cost, show-nothing cost)
    n = len(rows)
    final = sum(r[0] for r in rows) / n; nothing = sum(r[1] for r in rows) / n
    d0 = final - nothing
    rng = random.Random(0); ds = []
    for _ in range(10000):
        smp = [rows[rng.randrange(n)] for _ in range(n)]
        ds.append(sum(a - b for a, b in smp) / n)
    ds.sort()
    lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]
    p_ge = sum(x >= 0 for x in ds) / len(ds); p_le = sum(x <= 0 for x in ds) / len(ds)
    p2 = min(1.0, 2 * min(p_ge, p_le))
    print(f"{split}: clips {n} hits {H} misses {M} wrong {W} | final {final:.3f} nothing {nothing:.3f} d {d0:+.3f} 95% [{lo:+.3f}, {hi:+.3f}] two-sided p {p2:.4f} (draws with d>=0: {p_ge})")
