"""Cost when one kind of wrong picture is priced 0 or 4 and the other two stay at 2 (a miss costs 4).
Report Table 'kindprice' (Section sec:sensitivity). Paired clip bootstrap, 100,000 draws, seed 0.
Reads the saved per-clip results of the final system and direct audio-to-image generation.
Run: python benchmark/gold/kind_price_sweep.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "one_model_baseline", "result_m2d.json")
KINDS = [("on screen", 1), ("other sound", 2), ("nothing", 3)]


def load(split):
    rows = json.load(open(SRC))[split]["per_clip"]
    final = [r["proposed"] for r in rows]
    a2i = [r["blind_a2i"] for r in rows]
    nothing = [{"miss": r["hit"] + r["miss"], "visible": 0, "cross": 0, "phantom": 0} for r in final]
    return {"final": final, "a2i": a2i, "nothing": nothing}


def cost(rows, w):
    return np.array([w[0] * r["miss"] + w[1] * r["visible"] + w[2] * r["cross"] + w[3] * r["phantom"] for r in rows], float)


def boot(d, n=100000, seed=0):
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), min(1.0, 2 * min((m >= 0).mean(), (m <= 0).mean()))


for split in ("development", "test"):
    S = load(split)
    print(f"\n{split} ({len(S['final'])} clips)")
    rows = [("none (all 2)", (4, 2, 2, 2))]
    for kind, pos in KINDS:
        for v in (0, 4):
            w = [4, 2, 2, 2]; w[pos] = v; rows.append((f"{kind} {v}", tuple(w)))
    for name, w in rows:
        c = {k: cost(v, w) for k, v in S.items()}
        d1, p1 = boot(c["final"] - c["nothing"]); d2, p2 = boot(c["final"] - c["a2i"])
        print(f"{name:14s} final {c['final'].mean():.3f}  a2i {c['a2i'].mean():.3f}  nothing {c['nothing'].mean():.3f}"
              f"  final-nothing {d1:+.3f} (p {p1:.3f})  final-a2i {d2:+.3f} (p {p2:.3f})")
