"""Graphs for the 26 Sept supervisor page, from committed result files only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig"
INK, MUTED, OURS, BLIND, SIL, GOOD = "#1d2433", "#6b7385", "#2f6fdb", "#c9772b", "#9aa3b2", "#1f9d6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


t = json.loads((R / "benchmark/gold/holm_test_final_v33_test_bench.json").read_text(encoding="utf-8"))

# 1. headline: ours vs blind on TEST, effect +/- CI, Holm survivors marked
rows = [("Precision", "dP", 1), ("Wrong pictures / clip", "dFA/clip", -1), ("Viewer cost / clip", "d cost/clip", -1),
        ("Clean-clip accuracy", "d clean-acc", 1), ("Recall", "dR", 1)]
fam = {r["row"]: r for r in t["family1"]}
fig, ax = plt.subplots(figsize=(7.2, 3.3))
labels = ["F1 (primary)"] + [r[0] for r in rows]
vals = [(t["primary"]["d"], t["primary"]["ci"], False)] + [(fam[k]["d"] * s, sorted(c * s for c in fam[k]["ci"]), fam[k]["survives"]) for _, k, s in rows]
for i, (d, ci, ok) in enumerate(vals):
    y = len(vals) - 1 - i
    col = GOOD if ok and d > 0 else (BLIND if ok else MUTED)
    ax.plot(ci, [y, y], color=col, lw=3, solid_capstyle="round")
    ax.plot([d], [y], "o", color=col, ms=9)
    ax.text(ci[1] + 0.03, y, ("significant" if ok else "n.s."), va="center", color=col, fontsize=10)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels[::-1])
ax.set_xlabel("difference, ours − blind (right = ours better; bar = 95% CI; “significant” = survives Holm)")
ax.set_title("TEST, 60 clips: gate vs “draw every sound”", loc="left", color=INK, fontsize=12)
save(fig, "test_vs_blind.png")

# 2. viewer cost per clip category, ours / blind / silence
cats = [("cat_unseen", "sound off screen"), ("cat_mixed", "mixed"), ("cat_seen", "source on screen"), ("cat_no_ambient", "nothing to draw")]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
w = 0.26
for j, (sys_, col, lab) in enumerate((("proposed", OURS, "ours"), ("blind_a2i", BLIND, "draw every sound"), ("silence", SIL, "show nothing"))):
    xs = [i + (j - 1) * w for i in range(len(cats))]
    ax.bar(xs, [t["categories"][c]["cost"][sys_] for c, _ in cats], w, color=col, label=lab)
ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"{n}\n({t['categories'][c]['clips']} clips)" for c, n in cats])
ax.set_ylabel("viewer cost per clip (lower = better)")
ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0)); ax.set_ylim(0, 9)
ax.set_title("TEST: where the system pays off", loc="left", color=INK, fontsize=12, pad=28)
save(fig, "cost_by_category.png")

# 3. pictures: blind human recognition, round 2 (54 sounds, fresh clips)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
arms = [("today (FLUX)", 14), ("new model (Qwen-Image)", 26), ("new model + new text", 32)]
ax.barh([a for a, _ in arms][::-1], [v / 54 * 100 for _, v in arms][::-1], color=[OURS, OURS, MUTED][::-1])
for i, (_, v) in enumerate(arms[::-1]):
    ax.text(v / 54 * 100 + 1, i, f"{v}/54", va="center", color=INK)
ax.set_xlim(0, 75); ax.set_xlabel("% of pictures recognised in a 1.5-s glance (blind)")
ax.set_title("Pictures: +22 points from the generator (significant)", loc="left", color=INK, fontsize=12)
save(fig, "pictures_round2.png")
print("ok")
