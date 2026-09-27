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

# 1. headline: ours vs blind, TEST (filled) and DEV (hollow), effect +/- CI, Holm survivors marked
dv = json.loads((R / "benchmark/gold/holm_dev_monocap_v31_dev.json").read_text(encoding="utf-8"))
rows = [("Precision", "dP", 1), ("Wrong pictures / clip", "dFA/clip", -1), ("Viewer cost / clip", "d cost/clip", -1),
        ("Clean-clip accuracy", "d clean-acc", 1), ("F0.5", "dF0.5", 1), ("Weighted F1", "dwF1", 1), ("Recall", "dR", 1)]
def series(tab):
    fam = {r["row"]: r for r in tab["family1"]}
    return [(tab["primary"]["d"], tab["primary"]["ci"], False)] + [(fam[k]["d"] * s, sorted(c * s for c in fam[k]["ci"]), fam[k]["survives"]) for _, k, s in rows]
labels = ["F1 (primary)"] + [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for off, tab, name, filled in ((0.17, t, "TEST 60 (final table)", True), (-0.17, dv, "DEV 49", False)):
    for i, (d, ci, ok) in enumerate(series(tab)):
        y = len(labels) - 1 - i + off
        col = GOOD if ok and d > 0 else (BLIND if ok else MUTED)
        ax.plot(ci, [y, y], color=col, lw=3 if filled else 1.6, solid_capstyle="round")
        ax.plot([d], [y], "o", ms=8, color=col, mfc=col if filled else "white", mew=1.8)
    ax.plot([], [], "o", color=INK, mfc=INK if filled else "white", label=name)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1])
ax.set_xlabel("difference, ours − blind (right = ours better; bar = 95% CI; green = survives Holm)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Gate vs “draw every sound”", loc="left", color=INK, fontsize=12)
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
ax.set_title("TEST: cost per clip by kind of clip (lower is better)", loc="left", color=INK, fontsize=12, pad=28)
save(fig, "cost_by_category.png")

# 3. pictures: blind human recognition, round 2 (54 sounds, fresh clips)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
arms = [("today (FLUX)", 14), ("new model (Qwen-Image)", 26), ("new model + new text", 32)]
ax.barh([a for a, _ in arms][::-1], [v / 54 * 100 for _, v in arms][::-1], color=[OURS, OURS, MUTED][::-1])
for i, (_, v) in enumerate(arms[::-1]):
    ax.text(v / 54 * 100 + 1, i, f"{v}/54", va="center", color=INK)
ax.set_xlim(0, 75); ax.set_xlabel("% of pictures recognised in a 1.5-s glance (blind)")
ax.set_title("Pictures: new generator recognised more often (+22 points)", loc="left", color=INK, fontsize=12)
save(fig, "pictures_round2.png")
# 4. gate accuracy on the gold sounds (balanced accuracy; GOLD_RERUN §4d)
fig, ax = plt.subplots(figsize=(6.2, 2.4))
m = [("OWLv2 (object detector)", 0.50), ("Qwen2.5-VL-7B", 0.61), ("Qwen3.8-27B (shipped)", 0.62)]
ax.barh([a for a, _ in m][::-1], [v for _, v in m][::-1], color=[OURS, SIL, SIL])
ax.axvline(0.5, color=BLIND, lw=1.2, ls="--"); ax.text(0.505, 2.35, "chance", color=BLIND, fontsize=9)
for i, (_, v) in enumerate(m[::-1]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK)
ax.set_xlim(0.4, 0.75); ax.set_xlabel("balanced accuracy: is the sound's source on screen? (gold, DEV)")
ax.set_title("Seeing the object ≠ seeing the sound's source", loc="left", color=INK, fontsize=12)
save(fig, "gate_accuracy.png")

# 5. why sounds are missed (DEV, 21 misses, corrected autopsy, amendment 10)
fig, ax = plt.subplots(figsize=(6.4, 2.6))
parts = [("detector never heard it", 11, BLIND), ("picture too late/early", 5, OURS), ("gate: \"visible\"", 3, SIL), ("label filter", 2, MUTED)]
left = 0
for n, v, c in parts:
    ax.barh([0], [v], left=left, color=c); ax.text(left + v / 2, 0, str(v), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    left += v
ax.set_yticks([]); ax.set_xlim(0, 21); ax.set_xticks([]); ax.set_xlabel("21 missed needed sounds, DEV (autopsy of the 22 Sept row, GOLD_RERUN §15)")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], [n for n, _, _ in parts], frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))
ax.set_title("Why sounds are missed: mostly the detector", loc="left", color=INK, fontsize=12, pad=52)
save(fig, "miss_causes.png")
print("ok")
