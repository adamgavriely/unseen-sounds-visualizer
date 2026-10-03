"""Two result figures of the report: where the gain over direct audio-to-image comes from, and needed sounds found
by importance. Numbers from benchmark/gold/final_vs_baselines.json and final_vs_baselines_extra.json.
Run from the repository root: python docs/report/figures/make_body_figures.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(ROOT, "..", "..", "..", "benchmark", "gold")
DARK, LIGHT, GREY, RED, GREEN = "#2a6f97", "#9ec5dd", "#888888", "#e07a5f", "#5a9e6f"
plt.rcParams.update({"font.size": 9})
B = json.load(open(os.path.join(GOLD, "final_vs_baselines.json"), encoding="utf-8"))
E = json.load(open(os.path.join(GOLD, "final_vs_baselines_extra.json"), encoding="utf-8"))
SETS = (("development", "development set (71 clips)"), ("test", "test set (88 clips)"))

# 1. Win breakdown: direct audio-to-image -> remove its extra wrong pictures -> lose its extra hits -> final system
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
for ax, (key, title) in zip(axes, SETS):
    r = B[key]["rows"]; n = B[key]["clips"]
    a2i, fin, nothing = r["blind_a2i"]["viewer_cost"], r["proposed"]["viewer_cost"], r["show_nothing"]["viewer_cost"]
    removed = r["blind_a2i"]["wrong"] - r["proposed"]["wrong"]
    lost = r["blind_a2i"]["hits"] - r["proposed"]["hits"]
    mid = a2i - 2 * removed / n
    ax.bar(0, a2i, color=GREY, width=0.6)
    ax.bar(1, a2i - mid, bottom=mid, color=GREEN, width=0.6)
    ax.bar(2, fin - mid, bottom=mid, color=RED, width=0.6)
    ax.bar(3, fin, color=DARK, width=0.6)
    ax.plot([0.3, 0.7], [a2i, a2i], color="#bbbbbb", lw=0.8); ax.plot([1.3, 1.7], [mid, mid], color="#bbbbbb", lw=0.8)
    ax.plot([2.3, 2.7], [fin, fin], color="#bbbbbb", lw=0.8)
    ax.text(0, a2i + 0.05, f"{a2i:.3f}", ha="center", fontsize=8)
    ax.text(1, mid - 0.06, f"$-${2 * removed / n:.3f}\n{removed} wrong\nremoved", ha="center", va="top", fontsize=7.5)
    ax.text(2, fin + 0.05, f"+{4 * lost / n:.3f}\n{lost} hits lost", ha="center", fontsize=7.5)
    ax.text(3, fin + 0.05, f"{fin:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(nothing, color=GREY, ls=":", lw=1)
    ax.text(3.45, nothing + 0.04, "show nothing", ha="right", fontsize=7.5, color="#555555")
    ax.set_xticks(range(4)); ax.set_xticklabels(["direct audio-\nto-image", "check removes\nwrong pictures",
                                                 "check loses\nhits", "final\nsystem"], fontsize=7.5)
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 4.0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("cost per clip (lower is better)")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "win_breakdown.pdf")); plt.close(fig); print("win_breakdown.pdf")

# 2. Needed sounds found, by importance
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
for ax, (key, title) in zip(axes, SETS):
    bi = E[key]["by_importance"]
    groups = (("importance_3", "danger / key moment\n(importance 3)"), ("importance_2", "ordinary event\n(importance 2)"))
    for g, (k, lab) in enumerate(groups):
        for j, (s, name, col) in enumerate((("blind_a2i", "direct audio-to-image", GREY), ("proposed", "final system", DARK))):
            found, needed = bi[s][k]
            x = g + (j - 0.5) * 0.36
            ax.bar(x, 100 * found / needed, 0.34, color=col, label=name if g == 0 else None)
            ax.text(x, 100 * found / needed + 2, f"{found}/{needed}", ha="center", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[1] for g in groups], fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("needed sounds found (%)")
axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "importance_bars.pdf")); plt.close(fig); print("importance_bars.pdf")
