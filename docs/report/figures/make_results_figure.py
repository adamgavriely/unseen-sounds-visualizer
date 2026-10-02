"""Figure 3 of the report: hits and wrong pictures per system, development and test set.
Numbers from Table 3 of the report. Run: python docs/report/figures/make_results_figure.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {  # system: (hits, wrong, cost) per set
    "development set (71 clips, 58 needed sounds)": {"needed": 58, "rows": [
        ("show\nnothing", 0, 0, 3.268), ("direct\naudio-to-image", 34, 39, 2.451), ("final\nsystem", 29, 15, 2.056)]},
    "test set (88 clips, 65 needed sounds)": {"needed": 65, "rows": [
        ("show\nnothing", 0, 0, 2.955), ("direct\naudio-to-image", 26, 62, 3.182), ("final\nsystem", 24, 24, 2.409)]},
}
HIT, WRONG, INK = "#2a6f97", "#e07a5f", "#333333"
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
for ax, (title, d) in zip(axes, DATA.items()):
    xs = range(len(d["rows"]))
    w = 0.36
    for i, (name, hits, wrong, cost) in enumerate(d["rows"]):
        ax.bar(i - w / 2, hits, w, color=HIT, label="hits (right picture)" if i == 0 else None)
        ax.bar(i + w / 2, wrong, w, color=WRONG, label="wrong pictures" if i == 0 else None)
        top = max(hits, wrong)
        ax.text(i, min(top + 2.5, d["needed"] - 7), f"cost {cost:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                fontweight="bold" if name.startswith("final") else "normal")
    ax.axhline(d["needed"], color="#888888", lw=1, ls="--")
    ax.text(len(d["rows"]) - 0.5, d["needed"] + 1, f"needed sounds: {d['needed']}", ha="right", va="bottom",
            fontsize=8, color="#666666")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in d["rows"]], fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 72)
axes[0].set_ylabel("count")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_bars.pdf")
fig.savefig(out); print(out)
