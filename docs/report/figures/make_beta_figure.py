"""Figure: viewer cost as a function of beta (cost of a wrong picture; a miss costs 4).
beta was assumed (2), because the planned DHH viewer study that would measure it was not run.
Counts from Table 'Main result'. Run: python docs/report/figures/make_beta_figure.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = {
    "development set (71 clips)": {"clips": 71, "needed": 58,
        "final system": (29, 15), "direct audio-to-image": (34, 39)},
    "test set (87 clips)": {"clips": 87, "needed": 65,
        "final system": (24, 23), "direct audio-to-image": (26, 61)},
}
COL = {"final system": "#2a6f97", "direct audio-to-image": "#e07a5f", "show nothing": "#777777"}
b = np.linspace(0, 6, 301)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
for ax, (title, d) in zip(axes, SETS.items()):
    n, need = d["clips"], d["needed"]
    nothing = 4 * need / n
    ax.plot(b, np.full_like(b, nothing), color=COL["show nothing"], lw=2, ls="--", label="show nothing")
    for name in ("direct audio-to-image", "final system"):
        hits, wrong = d[name]
        miss = need - hits
        cost = (4 * miss + b * wrong) / n
        ax.plot(b, cost, color=COL[name], lw=2.2, label=name)
        be = (4 * need - 4 * miss) / wrong  # break-even with show nothing
        if be <= 6:
            ax.plot([be], [nothing], "o", color=COL[name], ms=5)
            ax.annotate(f"break-even\n$\\beta$ = {be:.1f}", (be, nothing), textcoords="offset points",
                        xytext=(-6, -30 if name == "final system" else 10), ha="right" if name != "final system" else "center",
                        fontsize=8, color=COL[name])
    ax.axvline(2, color="#bbbbbb", lw=1)
    ax.text(2.05, ax.get_ylim()[1] if False else 0.3, "assumed $\\beta$ = 2", fontsize=8, color="#666666", rotation=90,
            va="bottom")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("$\\beta$ = cost of one wrong picture (a missed sound costs 4)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 6); ax.set_ylim(0, 6.5)
axes[0].set_ylabel("cost per clip (lower is better)", fontsize=9)
h, l = axes[0].get_legend_handles_labels()
order = [l.index("final system"), l.index("direct audio-to-image"), l.index("show nothing")]
leg = axes[0].legend([h[i] for i in order], [l[i] for i in order], frameon=True, framealpha=1, edgecolor="none", fontsize=8.5, loc="upper left")
leg.get_texts()[0].set_fontweight("bold")   # the final system first and bold, as in the other figures
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta_sensitivity.pdf")
fig.savefig(out); print(out)
for title, d in SETS.items():
    n, need = d["clips"], d["needed"]
    for name in ("direct audio-to-image", "final system"):
        hits, wrong = d[name]; miss = need - hits
        print(title, name, "break-even vs nothing at beta =", round(4 * hits / wrong, 2))
