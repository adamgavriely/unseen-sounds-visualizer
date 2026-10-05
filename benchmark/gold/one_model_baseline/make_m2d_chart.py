"""DEV-only figure: PretrainedSED M2D-strong at every bar of its DEV sweep (sounds caught vs wrong pictures), with the
final system and direct audio-to-image on the same DEV clips.  python make_m2d_chart.py -> fig_m2d_dev_sweep.pdf/.png"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "result_m2d.json").read_text(encoding="utf-8"))
g = r["dev_grid"]; dev = r["development"]["rows"]
bars = sorted(g, key=float)
x = [g[b]["wrong"] for b in bars]; y = [g[b]["hits"] for b in bars]
n = dev["proposed"]["hits"] + dev["proposed"]["misses"]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(x, y, "-o", color="#4C78A8", ms=4, lw=1.5, label="M2D-strong, one bar per point")
for b, xi, yi in zip(bars, x, y):
    if float(b) in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        ax.annotate(f"{float(b):g}", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#4C78A8")
ax.scatter([dev["proposed"]["wrong"]], [dev["proposed"]["hits"]], s=60, color="#E45756", zorder=3, label="final system")
ax.scatter([dev["blind_a2i"]["wrong"]], [dev["blind_a2i"]["hits"]], s=45, marker="s", color="#72B7B2", zorder=3,
           label="direct audio-to-image")
ax.set_xscale("log")
ax.set_xlabel("wrong pictures on DEV (log scale)")
ax.set_ylabel(f"sounds caught (of {n})")
ax.set_ylim(0, n)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_m2d_dev_sweep.{ext}", dpi=200)
print("ok")
