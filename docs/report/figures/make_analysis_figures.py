"""Analysis figures of the report (ceiling waterfall, gate trade-off, development progression, error breakdown,
listener agreement). Numbers are copied from the files named next to each block.
Run from the repository root: python docs/report/figures/make_analysis_figures.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
DARK, LIGHT, GREY, RED = "#2a6f97", "#9ec5dd", "#888888", "#e07a5f"
plt.rcParams.update({"font.size": 9})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name)); plt.close(fig); print(name)


# 1. Ceiling waterfall -- benchmark/gold/ceiling_ship7.md (older version, development set, 55 needed sounds)
steps = [("version\nas run", 2.451, 25), ("+ perfect\ntiming", 2.197, 26), ("+ perfect\nfamily", 2.141, 26),
         ("+ timing and\nfamily jointly", 2.056, 27), ("+ perfect\non-screen check", 1.380, 32),
         ("+ perfect\nvetoes", 1.155, 36), ("+ perfect\nlisteners", 0.761, 43)]
fig, ax = plt.subplots(figsize=(8.6, 3.2))
prev = None
for i, (lab, c, h) in enumerate(steps):
    if prev is None:
        ax.bar(i, c, color=GREY, width=0.6)
    else:
        ax.bar(i, prev - c, bottom=c, color=RED if "on-screen" in lab else LIGHT, width=0.6)
        ax.plot([i - 1.3, i + 0.3], [prev, prev], color="#bbbbbb", lw=0.8)
    ax.text(i, (c if prev is None else prev) + 0.06, f"{c:.2f}\n{h}/55 hits", ha="center", va="bottom", fontsize=7.5)
    prev = c
ax.bar(len(steps), steps[-1][1], color=DARK, width=0.6)
ax.text(len(steps), steps[-1][1] + 0.06, "floor\n12 missed", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(steps) + 1))
ax.set_xticklabels([s[0] for s in steps] + ["what no rule\ncan recover"], fontsize=7.5)
ax.set_ylabel("cost per clip"); ax.set_ylim(0, 3.1)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "ceiling_waterfall.pdf")

# 2. Gate trade-off -- docs/history/review/improvements_log_2026-09-30.md and the round-by-round record
# (development stretches labelled by the author: 41 on-screen sounds, 38 needed sounds)
pts = [("final system", 16, 33), ("synchrony veto", 11, 36), ("synchrony add", 19, 33), ("synchrony both", 14, 36),
       ("annotator's questions", 15, 32), ("human-style chain", 0, 38), ("human-style, alone", 5, 38),
       ("human-style, 4th vote", 18, 32), ("box and crop", 9, 35), ("box and crop v2", 13, 35),
       ("both must agree", 15, 34), ("name-all + crop", 21, 33)]
OFFS = {"synchrony veto": (-4, 4), "synchrony both": (4, 4), "synchrony add": (4, 6), "name-all + crop": (4, -9),
        "annotator's questions": (-4, -9), "human-style, 4th vote": (4, -9), "final system": (-5, -10),
        "box and crop": (-4, 4), "box and crop v2": (4, 4), "both must agree": (4, 3), "human-style chain": (4, 4),
        "human-style, alone": (4, -9)}
fig, ax = plt.subplots(figsize=(6.2, 4.0))
xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
import numpy as np
xx = np.linspace(-1, 23, 50)
ax.plot(xx, 33 - (xx - 16) / 2, color=GREY, ls="--", lw=1)
ax.text(1.2, 40.0, "same cost as the final system\n(2 more silenced = 1 more lost)", fontsize=7.5, color=GREY)
ax.fill_between([15, 23], 35, 39, color="#d8f0d8", alpha=0.7, lw=0)
ax.fill_between([19, 23], 32, 35, color="#d8f0d8", alpha=0.7, lw=0)
ax.text(19.2, 37.6, "pre-set\npass region", fontsize=7.5, color="#3a7a3a")
for name, x, y in pts:
    final = name == "final system"
    ax.plot(x, y, "o", color=DARK if final else RED, ms=7 if final else 5)
    off = OFFS.get(name, (4, 3))
    ax.annotate(name, (x, y), textcoords="offset points", xytext=off, fontsize=7, color=DARK if final else "#555555",
                ha="right" if off[0] < 0 else "left")
ax.set_xlabel("on-screen sounds correctly silenced (of 41)")
ax.set_ylabel("needed sounds correctly kept (of 38)")
ax.set_xlim(-1, 23); ax.set_ylim(30, 41)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "gate_tradeoff.pdf")

# 3. Development progression -- benchmark/gold/visible_weight_sweep.md, improvements log, history appendix
chain = [("Sept.\nbaseline", 3.690, 2.909), ("listener\nrescue", 3.070, 2.818), ("masked\nveto", 2.958, 2.750),
         ("DASM\nrescue", 2.901, 2.705), ("inventory\nkeep", 2.817, 2.727), ("DASM clip\nveto", 2.620, 2.682),
         ("onset\npull", 2.535, 2.659), ("continuation\nveto", 2.394, 2.591), ("FineLAP\nveto", 2.366, 2.568),
         ("inventory\n+ DASM", 2.282, 2.568), ("merge\n2.5 s", 2.254, 2.545), ("grouping", 2.197, 2.545),
         ("min. 0.3 s", 2.141, 2.545), ("witness rule\n= final", 2.056, 2.409)]
# The witness rule was first scored with its scene question read as text (2.028 / 2.386). That reading was cut off on a
# few answers, and one cut-off answer happened to remove a wrong picture. The final system reads the same question
# from the model's yes/no scores, which cannot be cut off; its honest numbers are the last point. The text-read
# point is not a real step and is not plotted (see the history appendix).
fig, ax = plt.subplots(figsize=(9.2, 4.0))
x = range(len(chain))
ax.plot(x, [c[1] for c in chain], "o-", color=DARK, label="development set (decisions made here)")
ax.plot(x, [c[2] for c in chain], "s-", color=RED, label="test set (read after each change)")
ax.axhline(3.268, color=DARK, ls=":", lw=1); ax.axhline(2.955, color=RED, ls=":", lw=1)
ax.text(len(chain) - 0.6, 3.29, "show nothing (dev.)", ha="right", fontsize=7.5, color=DARK)
ax.text(len(chain) - 0.6, 2.975, "show nothing (test)", ha="right", fontsize=7.5, color=RED)
ax.set_xticks(list(x)); ax.set_xticklabels([c[0].replace("\n", " ") for c in chain], fontsize=8.5, rotation=35, ha="right")
ax.set_ylabel("cost per clip"); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "progression.pdf")

# 4. Error breakdown of the final system -- docs/inspector2/data.js (lost_at of each miss; verdict of each picture)
# Each logged step where a needed sound was lost (its "lost_at" value) is put in one bar:
LOST_AT_BAR = {
    "never_heard": "never heard",
    # listeners and the filters on their answers
    "band_rescue": "listeners and their filters", "dasm_vote": "listeners and their filters",
    "dasm_rescue": "listeners and their filters", "rescue_once": "listeners and their filters",
    "scene_margin": "listeners and their filters", "k4a_inventory": "listeners and their filters",
    "dasm_local_veto": "listeners and their filters",
    "gate": "on-screen check",
    # vetoes: another audio model does not hear the candidate
    "mirror_veto": "vetoes", "masked_weak": "vetoes", "finelap_veto": "vetoes", "continuation_veto": "vetoes",
    # timing, length and grouping: picture outside the onset window, too short, or merged into a neighbour
    "scorer": "timing, length and grouping", "family_merge": "timing, length and grouping",
    "group": "timing, length and grouping", "beats_extract": "timing, length and grouping",
}
BARS = ["never heard", "listeners and their filters", "on-screen check", "vetoes", "timing, length and grouping"]


def miss_counts(path=os.path.join(OUT, "..", "..", "inspector2", "data.js")):
    """Recount the misses per bar from the decision trail (falls back to the stored counts if it is absent)."""
    stored = {"DEV": [6, 10, 5, 3, 5], "TEST": [6, 12, 3, 10, 10]}
    if not os.path.exists(path):
        return {sp: dict(zip(BARS, v)) for sp, v in stored.items()}
    import json
    s = open(path, encoding="utf-8").read()
    d = json.loads(s[s.index("{"):s.rstrip().rstrip(";").rindex("}") + 1])
    out = {sp: dict.fromkeys(BARS, 0) for sp in ("DEV", "TEST")}

    def walk(x, sp):
        if isinstance(x, dict):
            if x.get("lost_at"):
                out[sp][LOST_AT_BAR[x["lost_at"]]] += 1
            for v in x.values():
                walk(v, sp)
        elif isinstance(x, list):
            for v in x:
                walk(v, sp)
    for c in d["clips"]:
        walk(c["gold"], c["split"])
    for sp in out:
        assert list(out[sp].values()) == stored[sp], (sp, out[sp])
    return out


miss = miss_counts()
wrong = {"DEV": {"other sound": 7, "on screen": 6, "nothing": 2}, "TEST": {"other sound": 15, "on screen": 4, "nothing": 5}}
cols_m = ["#555555", "#2a6f97", "#e07a5f", "#9ec5dd", "#c9b18a"]
cols_w = ["#e07a5f", "#9ec5dd", "#555555"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6))
for ax, data, cols, title in ((axes[0], miss, cols_m, "missed needed sounds, by where they were lost"),
                              (axes[1], wrong, cols_w, "wrong pictures, by kind")):
    for row, sp in enumerate(("TEST", "DEV")):
        left = 0
        for (k, v), c in zip(data[sp].items(), cols):
            ax.barh(row, v, left=left, color=c, label=k if row == 0 else None, height=0.55)
            if v >= 2:
                ax.text(left + v / 2, row, str(v), ha="center", va="center", fontsize=7.5, color="white")
            left += v
    ax.set_yticks([0, 1]); ax.set_yticklabels(["test", "development"])
    ax.set_title(title, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
save(fig, "error_breakdown.pdf")

# 5. Listener agreement -- improvements log (held-out AudioSet-Strong screening set, 415 clips)
bars = [("BEATs detections", 0.38), ("FlexSED detections", 0.39), ("named by no listener", 0.22),
        ("named by one listener", 0.37), ("named by both listeners", 0.62), ("listener + timing + DASM chain", 0.80)]
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.barh(range(len(bars))[::-1], [b[1] for b in bars], color=[GREY, GREY, LIGHT, LIGHT, DARK, DARK], height=0.6)
for i, (lab, v) in enumerate(bars):
    ax.text(v + 0.01, len(bars) - 1 - i, f"{v:.0%}", va="center", fontsize=8)
ax.set_yticks(range(len(bars))[::-1]); ax.set_yticklabels([b[0] for b in bars], fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("share of detections that are real sounds (precision)")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "listener_agreement.pdf")
