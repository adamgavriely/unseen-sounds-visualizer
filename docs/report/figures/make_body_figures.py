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

# 3. Wrong pictures by kind: direct audio-to-image against the final system
KINDS = (("visible", "on screen"), ("cross", "other sound"), ("phantom", "nothing"))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
for ax, (key, title) in zip(axes, SETS):
    r = B[key]["rows"]
    for j, (s, name, col) in enumerate((("blind_a2i", "direct audio-to-image", GREY), ("proposed", "final system", DARK))):
        for g, (k, lab) in enumerate(KINDS):
            x = g + (j - 0.5) * 0.36
            ax.bar(x, r[s][k], 0.34, color=col, label=name if g == 0 else None)
            ax.text(x, r[s][k] + 0.6, str(r[s][k]), ha="center", fontsize=7.5)
    for g, (k, lab) in enumerate(KINDS):
        gone = r["blind_a2i"][k] - r["proposed"][k]
        ax.text(g, max(r["blind_a2i"][k], r["proposed"][k]) + 4.2, f"$-${gone}", ha="center", fontsize=7.5, color=GREEN)
    ax.set_xticks(range(3)); ax.set_xticklabels([k[1] for k in KINDS], fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 40)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("wrong pictures")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "wrong_by_kind.pdf")); plt.close(fig); print("wrong_by_kind.pdf")

# 4. Needed sounds found by the final system, by gold label (labels with at least 2 needed sounds in that set).
# Per-sound outcomes of the final system from docs/decision_trail/data.js; needed = needed and importance >= 2.
# data.js holds no per-sound outcomes of direct audio-to-image, so only the final system is shown.
_s = open(os.path.join(ROOT, "..", "..", "decision_trail", "data.js"), encoding="utf-8").read()
D = json.loads(_s[_s.index("{"):_s.rstrip().rstrip(";").rindex("}") + 1])
SHORT = {"Vehicle horn, car horn, honking": "Vehicle horn", "Gunshot, gunfire": "Gunshot", "Whack, thwack": "Whack",
         "Ice cream truck, ice cream van": "Ice cream truck"}
# The unfound part of each bar is stacked by the step where the sound was lost ("lost_at"), grouped and coloured as in
# the error breakdown figure (LOST_AT_BAR and its colours in make_analysis_figures.py).
LOST_AT_BAR = {
    "never_heard": "never heard",
    "band_rescue": "listeners and their filters", "dasm_vote": "listeners and their filters",
    "dasm_rescue": "listeners and their filters", "rescue_once": "listeners and their filters",
    "scene_margin": "listeners and their filters", "k4a_inventory": "listeners and their filters",
    "dasm_local_veto": "listeners and their filters",
    "gate": "on-screen check",
    "mirror_veto": "vetoes", "masked_weak": "vetoes", "finelap_veto": "vetoes", "continuation_veto": "vetoes",
    "scorer": "timing, length and grouping", "family_merge": "timing, length and grouping",
    "group": "timing, length and grouping", "beats_extract": "timing, length and grouping",
}
LOST = (("never heard", "#555555"), ("listeners and their filters", "#2a6f97"), ("on-screen check", "#e07a5f"),
        ("vetoes", "#9ec5dd"), ("timing, length and grouping", "#c9b18a"))
rows_by_set = {}
for split in ("DEV", "TEST"):
    need, found, lost = {}, {}, {}
    for c in D["clips"]:
        if c["split"] != split:
            continue
        for g in c["gold"]:
            if g["needed"] and g["importance"] >= 2:
                lab = g["label"]
                need[lab] = need.get(lab, 0) + 1
                found[lab] = found.get(lab, 0) + (g["outcome"] == "hit")
                if g["outcome"] != "hit":
                    lost.setdefault(lab, {}).setdefault(LOST_AT_BAR[g["lost_at"]], 0)
                    lost[lab][LOST_AT_BAR[g["lost_at"]]] += 1
    assert sum(need.values()) == (58 if split == "DEV" else 65)
    rows_by_set[split] = sorted(((SHORT.get(k, k), found[k], n, lost.get(k, {})) for k, n in need.items() if n >= 2),
                                key=lambda t: (-t[2], -t[1], t[0]))
nmax = max(len(v) for v in rows_by_set.values())
fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.24 * nmax + 1.5))
for ax, split, (key, title) in zip(axes, ("DEV", "TEST"), SETS):
    rows = rows_by_set[split]
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [f for _, f, _, _ in rows], 0.62, color=GREEN, label="found by the final system")
    left = [f for _, f, _, _ in rows]
    for name, col in LOST:
        w = [l.get(name, 0) for _, _, _, l in rows]
        ax.barh(y, w, 0.62, left=left, color=col, label=f"lost: {name}")
        left = [a + b for a, b in zip(left, w)]
    assert left == [n for _, _, n, _ in rows]
    for yy, (lab, f, n, _) in zip(y, rows):
        ax.text(n + 0.12, yy, f"{f}/{n}", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 6.6); ax.set_xlabel("needed sounds", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(os.path.join(ROOT, "found_by_label.pdf")); plt.close(fig); print("found_by_label.pdf")
