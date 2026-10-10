"""opusC (10 Oct): audio + decision-logic drop rules on top of v1.6, scored on all 158 clips by re-scoring each clip's full
picture list without the dropped pictures (score_per_sound.score_clip), plus clip-grouped 5-fold CV of every tuned
threshold (random.Random(0) shuffle of stems('dev') + stems('test'), folds sh[k::5]; on the 4 training folds the threshold
with the lowest cost is chosen, the lowest of the tied best range (most cautious); applied to the held-out fold).
Needs opusC_feats.json (python benchmark/gold/coverage/opusC_feats.py).   -> opusC_rules.md
    python benchmark/gold/coverage/opusC_rules.py"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import stems, GOLD

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "opusC_feats.json").read_text(encoding="utf-8"))
GOLD_ = S.load_gold([GOLD])
CLIPS = stems("dev") + stems("test")
BY = defaultdict(list)
for i, r in enumerate(R):
    BY[r["clip"]].append(i)

# --- rules: (name, plain text, value(row) -> float or None, drop if value < t, prereg t, grid) ---
seen_any = lambda r: r["gate_seen"] >= 1 or r["gate_desc"] >= 1
RULES = [
    ("GATE_DOUBT_DASM", "the on-screen check said 'seen' (majority or description) in >= 1 stretch, yet the picture is drawn: "
     "DASM's family max over the picture must be >= t", lambda r: r["dasm_inmax"] if seen_any(r) else None, 0.6),
    ("GATE_SEEN_DASM", "as above, majority 'seen' votes only", lambda r: r["dasm_inmax"] if r["gate_seen"] >= 1 else None, 0.6),
    ("FLEX_CLIP_ALL", "FlexSED's family max over the whole clip must be >= t, for every origin (the FlexSED clip veto, 0.3, "
     "already applies to some)", lambda r: r["flex_clipmax"], 0.1),
    ("DASM_CLIP_HARD", "DASM's family max over the whole clip must be >= t, no listener override", lambda r: r["dasm_clipmax"], 0.084),
    ("V1_ABSENT", "Qwen V1 multiple choice p(family) must be >= t when it was asked", lambda r: r.get("v1_p"), 0.05),
    ("CONF_BAR", "stage-5 confidence >= t (display bar is 0.35)", lambda r: r["conf"], 0.37),
    ("AB_CONF", "an a/b override of a majority-'visible' verdict needs stage-5 confidence >= t",
     lambda r: r["conf"] if r["ab_override"] else None, 0.69),
]


def score(drop, clips):
    h = w = m = 0
    for st in clips:
        pics = [(R[i]["label"], R[i]["start"], R[i]["end"]) for i in BY[st] if i not in drop]
        a = S.score_clip(GOLD_[st], pics)
        h += a["hit"]; m += a["miss"]; w += a["visible"] + a["cross"] + a["phantom"]
    return h, w, m, (4 * m + 2 * w) / len(clips)


def drops(fn, t, idx=None):
    return {i for i in (range(len(R)) if idx is None else idx) if fn(R[i]) is not None and fn(R[i]) < t}


def grid(fn):
    v = sorted({fn(r) for r in R if fn(r) is not None})
    return [v[0] - 1e-6] + [(a + b) / 2 for a, b in zip(v, v[1:])] + [v[-1] + 1e-6]


def tune(fn, clips):
    idx = [i for st in clips for i in BY[st]]
    res = [(score(drops(fn, t, idx), clips)[3], t) for t in grid(fn)]
    best = min(c for c, _ in res)
    ts = [t for c, t in res if abs(c - best) < 1e-9]
    return ts[0]          # the most cautious best threshold: just above the last value it must drop


def cv(fns):
    sh = CLIPS[:]; random.Random(0).shuffle(sh)
    drop = set(); chosen = []
    for k in range(5):
        test = sh[k::5]; train = [c for c in sh if c not in test]
        tidx = [i for st in test for i in BY[st]]
        ts = [tune(fn, train) for fn in fns]
        chosen.append(ts)
        for fn, t in zip(fns, ts):
            drop |= drops(fn, t, tidx)
    return score(drop, CLIPS), chosen, drop


def lab(i):
    r = R[i]
    return f"{r['clip']} {r['label']} {r['start']:.1f} ({r['cls']})"


def main():
    base = score(set(), CLIPS)
    L = ["# opusC: audio + decision-logic drop rules on v1.6 (158 clips)", "",
         f"v1.6: {base[0]} hits, {base[1]} wrong, cost {base[3]:.3f}", "",
         "| rule | t (pre-set) | hits | wrong | cost | dropped wrongs | lost hits | CV out-of-fold hits / wrong / cost | CV t per fold |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, text, fn, t0 in RULES:
        d = drops(fn, t0); s = score(d, CLIPS)
        (oh, ow, om, oc), ch, od = cv([fn])
        wr = [lab(i) for i in sorted(d) if R[i]["d_wrong"] > 0]; lh = [lab(i) for i in sorted(d) if R[i]["d_hit"] > 0]
        L.append(f"| {name} | {t0} | {s[0]} | {s[1]} | {s[3]:.3f} | {len(wr)}: {'; '.join(wr)} | {len(lh)}: {'; '.join(lh)} | "
                 f"{oh} / {ow} / {oc:.3f} | {', '.join(f'{c[0]:.3f}' for c in ch)} |")
        print(L[-1])
    L += ["", "Rule text:"] + [f"* {n}: {t}" for n, t, _, _ in RULES]
    for combo in (["GATE_DOUBT_DASM", "FLEX_CLIP_ALL"], ["GATE_DOUBT_DASM", "FLEX_CLIP_ALL", "DASM_CLIP_HARD", "V1_ABSENT"]):
        fs = [x for x in RULES if x[0] in combo]
        d = set().union(*[drops(fn, t0) for _, _, fn, t0 in fs]); s = score(d, CLIPS)
        (oh, ow, om, oc), ch, od = cv([fn for _, _, fn, _ in fs])
        L += ["", f"Stack {' + '.join(combo)}: pre-set t -> {s[0]} hits, {s[1]} wrong, cost {s[3]:.3f}; "
                  f"CV out-of-fold {oh} hits, {ow} wrong, cost {oc:.3f}; CV t per fold {ch}"]
        print(L[-1])
    (HERE / "opusC_rules.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
