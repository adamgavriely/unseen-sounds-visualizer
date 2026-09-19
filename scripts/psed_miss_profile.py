"""PSED misses on slice B at the calibrated bar (0.15), profiled by event length and by what
covers the sound (speech / music / nothing). Answers 'does PSED fail all long sounds or
only some?'. CPU, cached windows."""
import sys, json, collections
sys.path.insert(0, ".")
import numpy as np
from benchmark import audioset_detector_eval as E
from src.labels import is_salient_nonspeech, is_music

BAR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
E.use_set(sys.argv[2] if len(sys.argv) > 2 else "sliceB")
COVER = ("Speech", "Music", "Male speech, man speaking", "Female speech, woman speaking", "Conversation", "Narration, monologue")

def bucket(d):
    return "<1s" if d < 1 else "1-2s" if d < 2 else "2-5s" if d < 5 else ">=5s"

stats = collections.defaultdict(lambda: [0, 0])        # key -> [hit, total]
misses = []
for c in E.clips():
    p = E.WIN / "psed" / f"{c['id']}.npz"
    if not p.exists():
        continue
    fw, t, lab = E._load(p)
    ev = [e for e in E._spans(fw, t, lab, BAR) if is_salient_nonspeech(e.label) and not is_music(e.label)]
    for g in c["events"]:
        if not (g["consequential"] and g["masked"] and is_salient_nonspeech(g["label"]) and not is_music(g["label"])):
            continue
        hit = any(E._same(e.label, g["label"]) and E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev)
        dur = g["end"] - g["start"]
        cov = sorted({x["label"] for x in c["events"] if x["label"] in COVER and min(x["end"], g["end"]) - max(x["start"], g["start"]) > 0})
        covk = "speech+music" if any("peech" in x or "onversation" in x or "arration" in x for x in cov) and "Music" in cov else \
               "music" if "Music" in cov else "speech" if cov else "nothing"
        for k in (("len", bucket(dur)), ("cover", covk), ("all", "all")):
            stats[k][1] += 1; stats[k][0] += hit
        if not hit:
            m = (t >= g["start"]) & (t <= g["end"])
            fam = [i for i, l in enumerate(lab) if E._same(l, g["label"])]
            best = float(fw[m][:, fam].max()) if m.any() and fam else 0.0
            misses.append((g["label"], round(dur, 1), covk, round(best, 2), c["id"]))

print(f"PSED bar {BAR}, masked consequential events on {E.SLICE.name}")
for grp in ("all", "len", "cover"):
    for (g, k), (h, n) in sorted(stats.items()):
        if g == grp:
            print(f"  {grp:6s} {k:13s} recall {h}/{n} = {h/n:.0%}")
print("\nmisses (label, length s, covered by, PSED best score for that family, clip):")
for m in sorted(misses, key=lambda x: -x[1]):
    print("  ", m)
lab_count = collections.Counter(m[0] for m in misses)
unreach = sum(1 for m in misses if m[3] < 0.05)
print(f"ceiling: {unreach}/{len(misses)} misses have PSED family score < 0.05 inside the span on the ORIGINAL (unreachable by any rescue rule on original candidates)")
print("\nmissed labels:", lab_count.most_common())
