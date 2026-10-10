"""All 158 clips (Adam 10 Oct: "hold adopted, now refine the ban list and re-check the a/b gate").
Pipeline per gate variant (majority M / a/b AB-m / a/b split->show AB-s): gate -> flash rule -> ban list -> adopted hold
(FlexSED >= 0.5, no tail, extend only). Ban list learned by greedy forward selection over picture labels on the training
clips (add the label that lowers J most; stop when no label lowers it by >= 0.005), scored on held-out clips:
folds DEV->TEST, TEST->DEV, and 5 random clip folds (seed 0). J = cost_cov + 0.5 x (wrong + stale seconds) per clip.
Reference rows: no ban, and the pre-registered texture list. Paired clip bootstrap of the out-of-sample J between gates.

    python benchmark/gold/coverage/pooled_refine.py   (cluster CPU from ~/wt_slice) -> pooled_refine.md
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.step11_policies import policy, TEXTURE
from benchmark.gold.coverage.pooled_hold import hold, J

HERE = Path(__file__).resolve().parent
HOLD = (0.5, None, False, 0.0, "extend")
GATES = ("M", "AB-m", "AB-s")


def main():
    gold = S.load_gold([V.GOLD])
    dev, test = V.stems("dev"), V.stems("test")
    allc = dev + test
    Dc = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]
    Tc = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    dur = {}
    for st in allc:
        ts = [float(m["t"][-1]) for c in ev.get(st, {}).get("labels", {}).values() for m in (c or {}).values() if m and m.get("t")]
        dur[st] = max(ts) if ts else 10.0
    pre = {}
    for g in GATES:
        c = f"SHIP8+MD3+WW5+SL|{g}"
        pre[g] = {st: policy(st, [tuple(p) for p in (Dc if st in dev else Tc)[c]["clips"][st]["pics_none"]], "F", {}, {}, fl)
                  for st in allc}
    cache = {}

    def rows(g, ban, clips):
        key_b = tuple(sorted(ban))
        out = []
        for st in clips:
            k = (g, key_b, st)
            if k not in cache:
                p = [x for x in pre[g][st] if x[0] not in ban]
                cache[k] = V.score_clip_v2(gold[st], hold(p, ev.get(st, {"labels": {}}), HOLD, dur[st]))
            out.append(cache[k])
        return out

    def learn(g, clips):
        labels = sorted({x[0] for st in clips for x in pre[g][st]})
        ban, cur = set(), J(rows(g, set(), clips))[0]
        while True:
            best = None
            for lab in labels:
                if lab in ban:
                    continue
                j = J(rows(g, ban | {lab}, clips))[0]
                if best is None or j < best[0]:
                    best = (j, lab)
            if best is None or cur - best[0] < 0.005:
                return ban
            ban.add(best[1]); cur = best[0]

    rng = random.Random(0)
    sh = allc[:]; rng.shuffle(sh)
    folds = [("DEV->TEST", dev, test), ("TEST->DEV", test, dev)] + \
            [(f"random {k + 1}/5", [c for c in sh if c not in sh[k::5]], sh[k::5]) for k in range(5)]
    L = ["# All 158 clips: learned ban list and gate re-check (flash + adopted hold in every row)", "",
         "J = cost_cov + 0.5 x (wrong + stale s) per clip; all numbers on held-out clips.", ""]
    oos = {}
    for g in GATES:
        L += [f"## gate {g}", "", "| fold | learned ban list | held-out J: none / texture list / learned | hits / wrong (learned) |", "|---|---|---|---|"]
        per_clip = {}
        for name, tr, te in folds:
            ban = learn(g, tr)
            jn = J(rows(g, set(), te))[0]; jt = J(rows(g, TEXTURE, te))[0]; jl, a = J(rows(g, ban, te))
            L.append(f"| {name} | {', '.join(sorted(ban)) or '-'} | {jn:.3f} / {jt:.3f} / {jl:.3f} | {a['hits']} / {a['wrong']} |")
            if name.startswith("random"):
                for st in te:
                    per_clip[st] = ban
        r_none = rows(g, set(), allc); r_tex = rows(g, TEXTURE, allc)
        r_l = [rows(g, per_clip[st], [st])[0] for st in allc]
        oos[g] = r_l
        for nm, rr in (("no ban", r_none), ("texture list", r_tex), ("learned (5-fold out-of-sample)", r_l)):
            j, a = J(rr)
            L.append(f"\n{g} {nm}: J {j:.3f}, hits {a['hits']}/124, wrong {a['wrong']}, onset cost {a['onset_cost']:.3f}, cost_cov {a['cost_cov']:.3f}, "
                     f"cover {a['hit_cov']:.2f}, |end err| {a['end_abs_med']:.2f} s, wrong s/clip {a['wrong_s_per_clip']:.2f}")
        full = learn(g, allc)
        L.append(f"\nBan list learned on all 158 clips (for shipping, not a score): {', '.join(sorted(full)) or '-'}\n")
    # gate comparison on the out-of-sample learned-ban rows
    def jclip(r):
        return J([r])[0]
    rng2 = np.random.default_rng(0)
    L += ["## Gate comparison (out-of-sample, learned ban, paired clip bootstrap of per-clip J)", ""]
    for a_, b_ in (("AB-m", "M"), ("AB-s", "M"), ("AB-m", "AB-s")):
        d = np.array([jclip(x) - jclip(y) for x, y in zip(oos[a_], oos[b_])])
        m = d[rng2.integers(0, len(d), (50000, len(d)))].mean(1)
        p = min(1.0, 2 * min((m >= 0).mean(), (m <= 0).mean()))
        L.append(f"- {a_} - {b_}: {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}], p {p:.2f}")
    (HERE / "pooled_refine.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
