"""Hold from Qwen3-Omni's per-second "is the sound of {family} heard?" answers (Step 3b (3)) instead of FlexSED
(Adam 10 Oct: keep working). For each v1.4 picture before its hold: the overlapping burst of its family and its
second-by-second P(yes) from the burst start to end + 3 s; the end grows to the last second s whose P(yes) >= tau with
no gap of >= g seconds below tau after the current end (never into the next same-label picture minus 2.5 s, never past
the clip). Modes: Omni only, or Omni OR the v1.4 FlexSED hold. tau in {0.5, 0.7, 0.9}, g in {1, 2};
clip-grouped 5-fold CV on all 158 clips by J (cost_cov + 0.5 x (wrong + stale s) per clip).

    python benchmark/gold/coverage/omni_hold.py   (cluster CPU from ~/wt_slice) -> omni_hold.md
"""
import itertools, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.pooled_hold import hold, J
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
GRID = [None] + list(itertools.product(("omni", "omni|flex"), (0.5, 0.7, 0.9), (1, 2)))


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.HOLD_FLEXSED = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE = str(IN / "flashes.json")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    curves = json.loads((IN / "flexsed_curves.json").read_text())
    allc = stems("dev") + stems("test")
    secs, bursts = {}, {}
    for suf in ("dev", "test"):
        it = json.loads((HERE / f"verify_items_{suf}.json").read_text()); om = json.loads((HERE / f"omni_verify_{suf}.json").read_text())["v3"]
        for b in it["bursts"]:
            bursts.setdefault(b["clip"], []).append(b)
        for s in it["seconds"]:
            if s["id"] in om:
                secs.setdefault(s["burst"], []).append((s["t"], om[s["id"]]))
    base = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}
    flexheld = {st: hold(base[st], curves.get(st, {"labels": {}}), (0.5, None, False, 0.0, "extend"), float(dur[st] or 10)) for st in allc}

    def apply(st, g):
        if g is None:
            return flexheld[st]
        mode, tau, gap = g
        out = []
        for i, (lab, a, b) in enumerate(base[st]):
            nxt = [p[1] for p in base[st] if p[0] == lab and p[1] > a + 1e-9]
            cap = min([float(dur[st] or 10)] + [n - 2.5 for n in nxt])
            ss = sorted(x for bu in bursts.get(st, []) if S.same_family(lab, bu["family"]) and bu["end"] > a - 0.5 and bu["start"] < b + 0.5
                        for x in secs.get(bu["id"], []))
            e, last_ok = b, None
            for t, p in ss:
                if t + 1.0 <= b:
                    continue
                if p >= tau:
                    last_ok = t + 1.0
                elif last_ok is not None and t - last_ok >= gap:
                    break
                elif last_ok is None and t - b >= gap:
                    break
            if last_ok is not None:
                e = max(b, min(cap, last_ok))
            if mode == "omni|flex":
                e = max(e, flexheld[st][i][2])
            out.append((lab, a, e))
        return out

    memo = {}
    row = lambda st, g: memo.setdefault((st, g), V.score_clip_v2(gold[st], apply(st, g)))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: J([row(st, g) for st in tr])[0]); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    def s(rr):
        j, a = J(rr)
        return f"J {j:.3f}, cost_cov {a['cost_cov']:.3f}, cover {a['hit_cov']:.2f}, |end err| {a['end_abs_med']:.2f} s, wrong s/clip {a['wrong_s_per_clip']:.2f}, stale s/clip {a['stale_s_per_clip']:.2f}"
    L = ["# Hold from Qwen3-Omni 'still heard', all 158 clips", "", f"v1.4 (FlexSED hold): {s([row(st, None) for st in allc])}",
         f"CV choices {ch}", f"out of fold: {s([oof[st] for st in allc])}", "", "| mode, tau, gap | J | cost_cov | cover | wrong s | stale s |", "|---|---|---|---|---|---|"]
    for g in GRID[1:]:
        j, a = J([row(st, g) for st in allc])
        L.append(f"| {g} | {j:.3f} | {a['cost_cov']:.3f} | {a['hit_cov']:.2f} | {a['wrong_s_per_clip']:.2f} | {a['stale_s_per_clip']:.2f} |")
    (HERE / "omni_hold.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
