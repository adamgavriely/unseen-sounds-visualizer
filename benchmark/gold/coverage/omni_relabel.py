"""Wrong-dropping R3 (Adam 10 Oct; Fable): Qwen3-Omni closed choice among sibling labels (omni_verify v1, fwd + rev order
averaged) on the burst under each v1.4 picture. When another option beats the picture's label by more than m: relabel
the picture to that option ("none of these" -> drop), or drop it. Scope: weapon pictures only, or all pictures.
m in {0, 0.2, 0.4}. Clip-grouped 5-fold CV (seed 0) on all 158 clips by w=2 and w=4.
    python benchmark/gold/coverage/omni_relabel.py -> omni_relabel.md"""
import itertools, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
WEAPON = ("Gunshot", "Explosion", "Machine gun", "Fireworks", "Artillery")
GRID = [None] + list(itertools.product(("weapon", "all"), ("relabel", "drop"), (0.0, 0.2, 0.4)))


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    import os
    from benchmark.gold.coverage import scene_visible as SV
    r1 = os.environ.get("R1")                        # combine with scene_visible.py's variant (e.g. own_ab)
    base = {st: pictures(SV.apply(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), st5[st]["gate"], r1), float(dur[st] or 10), st) for st in allc}
    bursts, om = {}, {}
    for suf in ("dev", "test"):
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text())["bursts"]:
            bursts.setdefault(b["clip"], []).append(b)
        om.update(json.loads((HERE / f"omni_verify_{suf}.json").read_text())["v1"])
    vote = {}
    for st in allc:
        for lab, a, b in base[st]:
            P, n = {}, 0
            for bu in bursts.get(st, []):
                if not (S.same_family(lab, bu["family"]) and bu["end"] > a - 0.5 and bu["start"] < b + 0.5) or bu["id"] not in om:
                    continue
                n += 1
                for o in ("fwd", "rev"):
                    for opt, p in zip(om[bu["id"]][o]["options"], om[bu["id"]][o]["p"]):
                        P[opt] = P.get(opt, 0.0) + p / 2
            if n:
                P = {k: v / n for k, v in P.items()}
                own = max([v for k, v in P.items() if S.same_family(k, lab)] or [0.0])
                top = max((k for k in P if not S.same_family(k, lab)), key=P.get, default=None)
                vote[(st, lab, a)] = (top, (P[top] if top else 0.0) - own)

    def apply(st, g):
        if g is None:
            return base[st]
        scope, mode, m = g; out = []
        for lab, a, b in base[st]:
            v = vote.get((st, lab, a))
            if v and v[0] and v[1] > m and (scope == "all" or lab.startswith(WEAPON)):
                if mode == "relabel" and v[0] != "none of these":
                    out.append((v[0], a, b))
                continue
            out.append((lab, a, b))
        return out

    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr, w: (4 * sum(x["miss"] for x in rr) + w * wr(rr)) / len(rr)
    L = [f"# Omni closed-choice relabel / drop (R3){' on top of R1 ' + r1 if r1 else ''}, all 158 clips", "", f"pictures with an Omni closed-choice vote: {len(vote)}", "",
         "| scope, mode, m | hits | wrong | cost (w=2) | cost (w=4) |", "|---|---|---|---|---|"]
    for g in GRID:
        rr = [row(st, g) for st in allc]
        L.append(f"| {g or 'v1.4'} | {sum(x['hit'] for x in rr)} | {wr(rr)} | {cost(rr, 2):.3f} | {cost(rr, 4):.3f} |")
    for w in (2, 4):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            best = min(GRID, key=lambda g: cost([row(st, g) for st in tr], w)); ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        L += ["", f"CV by w={w}: choices {ch}; out of fold hits {sum(x['hit'] for x in rr)}, wrong {wr(rr)}, cost w=2 {cost(rr, 2):.3f}, w=4 {cost(rr, 4):.3f}"]
    (HERE / f"omni_relabel{'_' + r1 if r1 else ''}.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
