"""Wrong-dropping R2 (Adam 10 Oct; Fable): repeat pictures. A v1.4 picture is dropped when an earlier picture of the same
group started less than g seconds before it in the same clip. Groups: the label, the canonical family, or Fable's coarse
groups (weapon = Gunshot / Explosion / Machine gun / Fireworks / Artillery fire; siren = Siren / Shofar / Alarm / Ambulance /
Police car / Fire engine / Civil defense siren; voice = Baby cry / Screaming / Crying / Laughter / Shout / Children shouting).
g in {3, 6, 10, inf}. Clip-grouped 5-fold CV (seed 0) on all 158 clips by w=2 and w=4.
    python benchmark/gold/coverage/family_once.py -> family_once.md"""
import itertools, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
from src.labels import canonical

HERE = Path(__file__).resolve().parent
COARSE = {"weapon": ("Gunshot", "Explosion", "Machine gun", "Fireworks", "Artillery"), "siren": ("Siren", "Shofar", "Alarm", "Ambulance", "Police car", "Fire engine", "Civil defense"),
          "voice": ("Baby cry", "Screaming", "Crying", "Laughter", "Shout", "Children shouting")}
GRID = [None] + list(itertools.product(("label", "family", "coarse"), (3.0, 6.0, 10.0, 1e9)))


def key(lab, how):
    if how == "label":
        return lab
    if how == "family":
        return canonical(lab)
    for k, ws in COARSE.items():
        if any(lab.startswith(w) for w in ws):
            return k
    return canonical(lab)


def apply(pics, g):
    if g is None:
        return pics
    how, gap = g; out, last = [], {}
    for p in sorted(pics, key=lambda p: p[1]):
        k = key(p[0], how)
        if k in last and p[1] - last[k] < gap:
            continue
        last[k] = p[1]; out.append(p)
    return out


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    base = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(base[st], g)))
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr, w: (4 * sum(x["miss"] for x in rr) + w * wr(rr)) / len(rr)
    L = ["# One picture per group per gap (R2), all 158 clips", "", "| group, gap s | hits | wrong | cost (w=2) | cost (w=4) |", "|---|---|---|---|---|"]
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
    (HERE / "family_once.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
