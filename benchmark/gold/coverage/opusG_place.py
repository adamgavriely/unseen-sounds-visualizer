"""opusG: change WHAT or WHEN we draw (not whether), all 158 clips, on the v1.7 pictures (screen_reason.base_pics()).

Ideas, each a generic rule on one video's own pictures + the drawn family's detector curves (detector_curves.json):
  A  back-extend: a picture whose family is already sounding just before its start (curve >= bar in the last 0.5 s) gets
     its start moved back to where the family was first heard: walk back over the curve while it stays >= bar, allowing
     dips below the bar of at most `gap` s, at most N s, never before t=0 or the end of an earlier picture of the family.
  S  split: a picture is split at a fresh rise of its family curve inside it (curve below bar for >= `quiet` s, then
     >= bar, at least `lead` s after the picture start): the later part becomes its own picture (its own onset).
  L  label: weapon-family pictures (Gunshot) drawn as their ontology parent (Explosion); Siren drawn as Alarm.
  M  merging: MERGE_GAP / GROUP_ASK / GROUP_MAX_GAP sweep (pictures rebuilt).
Thresholds chosen by clip-grouped 5-fold CV (random.Random(0) shuffle of stems('dev') + stems('test'), sh[k::5]) on
onset cost (4 x misses + 2 x wrongs) / clips; the grid's first entry is "no change" and wins ties.
    python benchmark/gold/coverage/opusG_place.py   -> opusG_place.md"""
import json, random, sys, copy
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
from benchmark.gold.coverage.screen_reason import base_pics
from src.stage6_visual_augmentation import v16

HERE = Path(__file__).resolve().parent
gold = S.load_gold([GOLD])
PICS = base_pics()
ALL = stems("dev") + stems("test")
FOCUS = {("b3_golf_course", "Bird"), ("tg_d031", "Bird"), ("w8_hide_wolves_howl_1a", "Baby cry, infant cry"),
         ("w8_hide_wolves_howl_1a", "Human locomotion")}


def curve(stem, label, det):
    st = v16._offline(stem) or {}
    if det == "fd":                                      # mean of FlexSED and DASM on the DASM grid
        f, d = curve(stem, label, "flex"), curve(stem, label, "dasm")
        if f is None or d is None:
            return f or d
        return d[0], (d[1] + np.interp(d[0], f[0], f[1])) / 2
    return v16.curves(label, stem, None, st).get(det)


def back_start(stem, pics, i, det, bar, gap, N, rmax=None, found=False):
    lab, a, b = pics[i]
    c = curve(stem, lab, det)
    if c is None:
        return a
    t, v = c
    if rmax is not None:                                 # only a picture that does NOT start on a rise of its family
        r = v16.onset(c, a)[1]
        if r is None or r > rmax:
            return a
    pre = (t >= a - 0.5) & (t < a)
    if not pre.any() or v[pre].max() < bar:
        return a
    floor = max([0.0, a - N] + [q[2] for q in pics if q[0] == lab and q[1] < a and q is not pics[i]])
    idx = np.where(t < a)[0][::-1]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    earliest, below, why = a, 0.0, "start"            # why the walk ended: quiet stretch / limit (N or earlier picture) / clip start
    for k in idx:
        if t[k] < floor:
            why = "limit" if floor > 0.0 else "start"
            break
        if v[k] >= bar:
            earliest, below = float(t[k]), 0.0
        else:
            below += dt
            if below > gap:
                why = "quiet"
                break
    if found and why == "limit":                     # "found": the family's first sound was actually reached
        return a
    return earliest if earliest < a - 0.25 else a


def rule_A(stem, g):
    pics = list(PICS[stem])
    if g is None:
        return pics
    det, bar, gap, N, rmax, found = g
    return [(p[0], back_start(stem, pics, i, det, bar, gap, N, rmax, found), p[2]) for i, p in enumerate(pics)]


def rule_S(stem, g):
    pics = list(PICS[stem])
    if g is None:
        return pics
    det, bar, quiet, lead = g
    out = []
    for lab, a, b in pics:
        c = curve(stem, lab, det)
        cuts = []
        if c is not None:
            t, v = c
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
            below = 0.0
            for k in np.where((t >= a) & (t < b))[0]:
                if v[k] >= bar:
                    if below >= quiet and t[k] >= a + lead and (not cuts or t[k] >= cuts[-1] + lead):
                        cuts.append(float(t[k]))
                    below = 0.0
                else:
                    below += dt
        edges = [a] + cuts + [b]
        out += [(lab, edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
    return out


PARENT = {"Gunshot!": "Gunshot, gunfire", "Gunshot": "Explosion", "Gunshot, gunfire": "Explosion", "Machine gun": "Explosion", "Fireworks": "Explosion",
          "Siren": "Alarm"}


def rule_L(stem, g):
    pics = list(PICS[stem])
    if g is None:
        return pics
    keys = {"ontology name": ("Gunshot!",), "weapon": ("Gunshot", "Gunshot, gunfire", "Machine gun", "Fireworks"), "siren": ("Siren",), "both": tuple(PARENT)}[g]
    if g == "ontology name":                             # the family name "Gunshot" (no ontology node) -> its ontology name
        return [("Gunshot, gunfire" if p[0] == "Gunshot" else p[0], p[1], p[2]) for p in pics]
    return [(PARENT[p[0]] if p[0] in keys else p[0], p[1], p[2]) for p in pics]


memo = {}
def row(rule, stem, g):
    k = (rule.__name__, stem, g)
    if k not in memo:
        memo[k] = S.score_clip(gold[stem], rule(stem, g))
    return memo[k]

hits = lambda rr: sum(x["hit"] for x in rr)
wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)


def cv(rule, grid):
    rng = random.Random(0); sh = ALL[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in ALL if c not in te]
        best = min(grid, key=lambda g: cost([row(rule, st, g) for st in tr]))
        ch.append(best)
        for st in te:
            oof[st] = row(rule, st, best)
    rr = [oof[st] for st in ALL]
    return ch, rr


def delta(rule, g):
    """per-clip changes vs v1.7: (clip, dhit, dwrong)"""
    out = []
    for st in ALL:
        r0, r1 = row(rule, st, None), row(rule, st, g)
        dh, dw = r1["hit"] - r0["hit"], wr([r1]) - wr([r0])
        if dh or dw:
            out.append(f"{st} hit{dh:+d} wrong{dw:+d}")
    return out


def second_burst():
    """misses whose family already has a picture that STARTED earlier in the clip (and none in the onset window)"""
    L, n, n_in = [], 0, 0
    for st in ALL:
        ps = PICS[st]
        for gsd in gold[st]:
            if not gsd["needed"] or gsd["importance"] < 2:
                continue
            if any(S.same_family(p[0], gsd["label"]) and S.in_window(p[1], gsd["start"], S.EARLY, S.LATE) for p in ps):
                continue                                 # hit (or its picture taken by a twin)
            fam = [p for p in ps if S.same_family(p[0], gsd["label"]) and p[1] < gsd["start"] - S.EARLY]
            if fam:
                n += 1
                cover = any(p[1] < gsd["start"] <= p[2] for p in fam)
                n_in += cover
                L.append(f"| {st} | {gsd['label']} | {gsd['start']:.1f}-{gsd['end']:.1f} | "
                         + ", ".join(f"{p[0]} {p[1]:.1f}-{p[2]:.1f}" for p in fam) + f" | {'yes' if cover else 'no'} |")
    late = 0                                              # misses with a same-family picture starting > 1 s after onset, inside the sound
    for st in ALL:
        for gsd in gold[st]:
            if gsd["needed"] and gsd["importance"] >= 2 and not any(S.same_family(p[0], gsd["label"]) and S.in_window(p[1], gsd["start"], S.EARLY, S.LATE) for p in PICS[st]):
                late += any(S.same_family(p[0], gsd["label"]) and gsd["start"] + S.LATE < p[1] <= gsd["end"] for p in PICS[st])
    return n, n_in, late, L


def merge_rows(settings):
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    res = []
    for name, kv in settings:
        config.use_shipped(); config.MAX_AFTER_END = None
        config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
        config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
        config.ONSET_CURVES = str(IN / "detector_curves.json")
        for k, v in kv.items():
            setattr(config, k, v)
        rows = {st: S.score_clip(gold[st], pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st)) for st in ALL}
        res.append((name, kv, rows))
    return res


def main():
    r0 = [row(rule_A, st, None) for st in ALL]
    L = ["# opusG: where / what / how merged we draw, v1.7 pictures, all 158 clips", "",
         f"v1.7: hits {hits(r0)} / {sum(x['needed'] for x in r0)} needed-listed, wrong {wr(r0)}, cost {cost(r0):.3f}", ""]

    # ---- A back-extend
    gA = [None] + [(d, b, gp, N, rmax, found) for d, bars in (("flex", (0.3, 0.5)), ("dasm", (0.2, 0.3, 0.4, 0.5)), ("beats", (0.1, 0.15, 0.2)), ("fd", (0.3, 0.4, 0.5)))
                   for b in bars for gp in (0.25, 0.5, 1.0, 1.5, 2.0) for N in (1.0, 2.0, 3.0, 5.0, 30.0)
                   for rmax in (None, 0.0, 0.1, 0.2, 0.3) for found in (False, True)]
    L += ["## A  back-extend the start to where the family was first heard", ""]
    full = sorted(gA[1:], key=lambda g: cost([row(rule_A, st, g) for st in ALL]))
    L += ["Best 10 settings on all clips (reading only; CV below is the result):", "", "| det | bar | gap | N | rise<= | found | hits | wrong | cost | focus fixed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in full[:10]:
        rr = [row(rule_A, st, g) for st in ALL]
        fx = []
        for st, lab in FOCUS:
            a1 = [p for p in rule_A(st, g) if p[0] == lab]; a0 = [p for p in PICS[st] if p[0] == lab]
            if [p[1] for p in a1] != [p[1] for p in a0]:
                fx.append(f"{st.split('_')[0]}:{lab.split(',')[0]} {a0[0][1]:.1f}->{a1[0][1]:.1f}")
        L.append(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {g[4]} | {g[5]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} | {'; '.join(fx)} |")
    ch, rr = cv(rule_A, gA)
    L += ["", f"CV: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}"]
    for g in sorted(set(c for c in ch if c)):
        L.append(f"- {g}: " + "; ".join(delta(rule_A, g)))
    # the four focus cases under every setting: how many fixed at best
    L += ["", "Focus cases (picture start vs gold) under the most frequent CV choice:"]
    gbest = max(set(ch), key=ch.count)
    for st, lab in sorted(FOCUS):
        a0 = [p for p in PICS[st] if p[0] == lab]; a1 = [p for p in rule_A(st, gbest) if p[0] == lab]
        r0c, r1c = row(rule_A, st, None), row(rule_A, st, gbest)
        L.append(f"- {st} {lab}: {[round(p[1], 2) for p in a0]} -> {[round(p[1], 2) for p in a1]}; clip hit {r0c['hit']}->{r1c['hit']}, wrong {wr([r0c])}->{wr([r1c])}")
    nfix = {}
    for g in gA[1:]:
        k = 0
        for st, lab in FOCUS:
            a0 = [p for p in PICS[st] if p[0] == lab][0]; a1 = [p for p in rule_A(st, g) if p[0] == lab][0]
            gl = [x for x in gold[st] if S.same_family(lab, x["label"]) and x["start"] < a0[1]]
            k += any(S.in_window(a1[1], x["start"], S.EARLY, S.LATE) for x in gl)
        nfix[g] = k
    mx = max(nfix.values())
    bestfix = min([g for g in nfix if nfix[g] == mx], key=lambda g: cost([row(rule_A, st, g) for st in ALL]))
    rr2 = [row(rule_A, st, bestfix) for st in ALL]
    L += ["", f"Most focus cases any setting puts in the onset window: {mx} of 4 ({sum(1 for v in nfix.values() if v == mx)} settings); "
          f"cheapest such {bestfix}: hits {hits(rr2)}, wrong {wr(rr2)}, cost {cost(rr2):.3f}", "- " + "; ".join(delta(rule_A, bestfix))]

    # ---- second-burst misses
    n, n_in, late, SL = second_burst()
    nm = sum(x["miss"] for x in r0)
    L += ["", "## Second-burst misses", "", f"{nm} misses. {n} are sounds of a family already drawn by a picture that started earlier in the clip "
          f"(and no picture of the family in the onset window); {n_in} of them start while that earlier picture is still up "
          f"(merged / held over the new onset). {late} misses have a same-family picture starting > 1 s late inside the sound.", "",
          "| clip | gold | onset-end | earlier pictures of the family | still up at onset |", "|---|---|---|---|---|"] + SL

    # ---- S split
    gS = [None] + [(d, b, q, ld) for d, bars in (("flex", (0.3, 0.5)), ("dasm", (0.3, 0.4, 0.5)), ("beats", (0.15, 0.2)), ("fd", (0.4, 0.5)))
                   for b in bars for q in (0.5, 1.0, 2.0) for ld in (1.0, 2.0, 3.0)]
    full = sorted(gS[1:], key=lambda g: cost([row(rule_S, st, g) for st in ALL]))
    L += ["", "## S  split a picture at a fresh rise inside it", "", "| det | bar | quiet | lead | hits | wrong | cost |", "|---|---|---|---|---|---|---|"]
    for g in full[:6]:
        rr = [row(rule_S, st, g) for st in ALL]
        L.append(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    ch, rr = cv(rule_S, gS)
    L += ["", f"CV: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}"]
    for g in sorted(set(c for c in ch if c)):
        L.append(f"- {g}: " + "; ".join(delta(rule_S, g)))

    # ---- L label
    L += ["", "## L  parent label for disputed siblings", "",
          "Scorer: same_family(picture, gold) is true for ancestor/descendant pairs, so an 'Explosion' picture is a hit for a "
          "Gunshot, gunfire / Machine gun / Fireworks / Burst gold sound; the family name 'Gunshot' is not an ontology name (no ancestors), "
          "so a 'Gunshot' picture never matches Explosion or Fireworks gold. Fire is not under Explosion (Natural sounds > Fire). Siren is a child of Alarm; Shofar is under Music.", ""]
    cnt = {}
    for st in ALL:
        for p in PICS[st]:
            if p[0] in PARENT or p[0] in ("Explosion", "Alarm", "Fire"):
                cnt[p[0]] = cnt.get(p[0], 0) + 1
    L.append(f"pictures of these labels in v1.7: {cnt}")
    for g in ("ontology name", "weapon", "siren", "both"):
        rr = [row(rule_L, st, g) for st in ALL]
        L.append(f"- {g}: hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}; " + "; ".join(delta(rule_L, g)))

    # ---- M merging
    sets = [("v1.7 (MERGE_GAP 2.5, GROUP 8)", {})] + [(f"MERGE_GAP {m}", {"MERGE_GAP": m}) for m in (0.8, 1.0, 1.5, 2.0, 3.0, 4.0)] + \
           [("GROUP off", {"GROUP_ASK": False}), ("GROUP_MAX_GAP 4", {"GROUP_MAX_GAP": 4.0}), ("MERGE_GAP 1.0 + GROUP off", {"MERGE_GAP": 1.0, "GROUP_ASK": False})]
    res = merge_rows(sets)
    L += ["", "## M  merging", "", "| setting | hits | wrong | dup | cost |", "|---|---|---|---|---|"]
    for name, kv, rows in res:
        rr = [rows[st] for st in ALL]
        L.append(f"| {name} | {hits(rr)} | {wr(rr)} | {sum(x['dup'] for x in rr)} | {cost(rr):.3f} |")
    gm = list(range(len(res)))
    rng = random.Random(0); sh = ALL[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in ALL if c not in te]
        best = min(gm, key=lambda i: cost([res[i][2][st] for st in tr]))
        ch.append(res[best][0])
        for st in te:
            oof[st] = res[best][2][st]
    rr = [oof[st] for st in ALL]
    L += ["", f"CV: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}"]
    (HERE / "opusG_place.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
