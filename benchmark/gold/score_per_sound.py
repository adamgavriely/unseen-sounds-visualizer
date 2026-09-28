"""Per-sound scoring against the human gold set (docs/metric_per_sound.md, Adam's rules of
19 Sept 2026): every picture a system shows is matched to the gold sound it depicts, then
counted as hit / visible-picture / cross-trigger / phantom / duplicate; every needed sound
as hit / miss. Precision, recall, F1 (strict: visible pictures are false alarms), the
phantom-only F1 for comparison, importance-weighted versions, F0.5 / F2, clip bootstrap CI,
median lateness, clean-clip accuracy. One rule for every system.

    python benchmark/gold/score_per_sound.py --annotations benchmark/gold/annotations/adam.json \
        --tag v4ab --systems proposed blind_a2i audio_caption
    python benchmark/gold/score_per_sound.py --annotations ... --tag v4ab --late 0.5 2 5   # sensitivity

Gold per sound (annotation export): label, family, start, end, obvious, importance.
"needed" = not obvious. System output per clip: data/work/protocol_<system>_<tag>/<clip>/
augmentations.json, turned into on-screen spans by the same rule the compositor uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical, is_descendant, ancestors, is_salient_nonspeech, is_music, FAMILY

EARLY, LATE = 0.5, 1.0          # a picture may start 0.5 s before and at most 1.0 s after the sound (Adam)
# Importance rule (2026-09-22, Adam + two Fables x two rounds, declared before the re-run): importance is a
# property of the sound, not of the screen (1 = steady noise of the place, no start/end; 2 = an event you can
# say in one sentence; 3 = danger or a key story moment). A level-1 NEEDED sound has no onset, so the onset
# rule cannot judge it: it is "don't care" (no hit, no miss; a same-family picture on it is absorbed, not a
# false alarm). A picture of any visible/obvious sound, level 1 included, is a false alarm of weight 1.
# Headline = unweighted F1-strict over needed sounds rated 2-3; weighted F1 (2 -> 2, 3 -> 3, false alarms 1)
# is a side column with n(level 3) printed. OLD_RULE (sensitivity row): level-1 needed sounds count as before.
MIN_IMPORTANCE = 2
OLD_RULE = False
MIN_DEPTH = 1                   # top-level categories ("Sounds of things", "Animal") never match; "Vehicle",
                                # "Water", "Alarm", "Explosion" (depth 1) are real families and do (fix 2026-09-21:
                                # MIN_DEPTH=2 silently turned every Vehicle/Water/Alarm/Glass gold sound into a miss)


# ----------------------------------------------------------------------------- gold
# Annotator free text that no ontology name contains (export of 2026-09-22, 25 rows); each is
# mapped to the AudioSet class a picture of that sound would carry. Declared before any score was
# read; the raw text stays in the gold file.
ALIASES = {
    "machinegun": "Machine gun", "automatic machinegun": "Machine gun", "distant machinegun": "Machine gun",
    "shots": "Gunshot, gunfire", "tank shot": "Artillery fire",
    "phone alert": "Cellphone buzz, vibrating alert", "car turn signal": "Tick-tock",
    "keys jiggle": "Keys jangling", "clank": "Clang", "clank (keychain hits the wall)": "Clang",
    "steps": "Walk, footsteps", "golf swings / ball strikes": "Whack, thwack",
    "cooking": "Frying (food)", "placing the ruler (clack / tap)": "Tap",
    "something moves on the earth making rattling sound": "Rustle", "plastic bags": "Rustle",
    "dollar couting": "Rustle", "vuvuzela": "Air horn, truck horn", "van driving": "Truck",
}


def resolve_label(text: str) -> str:
    """Map an annotator's free-text family/label to an ontology name: exact, then case-insensitive,
    then the longest ontology name contained in the text ("distant explosion / boom" -> Explosion).
    Unresolved names are returned unchanged (they can never match a picture) and counted."""
    names = _ontology_names()
    if text.strip().lower() in ALIASES:
        return ALIASES[text.strip().lower()]
    if text in names:
        return text
    low = text.strip().lower()
    for n in names:
        if n.lower() == low:
            return n
    # a part of an ontology name ("Gunshot" in "Gunshot, gunfire"; "Footsteps" in "Walk, footsteps")
    for n in names:
        if low in [part.strip().lower() for part in n.split(",")]:
            return n
    best = ""
    for n in names:
        nl = n.lower()
        if len(nl) >= 4 and nl in low and len(nl) > len(best):
            best = n
    if best:
        return best
    # singular / plural ("washing machines" -> "Washing machine")
    if low.endswith("s"):
        r = resolve_label(low[:-1])
        if r in names:
            return r
    return text


_NAMES = None


def _ontology_names():
    global _NAMES
    if _NAMES is None:
        from src.labels import _parents
        par = _parents()
        _NAMES = sorted(set(par) | {p for ps in par.values() for p in (ps if isinstance(ps, (list, tuple, set)) else [ps])})
    return _NAMES


UNRESOLVED = []
PLACEHOLDERS = 0


def load_gold(paths):
    """{clip stem: [sound dicts]} from one or more annotation exports (later files win)."""
    gold = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for c in d.get("clips", []):
            if not isinstance(c, dict):     # the tool's export also dumps its UI state (filter values) into the list
                continue
            if c.get("bad"):            # annotator marked the video as unusable: out of the benchmark
                continue
            if not c.get("done"):
                continue
            stem = Path(c["clip"]).stem
            snds = []
            for s in c.get("sounds", []):
                lab = s.get("family") or s.get("label")
                if not lab or s.get("start") is None or s.get("end") is None:
                    continue
                raw = lab; lab = resolve_label(lab)
                if lab not in _ontology_names():
                    UNRESOLVED.append((stem, raw))
                if lab in ("Speech", "Music") or not is_salient_nonspeech(lab) or is_music(lab):
                    continue
                snds.append({"label": lab, "start": float(s["start"]), "end": float(s["end"]),
                             # needed = the source is neither on screen (visible) nor assumed by a viewer with no
                             # sound (obvious): amendment 4 / metric doc. Until 2026-09-22 15:00 only the obvious
                             # tick was read (22 visible-only rows counted as needed) -- corrected before any
                             # TEST number was read (docs/GOLD_RERUN_2026-09-22.md).
                             "needed": not (bool(s.get("obvious", False)) or bool(s.get("visible", False))),
                             "importance": int(s.get("importance") or 2),
                             # kept apart for the per-picture judge's counts (PP-1); nothing here reads them
                             "visible": bool(s.get("visible", False)),
                             "obvious": bool(s.get("obvious", False))})
            gold[stem] = snds
    return gold


# ----------------------------------------------------------------------------- system output
def load_pictures(work_root: Path, stem: str, system: str):
    """on-screen (label, start, end) spans of one system on one clip, as the compositor shows them"""
    f = work_root / stem / "augmentations.json"
    if not f.exists():
        return None
    specs = json.loads(f.read_text(encoding="utf-8"))
    dur = None
    m = work_root / stem / "media.json"
    if m.exists():
        dur = float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None
    # Amendment 3 (2026-09-21, bug fixes): (a) the caption baseline was scored on each tag's
    # strongest burst only while pictures got every burst -- both now go through the same
    # display timeline (require_image=False for captions); (b) pictures the panel never drew
    # (row overflow, _assign_rows) were counted -- only placed spans are scored now; (c) a
    # failed generation (backend "placeholder", a dark panel) still counts as shown, as the
    # judge saw it; its count is reported by main().
    global PLACEHOLDERS
    PLACEHOLDERS += sum(1 for s in specs if s.get("augment") and s.get("backend") == "placeholder")
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = []
    for s in specs:
        o = AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])])
        objs.append(o)
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    spans = _display_spans(objs, d, require_image=(system != "audio_caption"))
    if system == "audio_caption":                      # text tags have no row limit
        return [(lab, float(a), float(b)) for lab, a, b, _ in spans]
    placed, _ = _assign_rows(spans)
    return [(lab, float(a), float(b)) for _, lab, a, b, _ in placed]


# ----------------------------------------------------------------------------- matching
def depth(label: str) -> int:
    return len(ancestors(label))


_FAMILY_NAMES = set(FAMILY.values())


def _specific(x: str) -> bool:
    """deep enough in the ontology to name one source -- or a hand-written family name
    (amendment 3: six FAMILY targets such as "Footsteps", "Gunshot", "Boat" are not ontology
    names, so they had depth 0 and never matched anything)"""
    return depth(x) >= MIN_DEPTH or x in _FAMILY_NAMES


def same_family(a: str, b: str) -> bool:
    if not (_specific(a) and _specific(b)):
        return False
    return a == b or canonical(a) == canonical(b) or is_descendant(a, b) or is_descendant(b, a)


def in_window(pic_start: float, onset: float, early: float, late: float) -> bool:
    return onset - early <= pic_start <= onset + late


def score_clip(gold, pics, early=EARLY, late=LATE):
    """classes for one clip: returns dict of counts and per-hit lateness"""
    gold = sorted(gold, key=lambda g: g["start"]); pics = sorted(pics, key=lambda p: p[1])
    taken = [False] * len(gold)
    out = {"hit": 0, "miss": 0, "visible": 0, "cross": 0, "phantom": 0, "dup": 0, "dontcare": 0, "n3": 0, "collision": 0,
           "w_hit": 0, "w_miss": 0, "w_fa": 0, "late": [], "cov": []}
    scored = lambda g: OLD_RULE or g["importance"] >= MIN_IMPORTANCE     # level-1 needed sounds: don't care
    matched_needed = set()
    for lab, a, b in pics:
        # candidates: gold sounds of the same family whose onset window contains the picture start
        cands = [i for i, g in enumerate(gold) if same_family(lab, g["label"]) and in_window(a, g["start"], early, late)]
        if cands:
            free = [i for i in cands if not taken[i]]
            if not free:
                out["dup"] += 1; continue               # the sound already has its picture
            i = min(free, key=lambda i: abs(a - gold[i]["start"]))
            taken[i] = True
            # one picture may cover several overlapping sounds of the SAME family (their onsets
            # all sit in the picture's window): every covered needed sound is a hit
            # (amendment 3: they were marked taken but counted as misses)
            covered = [i] + [j for j in cands if not taken[j] and same_family(gold[j]["label"], gold[i]["label"])]
            hit_here = False
            for j in covered:
                taken[j] = True
                g = gold[j]
                if g["needed"] and scored(g):
                    out["hit"] += 1; out["w_hit"] += g["importance"]; out["late"].append(a - g["start"]); matched_needed.add(j); hit_here = True
                elif g["needed"]:
                    out["dontcare"] += 1; matched_needed.add(j)      # absorbed: a true picture of a texture, neither credit nor cost
            # same-family collision (a dog on screen and another dog off screen): the picture is a hit, not
            # also a false alarm (three reviewers, 2026-09-22); counted so the flip can be a sensitivity row
            if any(not gold[j]["needed"] for j in covered):
                if hit_here:
                    out["collision"] += 1
                else:
                    out["visible"] += 1; out["w_fa"] += 1             # a false alarm costs 1 whatever the sound
        else:
            # a real sound of another family at that moment -> cross-trigger; nothing -> phantom
            any_sound = any(in_window(a, g["start"], early, late) or (g["start"] <= a <= g["end"]) for g in gold)
            out["cross" if any_sound else "phantom"] += 1; out["w_fa"] += 1
    for i, g in enumerate(gold):
        if g["needed"] and scored(g) and g["importance"] >= 3:
            out["n3"] += 1
        if g["needed"] and scored(g) and i not in matched_needed:
            out["miss"] += 1; out["w_miss"] += g["importance"]
    # coverage (secondary, no new annotation): share of each needed sound's seconds that had a
    # same-family picture up; misses count 0. Onset stays the hit criterion (docs/metric_per_sound.md).
    for i, g in enumerate(gold):
        if not g["needed"] or not scored(g) or g["end"] <= g["start"]:
            continue
        if i not in matched_needed:
            out["cov"].append(0.0); continue           # a missed sound counts 0 (declared rule)
        ivs = sorted((max(a, g["start"]), min(b, g["end"])) for lab, a, b in pics if same_family(lab, g["label"]) and b > g["start"] and a < g["end"])
        cov = 0.0; cur = None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: cov += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur: cov += cur[1] - cur[0]
        out["cov"].append(cov / (g["end"] - g["start"]))
    out["needed"] = sum(1 for g in gold if g["needed"])
    out["clean_ok"] = (out["needed"] == 0 and not pics)
    out["clean_n"] = int(out["needed"] == 0)
    return out


def prf(tp, fp, fn, beta=1.0):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0
    return p, r, f


def aggregate(rows):
    H = sum(r["hit"] for r in rows); M = sum(r["miss"] for r in rows)
    V = sum(r["visible"] for r in rows); C = sum(r["cross"] for r in rows); PH = sum(r["phantom"] for r in rows); D = sum(r["dup"] for r in rows)
    DC = sum(r.get("dontcare", 0) for r in rows); N3 = sum(r.get("n3", 0) for r in rows); CO = sum(r.get("collision", 0) for r in rows)
    fa_strict = V + C + PH; fa_ph = C + PH
    p, r, f1 = prf(H, fa_strict, M)
    _, _, f05 = prf(H, fa_strict, M, 0.5); _, _, f2 = prf(H, fa_strict, M, 2.0)
    pp, rp, fp_ = prf(H, fa_ph, M)
    WH = sum(r["w_hit"] for r in rows); WM = sum(r["w_miss"] for r in rows); WF = sum(r["w_fa"] for r in rows)
    wp, wr, wf = prf(WH, WF, WM)
    late = [x for r in rows for x in r["late"]]
    cov = [x for r in rows for x in r["cov"]]
    clean_n = sum(r["clean_n"] for r in rows); clean_ok = sum(r["clean_ok"] for r in rows)
    fa_clip = float(np.mean([r["visible"] + r["cross"] + r["phantom"] for r in rows])) if rows else 0.0
    return {"fa_per_clip": fa_clip, "viewer_cost": viewer_cost(rows), "hits": H, "misses": M, "visible": V, "cross": C, "phantom": PH, "dup": D, "needed": H + M, "dontcare": DC, "n_level3": N3, "collisions": CO,
            "P": p, "R": r, "F1": f1, "F0.5": f05, "F2": f2, "P_phantom": pp, "F1_phantom": fp_,
            "wP": wp, "wR": wr, "wF1": wf, "median_late": float(np.median(late)) if late else None,
            "coverage": float(np.mean(cov)) if cov else None, "coverage_hits": float(np.mean([c for c in cov if c > 0])) if any(c > 0 for c in cov) else None,  # cov>0 only for hits
            "clean_acc": clean_ok / clean_n if clean_n else None, "clips": len(rows)}


# Viewer cost (Adam, 2026-09-22, approved after seeing that equal-weight F1 cannot separate the
# systems). It is NOT a new weighting invented for this result: the two numbers are the ones this
# project declared in September for the gate sweep (benchmark/gate_dev_sweep.py, COST_MISS = 4,
# COST_REDUNDANT = 2), read off the judging rubric -- a needed picture withheld scores 0 where it
# could have scored 4, and a picture shown where none is due is capped at 2 where silence scores 4.
# cost = 4 x (needed sounds rated >= 2 with no picture) + 2 x (pictures that are false alarms);
# lower is better. Reported per clip, with the break-even weight printed beside it so the reader can
# see the whole trade-off instead of one chosen point.
COST_MISS, COST_FA = 4.0, 2.0


def viewer_cost(rows, w_fa=None):
    w = COST_FA if w_fa is None else float(w_fa)
    n = max(1, len(rows))
    return sum(COST_MISS * r["miss"] + w * (r["visible"] + r["cross"] + r["phantom"]) for r in rows) / n


def fa_per_clip(rows):
    return float(np.mean([r["visible"] + r["cross"] + r["phantom"] for r in rows]))


def boot_ci(rows, key="F1", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [aggregate([rows[i] for i in rng.integers(0, len(rows), len(rows))])[key] for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_ci(rows_a, rows_b, key="F1", n=2000, seed=0):
    """(difference, lo, hi, P(d>0)) of any aggregate key; raises TypeError when the key is None
    for these rows (clean_acc on a subset with no clean clip)"""
    """clip bootstrap of the DIFFERENCE a - b: the same resampled clips for both systems
    (amendment 5, docs/prereg_v4.md); rows_a and rows_b are aligned lists (same clips, same order)"""
    rng = np.random.default_rng(seed)
    m = len(rows_a)
    if aggregate(rows_a)[key] is None or aggregate(rows_b)[key] is None:
        raise TypeError(key + " is undefined for these clips")
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        a = aggregate([rows_a[i] for i in idx])[key]; b = aggregate([rows_b[i] for i in idx])[key]
        diffs.append((a or 0.0) - (b or 0.0))
    d = aggregate(rows_a)[key] - aggregate(rows_b)[key]
    return d, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(np.asarray(diffs) > 0))


def category(sounds) -> str:
    """the tool's clip category from the ticks (benchmark/gold/tool_template.html category())"""
    if not sounds:
        return "no_ambient"
    needed = [s for s in sounds if s["needed"]]
    seen = [s for s in sounds if not s["needed"]]
    if not needed:
        return "seen"
    if any(s["importance"] >= 2 for s in needed) and any(s["importance"] >= 2 for s in seen):
        return "mixed"
    return "unseen"


def subsets_of(gold):
    """named clip subsets declared in amendment 5: population, dev/test, category, pre-screened"""
    from benchmark.gold.detector_dry import clip_path, JUDGE100
    judge = set(JUDGE100.read_text().split()) if JUDGE100.exists() else set()
    slice_file = _ROOT / "benchmark" / "gold" / "audioset_slice.json"
    ids = {c["id"] for c in json.loads(slice_file.read_text(encoding="utf-8"))["clips"]} if slice_file.exists() else set()
    sliceb = {st for st in gold if st in ids}                       # slice B = the AudioSet-Strong gold clips
    subs = {"all": set(gold), "bench": set(gold) - sliceb, "sliceB": sliceb,
            "dev": set(gold) & judge, "test": set(gold) - judge,
            "test_bench": (set(gold) - judge) - sliceb,
            "prescreened": {st for st in gold if st.startswith(("m5_", "t1_", "w8_"))}}
    for cat in ("mixed", "unseen", "seen", "no_ambient"):
        subs["cat_" + cat] = {st for st in gold if category(gold[st]) == cat}
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--systems", nargs="+", default=["proposed", "blind_a2i", "audio_caption", "silence"])
    ap.add_argument("--late", nargs="+", type=float, default=[LATE])
    ap.add_argument("--early", type=float, default=EARLY)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None, help="data/work root holding protocol_<system>_<tag> (default: the repo's)")
    ap.add_argument("--subsets", nargs="+", default=["bench", "all", "sliceB", "dev", "test", "test_bench", "prescreened", "cat_mixed", "cat_unseen", "cat_seen", "cat_no_ambient"])
    ap.add_argument("--old-rule", action="store_true", help="sensitivity row: level-1 needed sounds scored as before (hits/misses)")
    a = ap.parse_args()
    global OLD_RULE
    OLD_RULE = bool(a.old_rule)
    gold = load_gold(a.annotations)
    print(f"[gold] {len(gold)} annotated clips, {sum(len(v) for v in gold.values())} sounds, "
          f"{sum(1 for v in gold.values() for s in v if s['needed'])} needed")
    if UNRESOLVED:
        print(f"[gold] {len(UNRESOLVED)} sound names not in the ontology (can never match): " + "; ".join(f"{c}: {r}" for c, r in UNRESOLVED))
    work = Path(a.work) if a.work else _ROOT / "data" / "work"
    subs = subsets_of(gold)
    results = {}
    for late in a.late:
        # rows per system, keyed by clip stem (silence = no pictures, needs no render)
        per = {}
        for system in a.systems:
            root = work / f"protocol_{system}_{a.tag}"
            rows = {}
            for stem, snds in gold.items():
                pics = [] if system == "silence" else load_pictures(root, stem, system)
                if pics is None:
                    continue
                rows[stem] = score_clip(snds, pics, a.early, late)
            if not rows:
                print(f"[{system}] no rendered clips under {root}"); continue
            per[system] = rows
        for sub in a.subsets:
            if not any(st in subs.get(sub, set()) for rows in per.values() for st in rows):
                continue
            print(f"--- subset {sub}: {len(subs[sub])} clips in the gold (late <= {late:.1f} s); each system on the clips it has rendered")
            for system, rows in per.items():
                stems = sorted(st for st in subs[sub] if st in rows)
                if not stems:
                    continue
                rs = [rows[st] for st in stems]
                agg = aggregate(rs); lo, hi = boot_ci(rs)
                agg["F1_ci"] = [lo, hi]
                results[f"{sub}|{system}@late{late}"] = agg
                print(f"[{system:13s}] clips {len(stems):3d} needed {agg['needed']:3d} | P {agg['P']:.2f} R {agg['R']:.2f} F1 {agg['F1']:.2f} [{lo:.2f},{hi:.2f}] F0.5 {agg['F0.5']:.2f} F2 {agg['F2']:.2f} | "
                      f"F1-ph {agg['F1_phantom']:.2f} | wF1 {agg['wF1']:.2f} (n3={agg['n_level3']}) dc {agg['dontcare']} coll {agg['collisions']} | cov {agg['coverage'] if agg['coverage'] is None else round(agg['coverage'], 2)} | "
                      f"hit {agg['hits']} miss {agg['misses']} vis {agg['visible']} cross {agg['cross']} ph {agg['phantom']} dup {agg['dup']} | "
                      f"late-med {agg['median_late'] if agg['median_late'] is None else round(agg['median_late'], 2)} | clean {agg['clean_acc'] if agg['clean_acc'] is None else round(agg['clean_acc'], 2)}")
            if "proposed" in per:
                for other in [s for s in per if s != "proposed"]:
                    both = sorted(st for st in subs[sub] if st in per["proposed"] and st in per[other])
                    if not both:
                        continue
                    row = {"clips": len(both)}
                    for key, fmt in (("F1", "dF1"), ("F0.5", "dF0.5"), ("wF1", "dwF1"), ("P", "dP"), ("R", "dR"), ("fa_per_clip", "dFA/clip"), ("viewer_cost", "d cost/clip"), ("clean_acc", "d clean-acc")):
                        try:
                            d, lo, hi, pgt = paired_ci([per["proposed"][st] for st in both], [per[other][st] for st in both], key=key)
                        except TypeError:
                            continue
                        row[fmt] = {"d": d, "ci": [lo, hi], "p_gt0": pgt}
                        print(f"   {fmt:11s} proposed - {other:13s} = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(d>0)={pgt:.3f}  (paired on {len(both)} clips)")
                    results[f"{sub}|delta_proposed-{other}@late{late}"] = row
    if PLACEHOLDERS:
        print(f"[pictures] {PLACEHOLDERS} augmentation(s) were placeholder panels (failed generation); counted as shown")
    out = Path(a.out) if a.out else _ROOT / "benchmark" / "gold" / f"per_sound_{a.tag}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
