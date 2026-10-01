"""Round 49 BANDLIST (docs/prereg_round13_detector_push.md "Round 49 BANDLIST"), merged DEV: weak FlexSED runs (band
0.5 <= max < 0.8, length >= 0.5 s, depictable family; formed exactly as bandlist_415.py forms them) whose family the
Qwen3-Omni whole-clip list names (expect_a/listen, expect_a_screen.items_of / map_item, all items) AND whose family DASM is
>= 0.575 within run onset +- 0.5 s (expect_a4_screen.dasm_max; no column -> not kept). A run whose family already has a SHIP8
picture (canonical or same_family) starting within +- 2 s is excluded. The shipped stage-5 gate at onset - 1 ... + 1 s
(expect_screen.cmd_gate, as Round 40e) decides; an unseen candidate becomes a 2-s picture (label = the family) added as an
AugmentationSpec to SHIP8's specs, then the display timeline runs as shipped (MERGE_GAP 2.5, GROUP with the bench answers,
clip = stem, exactly as score_per_sound.load_pictures). Base (no additions) must be 28/58, 18 (6/10/2), cost 2.197.

    TG_ARMS=SHIP8 python benchmark/gold/bandlist_screen.py cands   # CPU -> bandlist/cands.json
    TG_ARMS=SHIP8 python benchmark/gold/bandlist_screen.py gate    # GPU: Qwen3.8-27B -> bandlist/gate/
    TG_ARMS=SHIP8 python benchmark/gold/bandlist_screen.py score   # CPU -> bandlist_screen.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import expect_screen as E  # noqa: E402

DIR = _ROOT / "benchmark" / "gold" / "bandlist"
OUT = _ROOT / "benchmark" / "gold" / "bandlist_screen.json"
E.DIR = DIR
LO, HI, MIN_LEN, HALF, NEAR, PIC_LEN = 0.5, 0.8, 0.5, 0.5, 2.0, 2.0
BASE = {"hits": 28, "wrong": 18, "visible": 6, "cross": 10, "phantom": 2, "cost": 2.197}


def setup():
    """-> (parts, {part: SHIP8 work root}, display flags). Roots are taken before parts() (tagger configure moves R.R13)."""
    from benchmark.gold import round13_dev as R
    roots = {"dev": R.R13 / f"{E.B_ARM}_proposed"}
    disp = {k: R.arm_cfg(E.B_ARM)[k] for k in R.DISPLAY_KEYS}
    P = E.parts()
    from benchmark.gold import tagger_prep as T                  # imported only after parts() has read the gold
    roots["dev2"] = T.out("dev2") / f"{E.B_ARM}_proposed"
    return P, roots, disp


def pictures(root, stem, extra=()):
    """score_per_sound.load_pictures (clip = stem, so GROUP applies) with extra (family, start, end, conf) specs appended"""
    from benchmark.gold import score_per_sound as S
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    f = root / stem / "augmentations.json"
    specs = json.loads(f.read_text(encoding="utf-8"))
    m = root / stem / "media.json"
    dur = (float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None) if m.exists() else None
    objs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])]) for s in specs]
    n = len(objs)
    for i, (fam, a, b, c) in enumerate(extra):
        objs.append(AugmentationSpec(index=n + i, event_label=fam, start=float(a), end=float(b), augment=True, confidence=float(c),
                                     image_path="bandlist_added", talked_about=False, spans=[(float(a), float(b))], breaks=[]))
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    pl, _ = _assign_rows(_display_spans(objs, d, require_image=True, clip=stem))
    if not extra:
        assert [(l, float(a), float(b)) for _, l, a, b, _ in pl] == S.load_pictures(root, stem, "proposed"), stem
    return [(lab, float(a), float(b)) for _, lab, a, b, _ in pl]


def cmd_cands():
    from benchmark.gold import expect_a_screen as A
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import cross_group as CG
    from benchmark.gold import round13_dev as R
    from benchmark.gold.bandlist_415 import fams_of
    from benchmark.gold.score_per_sound import same_family
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    assert A4.BAR == 0.575
    P, roots, disp = setup()
    dep = set(E.FAMILIES)
    tally = {"band_runs": 0, "min_len": 0, "omni_named": 0, "dasm_ok": 0, "no_dasm_column": 0, "near_ship8": 0,
             "dup": 0, "candidate": 0}
    rows = []
    with R.flags(disp):
        for pt, st, g, _pics in P:
            ship = pictures(roots[pt], st)
            L = json.loads((DIR.parent / "expect_a" / "listen" / f"{st}.json").read_text(encoding="utf-8"))
            omni = fams_of(L["text"], A)
            fw, t, labs = DCC.load_fr(DCC.FLEX_DIR / f"{st}.npz")
            dp = CG.PARTS[pt]["dasm"] / f"{st}.npz"
            dfr = DCC.load_fr(dp) if dp.exists() else None
            seen = set()
            for e in _extract_events(fw, t, labs, LO, None, 0.0, low=LO):
                f = canonical(e.label)
                if f not in dep or e.confidence >= HI:
                    continue
                a, b = float(e.start), float(e.end)
                tally["band_runs"] += 1
                row = {"part": pt, "clip": st, "family": f, "label": e.label, "onset": round(a, 2), "end": round(b, 2),
                       "max": round(float(e.confidence), 3)}
                if b - a < MIN_LEN - 1e-9:
                    continue
                tally["min_len"] += 1
                if f not in omni:
                    continue
                tally["omni_named"] += 1
                v = A4.dasm_max(dfr, f, a - HALF, a + HALF)
                row["dasm_onset"] = None if v is None else round(v, 3)
                if v is None:
                    tally["no_dasm_column"] += 1; row["outcome"] = "no DASM column"; rows.append(row); continue
                if v < A4.BAR:
                    row["outcome"] = "DASM below bar"; rows.append(row); continue
                tally["dasm_ok"] += 1
                near = [(l, round(pa, 2)) for l, pa, pb in ship if (canonical(l) == f or same_family(l, f)) and abs(pa - a) <= NEAR + 1e-9]
                if near:
                    tally["near_ship8"] += 1; row["outcome"] = "SHIP8 same-family picture within 2 s"; row["near"] = near
                    rows.append(row); continue
                k = (pt, st, f, round(a, 2))
                if k in seen:
                    tally["dup"] += 1; continue
                seen.add(k)
                tally["candidate"] += 1; row["outcome"] = "candidate"; rows.append(row)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "rows": rows}, indent=1, default=float), encoding="utf-8")
    print(tally)
    for r in rows:
        if r["outcome"] == "candidate":
            print(f"   {r['part']:4s} {r['clip']} {r['family']} ({r['label']}) {r['onset']}-{r['end']} max {r['max']} dasm {r['dasm_onset']}")


def cmd_score():
    from benchmark.gold import btp_screen as B
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import round13_dev as R
    from benchmark.gold.cross_group import classify
    P, roots, disp = setup()
    C = json.loads((DIR / "cands.json").read_text(encoding="utf-8"))
    gates = {}
    for f in (DIR / "gate").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8")); gates[E.gate_key(r)] = r
    base = {"dev": [], "dev2": []}; rows = {"dev": [], "dev2": []}
    added, dropped, lost, unasked = [], [], [], 0
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    with R.flags(disp):
        for pt, st, g, _pics in P:
            old = pictures(roots[pt], st)
            extra, adds = [], []
            for r in C["rows"]:
                if r["part"] != pt or r["clip"] != st or r["outcome"] != "candidate":
                    continue
                gt = gates.get(E.gate_key(r))
                if gt is None:
                    unasked += 1; continue
                if gt["seen"]:
                    dropped.append({"part": pt, "clip": st, "family": r["family"], "onset": r["onset"],
                                    "votes": [gt.get(k) for k in ("name", "ab", "desc")]})
                    continue
                extra.append((r["family"], r["onset"], r["onset"] + PIC_LEN, r["max"])); adds.append((r, gt))
            new = pictures(roots[pt], st, extra) if extra else old
            r0 = S.score_clip(g, old); r1 = S.score_clip(g, new)
            base[pt].append(r0); rows[pt].append(r1)
            if adds:
                cl = classify(g, new)
                for r, gt in adds:
                    # the displayed picture carrying this family whose span covers the onset (merged / grouped spans keep the earlier start)
                    hit = [(l, a, b, k) for l, a, b, k, _ in cl if l == r["family"] and a - 1e-6 <= r["onset"] <= b + 1e-6]
                    added.append({"part": pt, "clip": st, "family": r["family"], "start": r["onset"], "run_end": r["end"],
                                  "max": r["max"], "dasm": r["dasm_onset"], "votes": [gt.get(k) for k in ("name", "ab", "desc")],
                                  "displayed_as": [(l, round(a, 2), round(b, 2)) for l, a, b, k in hit],
                                  "class": ("not displayed" if not hit else hit[0][3] if abs(hit[0][1] - r["onset"]) < 1e-6
                                            else f"merged into existing (start {hit[0][1]:.2f}, {hit[0][3]})"),
                                  "pics_before": len(old), "pics_after": len(new),
                                  "clip_before": oc(r0), "clip_after": oc(r1)})
            if r1["hit"] < r0["hit"]:
                lost.append((pt, st, r0["hit"] - r1["hit"]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    n = len(P)
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    bm = Bm["merged"]
    assert {k: bm[k] for k in ("hits", "wrong", "visible", "cross", "phantom")} == {k: BASE[k] for k in ("hits", "wrong", "visible", "cross", "phantom")}, bm
    assert bm["n"] == 58 and abs(bm["cost"] - BASE["cost"]) < 0.0005, bm
    c0 = {w: G.cost_w(bm, n, w) for w in (1, 2)}; c1 = {w: G.cost_w(X["merged"], n, w) for w in (1, 2)}
    assert abs(c0[2] - bm["cost"]) < 1e-9, (c0, bm)
    n_lost = sum(x for _, _, x in lost)
    main_go = X["merged"]["hits"] >= BASE["hits"] and not lost and c1[2] < c0[2]
    fewer_go = c1[2] < c0[2] and X["merged"]["wrong"] <= BASE["wrong"] - 3 * n_lost and n_lost <= 3
    gain = X["merged"]["hits"] - BASE["hits"]
    fa_ok = all(X[pt][k] <= Bm[pt][k] for pt in ("dev", "dev2") for k in ("cross", "phantom"))
    more_go = gain > 0 and fa_ok and c1[1] < c0[1]
    verdict = "GO (main rule)" if main_go else ("GO (fewer-pictures clause)" if fewer_go else "STOP")
    print(f"SHIP8+BANDLIST: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  added {len(added)}  "
          f"gate-dropped {len(dropped)}  unasked {unasked}  hits lost {lost}")
    print(f"  cost w=2 {c0[2]:.3f} -> {c1[2]:.3f}, w=1 {c0[1]:.3f} -> {c1[1]:.3f}   main {main_go}  fewer-pictures {fewer_go}  "
          f"more-hits (secondary) {more_go}  -> {verdict}")
    print("  funnel", C["tally"])
    for d in added:
        print(f"   {d['part']:4s} {d['clip']} {d['family']} +{d['start']} (max {d['max']}, dasm {d['dasm']}, votes {d['votes']}): "
              f"{d['class']} shown {d['displayed_as']}  pics {d['pics_before']}->{d['pics_after']}  clip {d['clip_before']} -> {d['clip_after']}")
    for d in dropped:
        print(f"   {d['part']:4s} {d['clip']} {d['family']} @{d['onset']} gate seen {d['votes']} -> dropped")
    res = {"what": "Round 49 BANDLIST, merged DEV", "base": Bm, "rows": X, "added": added, "gate_dropped": dropped,
           "hits_lost": lost, "unasked": unasked, "funnel": C["tally"],
           "cost": {str(w): {"base": round(c0[w], 3), "new": round(c1[w], 3)} for w in (1, 2)},
           "main_GO": main_go, "fewer_pictures_GO": fewer_go, "more_hits_PASS_secondary": more_go, "verdict": verdict}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"cands": cmd_cands, "gate": E.cmd_gate, "score": cmd_score}[sys.argv[1]]()
