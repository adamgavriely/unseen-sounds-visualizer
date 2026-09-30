"""Round 30 BTP screen (band-twin pull; docs/prereg_round13_detector_push.md "Round 30"), CPU only, on the saved SHIP3+DV
(= SHIP4) proposed pictures of merged DEV. A drawn non-rescued picture whose family has a FlexSED run (frame score >= BAR,
runs merged over LISTEN_RUN_GAP as the pipeline's _runs) ending <= 1.0 s before the picture's start and starting <= 1.5 s
before it gets its start pulled to that run's start (the latest such run). Rescored with score_per_sound; nothing in
src/ or config.py is edited. GO iff 0 current hits lost, hits >= 25, wrong <= 33.

    python benchmark/gold/btp_screen.py [--bar 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from src.labels import canonical
from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP

ARM, SYS = "SHIP3+DV", "proposed"
END_GAP, START_GAP = 1.0, 1.5
FLEX_DIR = DCC.FLEX_DIR                                  # data/work/flexsed_cache (shared by DEV and dev2)
OUT = _ROOT / "benchmark" / "gold" / "btp_screen.json"


def placed(work_root, stem):
    """S.load_pictures, keeping each placed picture's representative spec (for the `rescued` marker) and its chain"""
    f = work_root / stem / "augmentations.json"
    if not f.exists():
        return []
    specs = json.loads(f.read_text(encoding="utf-8"))
    m = work_root / stem / "media.json"
    dur = (float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None) if m.exists() else None
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs, raw = [], {}
    for s in specs:
        o = AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])])
        objs.append(o); raw[id(o)] = s
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    pl, _ = _assign_rows(_display_spans(objs, d, require_image=True))
    return [(lab, float(a), float(b), bool(raw[id(sp)].get("rescued", False))) for _, lab, a, b, sp in pl]


def flex_runs(stem, bar):
    p = FLEX_DIR / f"{stem}.npz"
    if not p.exists():
        return None
    fw, ft, labs = DCC.load_fr(p)
    out = {}
    for c, lab in enumerate(labs):
        rr, dt = _runs(fw[:, c], ft, bar, LISTEN_RUN_GAP)
        out.setdefault(canonical(lab), []).extend((float(ft[i]), float(ft[j - 1] + dt)) for i, j in rr)
    return out


def pull(pics, runs, only_nonrescued=True):
    new, changed = [], []
    for lab, a, b, resc in pics:
        a2 = a
        if runs is not None and not (only_nonrescued and resc):
            ok = [(s, e) for s, e in runs.get(canonical(lab), []) if a - END_GAP <= e <= a and a - START_GAP <= s < a]
            if ok:
                a2 = max(ok, key=lambda r: (r[1], r[0]))[0]          # the latest such run
        new.append((lab, a2, b))
        if a2 != a:
            changed.append((lab, a, a2, resc))
    return new, changed


def parts():
    gold, stems = DCC.dev_stems()
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        dev = [("dev", st, gold[st], placed(R.R13 / f"{ARM}_{SYS}", st)) for st in stems]
    from benchmark.gold import tagger_prep as T
    DCC2, R2, stems2 = T.configure("dev2")
    keep = set(stems2)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out("dev2") / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    g2 = T._REAL_LOAD_GOLD([tmp])
    o = T.out("dev2")
    with R2.flags({k: R2.arm_cfg(ARM)[k] for k in R2.DISPLAY_KEYS}):
        dev2 = [("dev2", st, g2[st], placed(o / f"{ARM}_{SYS}", st)) for st in stems2 if st in g2]
    return dev + dev2


def summ(rows):
    m = DCC.metrics(rows)
    return {"hits": m["hits"], "n": m["hits"] + m["misses"], "wrong": m["wrong"], "visible": m["visible"],
            "cross": m["cross"], "phantom": m["phantom"], "cost": m["viewer_cost"]}


def fmt(x):
    return f"{x['hits']}/{x['n']} wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['cost']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=float, default=0.5)
    a = ap.parse_args()
    P = parts()
    n_resc = sum(r for *_x, pics in P for *_y, r in pics)
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {n_resc} rescued; FlexSED bar {a.bar}")
    res = {"bar": a.bar}
    base_rows = {pt: [] for pt in ("dev", "dev2")}
    for mode in ("nonrescued", "all"):
        rows = {pt: [] for pt in ("dev", "dev2")}; ch = []; lost = []; nofx = 0
        for pt, st, g, pics in P:
            runs = flex_runs(st, a.bar); nofx += runs is None
            new, changed = pull(pics, runs, mode == "nonrescued")
            r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, new)
            if mode == "nonrescued":
                base_rows[pt].append(r0)
            rows[pt].append(r1)
            if changed:
                oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
                ch.append({"part": pt, "clip": st, "changes": [[l, round(x, 2), round(y, 2), rs] for l, x, y, rs in changed],
                           "before": oc(r0), "after": oc(r1)})
            if r1["hit"] < r0["hit"]:
                lost.append((st, r0["hit"] - r1["hit"]))
        if mode == "nonrescued":
            B = {"merged": summ(base_rows["dev"] + base_rows["dev2"]), "dev": summ(base_rows["dev"]), "dev2": summ(base_rows["dev2"])}
            res["base"] = B
            print(f"BASE {ARM}: merged {fmt(B['merged'])} | DEV {fmt(B['dev'])} | DEV2 {fmt(B['dev2'])}")
            print(f"  (clips with no FlexSED cache: {nofx})")
        X = {"merged": summ(rows["dev"] + rows["dev2"]), "dev": summ(rows["dev"]), "dev2": summ(rows["dev2"])}
        go = not lost and X["merged"]["hits"] >= 25 and X["merged"]["wrong"] <= 33
        res[mode] = {"rows": X, "changed": ch, "hits_lost": lost, "GO": go}
        print(f"BTP[{mode}]: merged {fmt(X['merged'])} | DEV {fmt(X['dev'])} | DEV2 {fmt(X['dev2'])}  "
              f"hits lost {lost}  -> {'GO' if go else 'STOP'}")
        for c in ch:
            print(f"   {c['part']:4s} {c['clip']}: {c['changes']}  {c['before']} -> {c['after']}")
    OUT.with_name(f"btp_screen_bar{a.bar}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
