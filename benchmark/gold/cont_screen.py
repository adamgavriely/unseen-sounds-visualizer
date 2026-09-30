"""Round 31 CONT screen (continuation veto; docs/prereg_round13_detector_push.md "Round 31 CONT"), CPU only, on the saved
SHIP5 (= SHIP4+BTP) proposed pictures of merged DEV. A placed picture is dropped when a FlexSED run of its family (frame
score >= BAR 0.5, gaps <= LISTEN_RUN_GAP merged, the pipeline's _runs) starts >= WIN 1.5 s before the picture's start and
reaches it (run end >= picture start); pictures starting in the clip's first WIN seconds are kept. Rescored with
score_per_sound; nothing in src/ or config.py is edited. Pass = the combined rule vs SHIP5 (25/55, 33, 2.620): old rule
(hits >= 25, wrong <= 33 + 2 x gain, cost < 2.620, no needed hit lost on either part) OR fewer-pictures clause (cost < 2.620,
wrong <= 33 - 3 x hits lost, hits >= 22). Reported only: BEATs column as the evidence, and WIN 1.0.

    python benchmark/gold/cont_screen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import btp_screen as B
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical
from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP

B.ARM = "SHIP4+BTP"
BAR, WIN = 0.5, 1.5
WORK = _ROOT / "data" / "work"
BEATS = {"dev": WORK / "j2_dev_beats", "dev2": WORK / "j2_dev2_beats"}
BASE = {"hits": 25, "wrong": 33, "cost": 2.620}
OUT = _ROOT / "benchmark" / "gold" / "cont_screen.json"


def runs_of(fr, bar):
    if fr is None:
        return None
    fw, ft, labs = fr
    out = {}
    for c, lab in enumerate(labs):
        rr, dt = _runs(fw[:, c], ft, bar, LISTEN_RUN_GAP)
        out.setdefault(canonical(lab), []).extend((float(ft[i]), float(ft[j - 1] + dt)) for i, j in rr)
    return out


def continuation(lab, a, runs, win):
    if a < win or runs is None:
        return False
    return any(s <= a - win and e >= a for s, e in runs.get(canonical(lab), []))


def passes(X, lost_parts):
    gain = X["hits"] - BASE["hits"]; lost = BASE["hits"] - X["hits"]
    old = X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"] and not lost_parts
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * max(lost, 0) and X["hits"] >= BASE["hits"] - 3
    return old, few


_ORIG_PLACED = B.placed


def placed_rows(work_root, stem):
    """the faithful stage-4 reading: CONT tests every burst (stage-4 row span) of a spec, not only the placed picture's
    start; a spec keeps its surviving bursts (start/end re-derived) and is dropped when none survives; then the display
    merge and row placement run as in the pipeline (B.placed)"""
    fx = B.FLEX_DIR / f"{stem}.npz"
    runs = runs_of(DCC.load_fr(fx), BAR) if fx.exists() else None
    f = work_root / stem / "augmentations.json"
    if not f.exists():
        return []
    specs = json.loads(f.read_text(encoding="utf-8"))
    kept = []
    for sp in specs:
        bursts = sp.get("spans") or [[sp["start"], sp["end"]]]
        ok = [x for x in bursts if not continuation(sp["event_label"], float(x[0]), runs, WIN)]
        if not ok:
            continue
        sp = dict(sp); sp["spans"] = ok; sp["start"] = min(float(x[0]) for x in ok); sp["end"] = max(float(x[1]) for x in ok)
        kept.append(sp)
    m = work_root / stem / "media.json"
    dur = (float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None) if m.exists() else None
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs, raw = [], {}
    for x in kept:                                   # as B.placed, on the filtered specs (no file is touched)
        o = AugmentationSpec(index=x.get("index", 0), event_label=x["event_label"], start=float(x["start"]), end=float(x["end"]),
                             augment=bool(x.get("augment")), confidence=float(x.get("confidence", 0)), image_path=x.get("image_path"),
                             talked_about=bool(x.get("talked_about")), spans=[tuple(y) for y in x.get("spans", [])],
                             breaks=[tuple(y) for y in x.get("breaks", [])])
        objs.append(o); raw[id(o)] = x
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    pl, _ = _assign_rows(_display_spans(objs, d, require_image=True))
    return [(lab, float(a), float(b), bool(raw[id(sp)].get("rescued", False))) for _, lab, a, b, sp in pl]


ROWS = {}


def placed_both(work_root, stem):
    """B.placed's answer (the saved pictures), also caching the rows-level CONT placement (parts() runs once: the tagger
    redirect in btp_screen.parts mutates DCC.dev_stems, so it cannot be called twice)"""
    ROWS[stem] = placed_rows(work_root, stem)
    return _ORIG_PLACED(work_root, stem)


def main():
    B.placed = placed_both
    P = B.parts()
    B.placed = _ORIG_PLACED
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures; bar {BAR}")
    res = {}
    variants = [("CONT", "flex", WIN), ("CONT-beats", "beats", WIN), ("CONT-1.0", "flex", 1.0)]
    for name, src, win in variants:
        rows = {"dev": [], "dev2": []}; base = {"dev": [], "dev2": []}; dropped = []; lost_parts = []
        for pt, st, g, pics in P:
            if src == "flex":
                fx = B.FLEX_DIR / f"{st}.npz"; fr = DCC.load_fr(fx) if fx.exists() else None
            else:
                bx = BEATS[pt] / f"{st}.npz"; fr = DCC.load_fr(bx) if bx.exists() else None
            runs = runs_of(fr, BAR)
            cl = classify(g, [p[:3] for p in pics])
            before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
            keep = [(l, a, b) for l, a, b, _r in pics if not continuation(l, a, runs, win)]
            gone = [(l, a, b) for l, a, b, _r in pics if continuation(l, a, runs, win)]
            r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
            base[pt].append(r0); rows[pt].append(r1)
            for l, a, b in gone:
                dropped.append({"part": pt, "clip": st, "label": l, "start": round(a, 2), "end": round(b, 2), "was": before[(l, round(a, 3))]})
            if r1["hit"] < r0["hit"]:
                lost_parts.append((pt, st, r0["hit"] - r1["hit"]))
        Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
        X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
        if name == "CONT":
            print(f"BASE SHIP5: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
            assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
            assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
            BASE["cost"] = Bm["merged"]["cost"]          # the exact reproduced base cost (2.620 is its rounding)
        old, few = passes(X["merged"], lost_parts)
        verdict = "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
        was = {}
        for d in dropped:
            was[d["was"]] = was.get(d["was"], 0) + 1
        print(f"{name}: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  dropped {len(dropped)} {was}  "
              f"hits lost {lost_parts} -> {verdict}")
        for d in dropped:
            print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['start']}-{d['end']} was {d['was']}")
        res[name] = {"src": src, "win": win, "rows": X, "base": Bm, "dropped": dropped, "hits_lost": lost_parts,
                     "old_rule": old, "fewer_pictures": few, "verdict": verdict}
    # faithful rows-level CONT (every burst tested), scored against the same base
    P2 = {(pt, st): ROWS[st] for pt, st, g, pics in P}
    rows = {"dev": [], "dev2": []}; lost_parts = []; changed = []
    for pt, st, g, pics in P:
        new = P2[(pt, st)]
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, [p[:3] for p in new])
        rows[pt].append(r1)
        if r1["hit"] < r0["hit"]:
            lost_parts.append((pt, st, r0["hit"] - r1["hit"]))
        a0 = sorted((l, round(a, 2), round(b, 2)) for l, a, b, _r in pics); a1 = sorted((l, round(a, 2), round(b, 2)) for l, a, b, _r in new)
        if a0 != a1:
            oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
            changed.append({"part": pt, "clip": st, "before": a0, "after": a1, "b": oc(r0), "a": oc(r1)})
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    old, few = passes(X["merged"], lost_parts)
    verdict = "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
    print(f"CONT-rows (faithful, every burst tested): merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  "
          f"hits lost {lost_parts} -> {verdict}")
    for c in changed:
        print(f"   {c['part']:4s} {c['clip']}: {c['before']} -> {c['after']}  {c['b']} -> {c['a']}")
    res["CONT-rows"] = {"rows": X, "changed": changed, "hits_lost": lost_parts, "old_rule": old, "fewer_pictures": few, "verdict": verdict}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
