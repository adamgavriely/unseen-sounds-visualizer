"""Round 54 SIGN step 2 (docs/prereg_round13_detector_push.md "Round 54 SIGN"): the SIGN question asked on the SAVED
SHIP8+MD3 merged-DEV pictures (49 DEV + DEV2 tagger clips). For every drawn spec (augment and image) of the arm, its own
gate stretches are read from the clip's gate_votes.json (label + spec start within 0.011 s); if none are logged, they are
recomputed from the spec's spans by the gate's VISIBILITY_STRETCH rule (flagged "fallback"). On each stretch the shipped
gate VLM (Qwen3.8-27B) gets the pipeline's own 6 frames (stretch -1 s .. +1 s, as reason.decide_subjects) and the SIGN a/b
question of sign_gate.py with the PIPELINE label, both letter orders. A spec is silenced iff sign = (a) in both orders on
EVERY stretch; the silenced specs are set augment=False and the pictures re-placed (_display_spans with clip=stem, as
score_per_sound.load_pictures), then scored on merged DEV. Kinship silencing of other sounds is NOT simulated.

    TG_ARMS=SHIP8+MD3 python benchmark/gold/sign_screen.py specs   # CPU: list asked specs + base reproduction
    TG_ARMS=SHIP8+MD3 python benchmark/gold/sign_screen.py run     # GPU -> benchmark/gold/sign_pics/
    TG_ARMS=SHIP8+MD3 python benchmark/gold/sign_screen.py score   # CPU -> benchmark/gold/sign_screen.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")

ARM, SYS = "SHIP8+MD3", "proposed"
MODEL = "Qwen/Qwen3.8-27B"
BASE = {"hits": 29, "wrong": 18, "cost": 2.141}
CACHE = _ROOT / "benchmark" / "gold" / "sign_pics"
OUT = _ROOT / "benchmark" / "gold" / "sign_screen.json"
NAMED = ("as_explosion_XJ8lc3I6", "tg_d088")


def placed(work_root, stem, silenced=frozenset()):
    """score_per_sound.load_pictures (proposed) with the specs whose (label, start) is in `silenced` set augment=False"""
    f = work_root / stem / "augmentations.json"
    if not f.exists():
        return []
    specs = json.loads(f.read_text(encoding="utf-8"))
    m = work_root / stem / "media.json"
    dur = (float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None) if m.exists() else None
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = []
    for s in specs:
        off = (s["event_label"], round(float(s["start"]), 3)) in silenced
        objs.append(AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]),
                                     end=float(s["end"]), augment=bool(s.get("augment")) and not off,
                                     confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                                     talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                                     breaks=[tuple(x) for x in s.get("breaks", [])]))
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    pl, _ = _assign_rows(_display_spans(objs, d, require_image=True, clip=stem))
    return [(lab, float(a), float(b)) for _, lab, a, b, sp in pl]


def parts(mask=None):
    """[(part, stem, gold, work_root, base pictures, masked pictures)] -- btp_screen.parts with placed() above"""
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import round13_dev as R
    mask = mask or {}
    gold, stems = DCC.dev_stems()
    out = []
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        w = R.R13 / f"{ARM}_{SYS}"
        out += [("dev", st, gold[st], w, placed(w, st), placed(w, st, mask.get(("dev", st), frozenset()))) for st in stems]
    from benchmark.gold import tagger_prep as T
    DCC2, R2, stems2 = T.configure("dev2")
    keep = set(stems2)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out("dev2") / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    g2 = T._REAL_LOAD_GOLD([tmp])
    with R2.flags({k: R2.arm_cfg(ARM)[k] for k in R2.DISPLAY_KEYS}):
        w = T.out("dev2") / f"{ARM}_{SYS}"
        out += [("dev2", st, g2[st], w, placed(w, st), placed(w, st, mask.get(("dev2", st), frozenset())))
                for st in stems2 if st in g2]
    return out


def stretches_of(spec, votes, step):
    got = [v["stretch"] for v in votes if v["label"] == spec["event_label"] and abs(float(v["start"]) - float(spec["start"])) < 0.011]
    if got:
        return [[float(a), float(b)] for a, b in got], "gate_votes"
    pieces = []
    for a, b in (spec.get("spans") or [[spec["start"], spec["end"]]]):
        k = max(1, int(round((b - a) / step)))
        e = [a + (b - a) * i / k for i in range(k + 1)]
        pieces += [[float(x), float(y)] for x, y in zip(e, e[1:])]
    return pieces, "fallback"


def asked(rows):
    """[(part, stem, work_root, spec label, spec start, stretches, source)] for every drawn spec of the arm"""
    import config
    step = float(getattr(config, "VISIBILITY_STRETCH", 5.0))
    out = []
    for pt, st, g, w, pics, _ in rows:
        f = w / st / "augmentations.json"
        if not f.exists():
            continue
        vf = w / st / "gate_votes.json"
        votes = json.loads(vf.read_text(encoding="utf-8")) if vf.exists() else []
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("augment") and s.get("image_path"):
                sts, src = stretches_of(s, votes, step)
                out.append((pt, st, w, s["event_label"], round(float(s["start"]), 3), sts, src))
    return out


def key(pt, st, lab, a):
    return f"{pt}_{st}_{lab.replace(' ', '_').replace(',', '').replace('.', '').replace('/', '_')}_{a:.2f}"


def summ_fmt(rows):
    from benchmark.gold import btp_screen as B
    return B.summ(rows), B.fmt(B.summ(rows))


def cmd_specs():
    rows = parts()
    A = asked(rows)
    print(f"{len(rows)} clips, {sum(len(r[4]) for r in rows)} placed pictures, {len(A)} drawn specs, "
          f"{sum(len(a[5]) for a in A)} stretches, fallback specs {sum(a[6] == 'fallback' for a in A)}")
    for a in A:
        if a[6] == "fallback":
            print("   fallback", a[0], a[1], a[3], a[4], a[5])
    from benchmark.gold import score_per_sound as S
    b = summ_fmt([S.score_clip(g, pics) for pt, st, g, w, pics, _ in rows])
    print("BASE", ARM, b[1])


def cmd_run(device="cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    from benchmark.gold.imp_v_screen import clip_file
    from benchmark.gold.sign_gate import ab_logged, STEM, OPT_YES, OPT_NO
    CACHE.mkdir(parents=True, exist_ok=True)
    A = [a for a in asked(parts()) if not (CACHE / f"{key(a[0], a[1], a[3], a[4])}.json").exists()]
    print(len(A), "specs to ask", flush=True)
    if not A:
        return
    mdl, proc = reason._load(MODEL, device)
    for pt, st, w, lab, a0, sts, src in A:
        vp = Path(clip_file(pt, st))
        res = []
        for a, b in sts:
            lo, hi = a - 1.0, b + 1.0
            times = [lo + (hi - lo) * t / 5 for t in range(6)]          # decide_subjects, n = 6, no clamp
            frames = _sample_frames_at(vp, times)
            if len(frames) < 2:
                res.append({"stretch": [a, b], "sign": False, "replies": [], "n_frames": len(frames), "note": "frames"}); continue
            v, rep = ab_logged(reason, mdl, proc, STEM, OPT_YES.format(label=lab), OPT_NO, frames)
            res.append({"stretch": [a, b], "sign": v, "replies": rep, "n_frames": len(frames)})
        sil = bool(res) and all(r["sign"] is True for r in res)
        rec = {"part": pt, "clip": st, "label": lab, "start": a0, "source": src, "clip_file": str(vp), "stretches": res,
               "silenced": sil}
        (CACHE / f"{key(pt, st, lab, a0)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(pt, st, lab, a0, src, [(r["sign"], r["replies"]) for r in res], "->", "SILENCED" if sil else "kept", flush=True)


def cmd_score():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    rows0 = parts()
    A = asked(rows0)
    mask, tally, recs = {}, {"silenced": 0, "kept": 0, "unasked": 0}, []
    for pt, st, w, lab, a0, sts, src in A:
        f = CACHE / f"{key(pt, st, lab, a0)}.json"
        if not f.exists():
            tally["unasked"] += 1; continue
        r = json.loads(f.read_text(encoding="utf-8"))
        tally["silenced" if r["silenced"] else "kept"] += 1
        if r["silenced"]:
            mask.setdefault((pt, st), set()).add((lab, round(a0, 3)))
            recs.append(r)
    mask = {k: frozenset(v) for k, v in mask.items()}
    rows = parts(mask)
    base, new = {"dev": [], "dev2": []}, {"dev": [], "dev2": []}
    lost, changed = [], []
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, g, w, pics, pics1 in rows:
        r0, r1 = S.score_clip(g, pics), S.score_clip(g, pics1)
        base[pt].append(r0); new[pt].append(r1)
        if pics != pics1:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, pics)}
            gone = [p for p in pics if p not in pics1]; added = [p for p in pics1 if p not in pics]
            changed.append({"part": pt, "clip": st, "removed": [[l, round(a, 2), round(b, 2), cl.get((l, round(a, 3)))] for l, a, b in gone],
                            "added": [[l, round(a, 2), round(b, 2)] for l, a, b in added], "before": oc(r0), "after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    Bm = {k: summ_fmt(v)[0] for k, v in (("merged", base["dev"] + base["dev2"]), ("dev", base["dev"]), ("dev2", base["dev2"]))}
    X = {k: summ_fmt(v)[0] for k, v in (("merged", new["dev"] + new["dev2"]), ("dev", new["dev"]), ("dev2", new["dev2"]))}
    from benchmark.gold import btp_screen as B
    print(f"BASE {ARM}: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    M = X["merged"]; ln = sum(x[2] for x in lost); removed_wrong = Bm["merged"]["wrong"] - M["wrong"]
    cheaper = M["cost"] < Bm["merged"]["cost"]
    unit = (removed_wrong >= 1) if ln == 0 else (removed_wrong >= 2 * ln)
    main = M["hits"] >= BASE["hits"] and not lost and cheaper
    few = cheaper and M["wrong"] <= BASE["wrong"] - 3 * ln and M["hits"] >= 26
    verdict = "PASS" if (cheaper and unit and (main or few)) else "FAIL"
    print(f"SIGN: merged {B.fmt(M)} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  specs {tally}  hits lost {lost}  "
          f"wrong removed {removed_wrong}  cheaper {cheaper} unit {unit} main {main} few {few} -> {verdict}")
    for c in changed:
        print("  ", c)
    named = {n: [c for c in changed if c["clip"] == n] or "unchanged" for n in NAMED}
    print("named:", named)
    res = {"base": Bm, "rows": X, "spec_tally": tally, "silenced_specs": recs, "changed": changed, "hits_lost": lost,
           "wrong_removed": removed_wrong, "cheaper": cheaper, "cost_unit": unit, "main_rule": main, "fewer_pictures": few,
           "verdict": verdict, "named": named, "model": MODEL}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"specs": cmd_specs, "run": cmd_run, "score": cmd_score}[cmd]()
