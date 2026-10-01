"""Round 39 MAKER-VIS screen (docs/prereg_round13_detector_push.md "Round 39 MAKER-VIS"): for every placed picture of the
saved SHIP8 arm on merged DEV (49 DEV + 22 tagger DEV2), the MAKER named by the picture's own depiction ("Train releases
steam" -> "train") is extracted by a fixed token rule (maker_of) and the shipped gate VLM (Qwen3.8-27B) is asked, on the
gate's own 6 frames (picture start -1 s .. +1 s), the a/b question "<maker> is visible in these frames / no <maker> is
visible" in both option orders (reason._ab). Both orders "visible" -> the picture is dropped. Rescored dbr_screen-style
on gold_AG.json vs SHIP8 (28/58, 21 (6/13/2), 2.282). The sound label is never in the prompt; the rule never reads the gold.
Nothing in src/ or config.py is edited.

    TG_ARMS=SHIP8 python benchmark/gold/maker_vis_screen.py makers   # CPU: the frozen (depiction -> maker) table
    TG_ARMS=SHIP8 python benchmark/gold/maker_vis_screen.py run      # GPU: one JSON per picture in benchmark/gold/maker_vis/
    TG_ARMS=SHIP8 python benchmark/gold/maker_vis_screen.py score    # CPU -> benchmark/gold/maker_vis_screen.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

B_ARM = "SHIP8"
MODEL = "Qwen/Qwen3.8-27B"                                       # the shipped gate VLM (config profile 5)
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}                 # SHIP8 on the final merged DEV (gold_AG.json)
CACHE = _ROOT / "benchmark" / "gold" / "maker_vis"
OUT = _ROOT / "benchmark" / "gold" / "maker_vis_screen.json"
STEM = "These frames are from a video. Judge from the frames alone."
MASS = {"water", "sky", "rain", "steam", "snow", "wind", "fire", "smoke", "thunder", "glass", "traffic", "people"}
# MASS words take no article; "people" is plural ("are"), the rest singular ("is").


def _plural(tok: str) -> bool:
    """plural noun by form: ends in s but not ss (glass, grass); 'people' by exception"""
    t = tok.lower()
    return t == "people" or (t.endswith("s") and not t.endswith("ss"))


def maker_of(depiction: str) -> str:
    """The grammatical subject of the depiction sentence, by a fixed whitespace-token rule (no parser on the cluster).

    The maker is every token before the first VERB token; a token (index i >= 1) is a verb when
      (i)   it ends in "ing" and is longer than 4 letters                        (People laughING, Bird flyING)
      (ii)  it ends in "s" (not "ss") and the token before it is NOT plural       (Sky rumbleS, Man shaveS, Glass shatterS)
      (iii) it does not end in "s" and the token before it IS plural             (Fingers STRIKE, Clouds RUMBLE)
    Token 0 is never a verb. No verb found -> the whole depiction is the maker (Emergency vehicle siren).
    Returned lower-cased, punctuation stripped.
    """
    toks = ["".join(c for c in t if c.isalnum() or c in "-'") for t in depiction.strip().split()]
    toks = [t for t in toks if t]
    cut = len(toks)
    for i in range(1, len(toks)):
        t = toks[i].lower(); prev_pl = _plural(toks[i - 1])
        if (t.endswith("ing") and len(t) > 4) or (t.endswith("s") and not t.endswith("ss") and not prev_pl) \
                or (not t.endswith("s") and prev_pl):
            cut = i
            break
    return " ".join(toks[:cut]).lower()


def phrase_options(maker: str):
    """-> (opt_yes, opt_no): article by first letter, none for plurals and MASS words; is/are by number"""
    head = maker.split()[-1]
    plural = _plural(head)
    verb = "are" if plural else "is"
    if plural or head in MASS:
        art = ""
    else:
        art = "an " if maker[0] in "aeiou" else "a "
    return f"{art}{maker} {verb} visible in these frames", f"no {maker} {verb} visible"


def clip_file(pt, st):
    from benchmark.gold.imp_v_screen import clip_file as cf
    return cf(pt, st)


def spec_root(pt):
    """the saved-arm roots btp_screen.placed() read (tagger_prep.configure re-points round13_dev, so not R.R13 after parts())"""
    if pt == "dev":
        return _ROOT / "data" / "work" / "r13" / f"{B_ARM}_proposed"
    from benchmark.gold import tagger_prep as T
    return T.out("dev2") / f"{B_ARM}_proposed"


def depiction_of(spec):
    s = (spec.get("subject") or "").strip()
    if s:
        return s
    r = spec.get("reason") or ""
    return r.split("depiction:")[-1].strip() if "depiction:" in r else ""


def pictures():
    """[(part, stem, gold, pics, [per placed picture: dict(label, start, end, rescued, depiction, maker, spec_start, match)])]"""
    from benchmark.gold import btp_screen as B
    B.ARM = B_ARM
    rows = []
    for pt, st, g, pics in B.parts():
        f = spec_root(pt) / st / "augmentations.json"
        specs = [s for s in json.loads(f.read_text(encoding="utf-8")) if s.get("augment") and s.get("image_path")] if f.exists() else []
        info = []
        for lab, a, b, resc in pics:
            same = [s for s in specs if s["event_label"] == lab]
            exact = [s for s in same if abs(float(s["start"]) - a) < 0.011
                     or any(abs(float(x[0]) - a) < 0.011 for x in s.get("spans", []))]
            if exact:
                sp, how = exact[0], "exact"
            elif same:
                sp, how = min(same, key=lambda s: abs(float(s["start"]) - a)), "nearest same-label"
            else:
                sp, how = None, "no spec"
            dep = depiction_of(sp) if sp else ""
            info.append({"label": lab, "start": round(a, 3), "end": round(b, 3), "rescued": bool(resc), "depiction": dep,
                         "maker": maker_of(dep) if dep else "", "spec_start": (round(float(sp["start"]), 3) if sp else None),
                         "match": how})
        rows.append((pt, st, g, pics, info))
    return rows


def key(pt, st, p):
    return f"{st}_{p['label'].replace(' ', '_').replace(',', '').replace('.', '')}_{p['start']:.2f}"


def cmd_makers():
    tab = []
    for pt, st, g, pics, info in pictures():
        for p in info:
            tab.append({"part": pt, "clip": st, **p})
            y, n = phrase_options(p["maker"]) if p["maker"] else ("", "")
            print(f"{pt:4s} {st:34s} {p['label']:30s} {p['start']:6.2f} {'R' if p['rescued'] else ' '} "
                  f"{p['depiction']!r:32s} -> {p['maker']!r:28s} [{y} / {n}] ({p['match']})")
    print(len(tab), "placed pictures")
    (CACHE.parent / "maker_vis_makers.json").write_text(json.dumps(tab, indent=1), encoding="utf-8")


def cmd_run(device="cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = pictures()
    todo = [(pt, st, p) for pt, st, g, pics, info in rows for p in info if not (CACHE / f"{key(pt, st, p)}.json").exists()]
    print(f"{sum(len(r[4]) for r in rows)} placed pictures, {len(todo)} to ask", flush=True)
    if not todo:
        return
    mdl, proc = reason._load(MODEL, device)
    for pt, st, p in todo:
        out = CACHE / f"{key(pt, st, p)}.json"
        if not p["maker"]:
            out.write_text(json.dumps({**p, "part": pt, "clip": st, "visible": None, "note": "no depiction"}, indent=1)); continue
        vp = clip_file(pt, st)
        a = p["start"]
        times = [a - 1.0 + 0.4 * t for t in range(6)]
        win = _sample_frames_at(Path(vp), times)
        y, n = phrase_options(p["maker"])
        replies = []
        orig_ask = reason._ask

        def logged(mdl_, proc_, prompt, images=None, max_new=48):
            r = orig_ask(mdl_, proc_, prompt, images=images, max_new=max_new); replies.append(r); return r
        reason._ask = logged
        try:
            vis = reason._ab(mdl, proc, STEM, y, n, frames=win)
        finally:
            reason._ask = orig_ask
        rec = {**p, "part": pt, "clip": st, "clip_file": str(vp), "times": [round(t, 2) for t in times], "n_frames": len(win),
               "opt_yes": y, "opt_no": n, "replies": replies, "visible": vis}
        out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{pt:4s} {st} {p['label']} {a:.2f} maker={p['maker']!r} -> {vis} {replies}", flush=True)


def cmd_score():
    from benchmark.gold import btp_screen as B
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    rows = pictures()
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics, info in rows:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    new = {"dev": [], "dev2": []}; dropped = []; lost = []; tally = {"visible (dropped)": 0, "not visible": 0, "split": 0, "unasked": 0}
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, g, pics, info in rows:
        cl = classify(g, [p[:3] for p in pics])
        before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
        keep = []
        for (lab, a, b, resc), p in zip(pics, info):
            f = CACHE / f"{key(pt, st, p)}.json"
            vis = json.loads(f.read_text(encoding="utf-8")).get("visible") if f.exists() else "unasked"
            if vis is True:
                tally["visible (dropped)"] += 1
                dropped.append({"part": pt, "clip": st, "label": lab, "start": round(a, 2), "end": round(b, 2), "rescued": bool(resc),
                                "depiction": p["depiction"], "maker": p["maker"], "before": before[(lab, round(a, 3))]})
            else:
                tally["not visible" if vis is False else "split" if vis is None else "unasked"] += 1
                keep.append((lab, a, b))
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
        new[pt].append(r1)
        for d in dropped:
            if d["part"] == pt and d["clip"] == st and "clip_after" not in d:
                d["clip_before"], d["clip_after"] = oc(r0), oc(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(new["dev"] + new["dev2"]), "dev": B.summ(new["dev"]), "dev2": B.summ(new["dev2"])}
    ln = sum(x[2] for x in lost)
    M = X["merged"]
    main = M["hits"] >= BASE["hits"] and not lost and M["cost"] < Bm["merged"]["cost"]
    few = M["cost"] < Bm["merged"]["cost"] and M["wrong"] <= BASE["wrong"] - 3 * ln and ln <= 3
    verdict = "GO (main rule)" if main else "GO (fewer-pictures clause)" if few else "STOP"
    print(f"MAKER-VIS: merged {B.fmt(M)} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  asked {tally}  "
          f"hits lost {lost} -> {verdict}")
    for d in dropped:
        print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['start']}-{d['end']}{' (rescued)' if d['rescued'] else ''} "
              f"{d['depiction']!r} maker={d['maker']!r}: {d['before']} dropped   clip {d['clip_before']} -> {d['clip_after']}")
    res = {"base": Bm, "rows": X, "dropped": dropped, "hits_lost": lost, "asked": tally, "main_rule": main,
           "fewer_pictures": few, "verdict": verdict, "model": MODEL, "stem": STEM}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"makers": cmd_makers, "run": cmd_run, "score": cmd_score}[cmd]()
