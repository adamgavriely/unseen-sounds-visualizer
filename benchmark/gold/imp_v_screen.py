"""Round 37 IMP-V (docs/prereg_round13_detector_push.md "Round 37 IMP-V"): at every Round 36 DASM impact peak that is not
inside an existing SHIP8 impact-type picture, the shipped gate VLM (Qwen3.8-27B) is asked ONE closed question on the gate's
own 6 frames (peak +- 1 s), in both option orders: which impact family made the sound. Both orders must agree on a family;
then the shipped visibility gate (reason._sound_is_visible, majority) decides on the same frames: seen -> no picture, else a
picture (family, t, t + 2 s). Arm = saved SHIP8 pictures + new pictures, rescored dbr_screen-style vs SHIP8 (28/58, 21, 2.282).
Nothing in src/ or config.py is edited. The rule never reads the gold.

    TG_ARMS=SHIP8 python benchmark/gold/imp_v_screen.py run      # GPU, one JSON per peak in benchmark/gold/imp_v/, resumable
    TG_ARMS=SHIP8 python benchmark/gold/imp_v_screen.py score    # CPU -> benchmark/gold/imp_v_screen.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import imp_screen as I
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from src.labels import canonical

B.ARM = "SHIP8"
MODEL = "Qwen/Qwen3.8-27B"                                       # the shipped gate VLM (config profile 5)
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}                 # SHIP8 on the final merged DEV
PIC_LEN, DUP_GAP = 2.0, 1.0
CACHE = _ROOT / "benchmark" / "gold" / "imp_v"
OUT = _ROOT / "benchmark" / "gold" / "imp_v_screen.json"
QUESTION = "A short impact sound is heard at this moment. Which of these most likely made it?"
OPTIONS = [                                                      # (phrase shown, AudioSet label drawn)
    ("a whack or thwack -- a club, bat, racket or hand striking something", "Whack, thwack"),
    ("a hammer striking", "Hammer"),
    ("a knock -- knuckles on a door or wood", "Knock"),
    ("a door slamming", "Slam"),
    ("a clang -- metal struck", "Clang"),
    ("something smashing or crashing, glass breaking", "Smash, crash"),
    ("dishes, pots or pans clattering", "Dishes, pots, and pans"),
    ("a gunshot", "Gunshot, gunfire"),
    ("an explosion", "Explosion"),
    ("a thump or thud -- a heavy object landing", "Thump, thud"),
    ("a door opening or closing", "Door"),
    ("none of these / cannot tell", None),
]
IMPACT_FAMS = {canonical(l) for _, l in OPTIONS if l} | {"Specific impact sounds"}
LETTERS = "abcdefghijkl"


def clip_file(pt, st):
    if pt == "dev":
        from benchmark.gold.detector_dry import clip_path
        return clip_path(st + ".mp4") or next(iter(p for d in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient")
                                                   for p in (_ROOT / "data" / "input" / "benchmark" / d).glob(st + ".*")), None)
    from benchmark.gold import tagger_prep as T
    return T.CLIPS / f"{st}.mp4"


def all_peaks():
    """Round 36's 30 peaks, recomputed from the DASM caches (asserted equal to imp_screen.json)"""
    _gold, ST = I.stems()
    rows = []
    for pt, st in ST:
        for pk in I.peaks(DCC.load_fr(I.PARTS[pt] / f"{st}.npz")):
            rows.append({"part": pt, "clip": st, "t": pk["t"], "peak": pk["peak"]})
    ref = json.load(open(I.OUT, encoding="utf-8"))["peaks"]
    assert {(r["clip"], r["t"]) for r in rows} == {(r["clip"], r["t"]) for r in ref}, "peaks differ from imp_screen.json"
    assert len(rows) == 30, len(rows)
    return rows


def pics_by_clip():
    return {(pt, st): (g, pics) for pt, st, g, pics in B.parts()}


def inside_existing(pics, t):
    return [(l, round(a, 2), round(b, 2)) for l, a, b, _r in pics if canonical(l) in IMPACT_FAMS and a - 1e-6 <= t <= b + 1e-6]


def ask_family(mdl, proc, frames):
    """the closed question in forward and fully reversed order -> (label or None, [raw replies], [labels per order])"""
    from src.stage5_cross_modal_analysis import reason
    picks, raws = [], []
    for order in (OPTIONS, OPTIONS[::-1]):
        prompt = QUESTION + chr(10) + chr(10).join(f"({LETTERS[i]}) {ph}" for i, (ph, _l) in enumerate(order)) \
                 + chr(10) + "Answer with the letter only."
        reply = reason._ask(mdl, proc, prompt, images=frames, max_new=6).strip().lower().lstrip("(")
        raws.append(reply)
        c = next((ch for ch in reply if ch.isalpha()), "")
        picks.append(order[LETTERS.index(c)][1] if c in LETTERS[:len(order)] else None)
    agreed = picks[0] if picks[0] is not None and picks[0] == picks[1] else None
    return agreed, raws, picks


def run(device="cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    CACHE.mkdir(parents=True, exist_ok=True)
    peaks = all_peaks()
    P = pics_by_clip()
    todo = [p for p in peaks if not (CACHE / f"{p['clip']}_{p['t']:.2f}.json").exists()]
    print(f"{len(peaks)} peaks, {len(todo)} to run", flush=True)
    mdl = proc = None
    for p in todo:
        pt, st, t = p["part"], p["clip"], p["t"]
        _g, pics = P[(pt, st)]
        rec = {**p, "inside_existing": inside_existing(pics, t)}
        if rec["inside_existing"]:
            rec["decision"] = "skip: inside existing impact-type picture"
        else:
            if mdl is None:
                mdl, proc = reason._load(MODEL, device)
            vp = clip_file(pt, st)
            assert vp is not None and Path(vp).exists(), (pt, st, vp)
            times = [max(0.0, t - 1.0 + 2.0 * i / 5) for i in range(6)]
            frames = _sample_frames_at(Path(vp), times)
            rec["frame_times"] = [round(x, 2) for x in times]
            fam, raws, picks = ask_family(mdl, proc, frames)
            rec.update({"vlm_raw": raws, "vlm_family_fwd": picks[0], "vlm_family_rev": picks[1], "family": fam})
            if fam is None:
                rec["decision"] = "no picture: none / split"
            else:
                seen, named = reason._sound_is_visible(fam, frames, mdl, proc, device)
                rec.update({"gate_seen": bool(seen), "gate_votes": dict(reason.LAST_VOTES)})
                dup = [(l, round(a, 2)) for l, a, _b, _r in pics if canonical(l) == canonical(fam) and abs(a - t) <= DUP_GAP]
                if seen:
                    rec["decision"] = "no picture: gate says seen"
                elif dup:
                    rec["decision"] = "no picture: duplicate of SHIP8 picture"; rec["dup_of"] = dup
                else:
                    rec["decision"] = "picture"; rec["picture"] = [fam, t, round(t + PIC_LEN, 2)]
        (CACHE / f"{st}_{t:.2f}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{st} t={t}: {rec.get('vlm_family_fwd')} / {rec.get('vlm_family_rev')} -> {rec['decision']}", flush=True)


def passes(X, lost_parts, lost_n):
    gain = X["hits"] - BASE["hits"]
    old = (X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"]
           and not lost_parts)
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * lost_n and lost_n <= 3
    return old, few


def score():
    peaks = all_peaks()
    recs = []
    for p in peaks:
        f = CACHE / f"{p['clip']}_{p['t']:.2f}.json"
        assert f.exists(), f
        recs.append(json.load(open(f, encoding="utf-8")))
    P = B.parts()
    base = {"dev": [], "dev2": []}; rows = {"dev": [], "dev2": []}; lost = []
    for pt, st, g, pics in P:
        old = [p[:3] for p in pics]
        new = [tuple(r["picture"]) for r in recs if r["clip"] == st and r.get("picture")]
        r0, r1 = S.score_clip(g, old), S.score_clip(g, old + new)
        base[pt].append(r0); rows[pt].append(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
        if new:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, old + new)}
            for r in recs:
                if r["clip"] == st and r.get("picture"):
                    r["class"] = cl[(r["picture"][0], round(r["picture"][1], 3))]
            oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
            for r in recs:
                if r["clip"] == st and r.get("picture"):
                    r["clip_before"], r["clip_after"] = oc(r0), oc(r1)
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    BASE["cost"] = Bm["merged"]["cost"]
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    ln = sum(x[2] for x in lost)
    old, few = passes(X["merged"], lost, ln)
    verdict = "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
    bins = {"inside existing": 0, "named -> gate seen": 0, "named -> duplicate": 0, "named -> drawn": 0, "unnamed / split / none": 0}
    for r in recs:
        d = r["decision"]
        k = ("inside existing" if d.startswith("skip") else "named -> gate seen" if "seen" in d else
             "named -> duplicate" if "duplicate" in d else "named -> drawn" if d == "picture" else "unnamed / split / none")
        bins[k] += 1
    print(f"IMP-V: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  bins {bins}  hits lost {lost} -> {verdict}")
    for r in recs:
        print(f"  {r['part']:4s} {r['clip']:28s} t={r['t']:6.2f} fwd={r.get('vlm_family_fwd')} rev={r.get('vlm_family_rev')} "
              f"gate={'seen' if r.get('gate_seen') else ('not seen' if 'gate_seen' in r else '-')} "
              f"votes={r.get('gate_votes', {}).get('name')}/{r.get('gate_votes', {}).get('ab')}/{r.get('gate_votes', {}).get('desc')} "
              f"-> {r['decision']}{(' = ' + r['class']) if r.get('class') else ''}")
    OUT.write_text(json.dumps({"base": Bm, "rows": X, "bins": bins, "hits_lost": lost, "old_rule": old, "fewer_pictures": few,
                               "verdict": verdict, "model": MODEL, "options": OPTIONS, "peaks": recs}, indent=1, default=float),
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
