"""Round 31 SUBJ screen (docs/prereg_round13_detector_push.md "Round 31 SUBJ"): scene-subject silence on the saved SHIP5
(= SHIP4+BTP) pictures of merged DEV. A drawn spec whose every gate stretch said "not seen" is silenced iff the clip's scene
sentence is about the sound: word overlap (`reason._about_the_sound`, SUBJ-W) OR the gate VLM answers yes to
MAKES_SOUND_PROMPT with the scene sentence as the thing visible (SUBJ; text only, greedy). Nothing in src/ or config.py is
edited. GO iff wrong <= 30 with 0 current hits lost.

    python benchmark/gold/subj_screen.py ask      # GPU: VLM answers -> benchmark/gold/subj_answers.json
    python benchmark/gold/subj_screen.py score    # CPU: SUBJ-W, SUBJ, SUBJ-50 vs base
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import btp_screen as B
from benchmark.gold import score_per_sound as S
from benchmark.gold import round13_dev as R

ARM = "SHIP4+BTP"
SCENES = _ROOT / "benchmark" / "gold" / "scene_sentences_ship5.json"
ANSWERS = _ROOT / "benchmark" / "gold" / "subj_answers.json"
OUT = _ROOT / "benchmark" / "gold" / "subj_screen.json"
GATE_MODEL = "Qwen/Qwen3.8-27B"


def load_parts():
    B.ARM = ARM
    dev_root = R.R13 / f"{ARM}_{B.SYS}"
    P = B.parts()
    from benchmark.gold import tagger_prep as T
    roots = {"dev": dev_root, "dev2": T.out("dev2") / f"{ARM}_{B.SYS}"}
    return P, roots


def spec_of(root, stem, lab, a, b):
    """the placed picture's spec (same family, a burst overlapping the picture) and its gate stretches"""
    f = root / stem / "augmentations.json"
    specs = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    g = root / stem / "gate_votes.json"
    votes = json.loads(g.read_text(encoding="utf-8")) if g.exists() else []
    best = None
    for r in specs:
        if not r.get("augment") or not S.same_family(r["event_label"], lab):
            continue
        sp = r.get("spans") or [[r["start"], r["end"]]]
        if any(x <= b + 0.01 and y >= a - 1.6 for x, y in sp):      # BTP may pull a start up to 1.5 s earlier
            best = r; break
    if best is None:
        return None, []
    v = [x for x in votes if x["label"] == best["event_label"] and abs(x["start"] - best["start"]) < 0.01]
    return best, v


def duration(root, stem):
    m = root / stem / "media.json"
    return float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) if m.exists() else 0.0


def items():
    P, roots = load_parts()
    scenes = json.loads(SCENES.read_text(encoding="utf-8"))
    rows = []
    for pt, st, g, pics in P:
        for lab, a, b, resc in pics:
            spec, votes = spec_of(roots[pt], st, lab, a, b)
            rows.append({"part": pt, "clip": st, "pic": [lab, a, b], "rescued": resc, "scene": scenes.get(st),
                         "spec_label": spec["event_label"] if spec else None,
                         "all_not_seen": bool(votes) and not any(v["seen"] for v in votes), "n_stretch": len(votes),
                         "dur": duration(roots[pt], st)})
    return P, rows


def ask():
    from src.stage5_cross_modal_analysis import reason
    P, rows = items()
    done = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}
    mdl, proc = reason._load(GATE_MODEL, "cuda")
    for r in rows:
        if not r["scene"] or not r["spec_label"]:
            continue
        key = f"{r['clip']}|{r['spec_label']}"
        if key in done:
            continue
        reply = reason._ask(mdl, proc, reason.MAKES_SOUND_PROMPT.format(named=r["scene"], label=r["spec_label"]), max_new=6)
        done[key] = reply.strip()
        print(key, "->", done[key], flush=True)
        ANSWERS.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(len(done), "answers")


def score():
    from src.stage5_cross_modal_analysis.reason import _about_the_sound
    P, rows = items()
    ans = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}
    by = {}
    for r in rows:
        w = bool(r["scene"] and r["spec_label"] and _about_the_sound(r["scene"], r["spec_label"]))
        a = ans.get(f"{r['clip']}|{r['spec_label']}", "")
        v = a.strip().lower().startswith("y")
        r["subj_w"], r["vlm"], r["vlm_answer"] = w, v, a
        cover = (r["pic"][2] - r["pic"][1]) / r["dur"] if r["dur"] else 0.0
        r["cover"] = round(cover, 2)
        r["drop"] = {"SUBJ-W": r["all_not_seen"] and w, "SUBJ": r["all_not_seen"] and (w or v),
                     "SUBJ-50": r["all_not_seen"] and (w or v) and cover >= 0.5}
        by.setdefault(r["clip"], []).append(r)
    res = {"n_pictures": len(rows), "n_all_not_seen": sum(r["all_not_seen"] for r in rows), "n_answers": len(ans)}
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = B.summ(base["dev"] + base["dev2"])
    print("BASE", B.fmt(Bm))
    res["base"] = Bm
    for rule in ("SUBJ-W", "SUBJ", "SUBJ-50"):
        out = {"dev": [], "dev2": []}; lost = []; dropped = []
        for pt, st, g, pics in P:
            rr = by.get(st, [])
            keep = [p[:3] for p, r in zip(pics, rr) if not r["drop"][rule]]
            dropped += [(st, r["pic"][0], round(r["pic"][1], 2)) for r in rr if r["drop"][rule]]
            r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
            out[pt].append(r1)
            if r1["hit"] < r0["hit"]:
                lost.append((st, r0["hit"] - r1["hit"]))
        X = B.summ(out["dev"] + out["dev2"])
        go = not lost and X["wrong"] <= 30
        res[rule] = {"merged": X, "dev": B.summ(out["dev"]), "dev2": B.summ(out["dev2"]), "dropped": dropped, "hits_lost": lost, "GO": go}
        print(f"{rule}: merged {B.fmt(X)} | DEV {B.fmt(res[rule]['dev'])} | DEV2 {B.fmt(res[rule]['dev2'])} hits lost {lost} -> {'GO' if go else 'STOP'}")
        print("   dropped:", dropped)
    res["rows"] = rows
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"ask": ask, "score": score}[sys.argv[1]]()
