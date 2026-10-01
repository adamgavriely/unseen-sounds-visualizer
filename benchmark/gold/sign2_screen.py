"""Round 54b SIGN-2 step 0 + step 2 (docs/prereg_round13_detector_push.md "Round 54b SIGN-2"): sign_screen.py with the two
yes/no questions below (opposite polarity); sign = Q1 yes AND Q2 no. Same 43 saved SHIP8+MD3 merged-DEV specs, same
stretches, frames, re-placement and scoring (sign_screen.cmd_score with CACHE / OUT re-pointed).

    TG_ARMS=SHIP8+MD3 python benchmark/gold/sign2_screen.py sanity   # GPU: 10 stretches; exit 3 if >= 9 share an answer pair
    TG_ARMS=SHIP8+MD3 python benchmark/gold/sign2_screen.py run      # GPU -> benchmark/gold/sign2_pics/
    TG_ARMS=SHIP8+MD3 python benchmark/gold/sign2_screen.py score    # CPU -> benchmark/gold/sign2_screen.json
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")

CACHE = _ROOT / "benchmark" / "gold" / "sign2_pics"
OUT = _ROOT / "benchmark" / "gold" / "sign2_screen.json"
SANITY = _ROOT / "benchmark" / "gold" / "sign2_sanity.json"
Q1 = ("Do these frames visibly show {label} happening — its effect or motion (for example a flash, splash, spray, "
      "moving mouth), even with no audio? Answer yes or no.")
Q2 = "Are these frames free of any visible sign that {label} is happening? Answer yes or no."


def yn(reply: str):
    t = reply.strip().lower().lstrip("*\"'([ ").split()
    w = t[0].strip(".,!*\"')]") if t else ""
    return w if w in ("yes", "no") else None


def ask_sign(reason, mdl, proc, label, frames):
    """-> (sign, [q1 reply, q2 reply], (q1 answer, q2 answer))"""
    r1 = reason._ask(mdl, proc, Q1.format(label=label), images=frames, max_new=4)
    r2 = reason._ask(mdl, proc, Q2.format(label=label), images=frames, max_new=4)
    a = (yn(r1), yn(r2))
    return a == ("yes", "no"), [r1, r2], a


def _frames(vp, a, b):
    from src.stage2_video_understanding import _sample_frames_at
    lo, hi = a - 1.0, b + 1.0
    return _sample_frames_at(vp, [lo + (hi - lo) * t / 5 for t in range(6)])     # decide_subjects, n = 6, no clamp


def _load():
    from src.stage5_cross_modal_analysis import reason
    from benchmark.gold import sign_screen as SS
    return reason, SS, reason._load(SS.MODEL, "cuda")


def cmd_sanity():
    from benchmark.gold.imp_v_screen import clip_file
    from benchmark.gold import sign_screen as SS
    flat = [(pt, st, lab, a, b) for pt, st, w, lab, a0, sts, src in SS.asked(SS.parts()) for a, b in sts]
    assert len(flat) == 72, len(flat)
    pick = [flat[i] for i in range(0, 64, 7)]
    reason, SS, (mdl, proc) = _load()
    rows = []
    for pt, st, lab, a, b in pick:
        fr = _frames(Path(clip_file(pt, st)), a, b)
        v, rep, pair = ask_sign(reason, mdl, proc, lab, fr)
        rows.append({"part": pt, "clip": st, "label": lab, "stretch": [a, b], "replies": rep, "pair": list(pair), "sign": v})
        print("SANITY", pt, st, lab, [round(a, 2), round(b, 2)], rep, pair, "sign", v, flush=True)
    c = collections.Counter(str(r["pair"]) for r in rows)
    top = c.most_common(1)[0][1]
    stop = top >= 9
    print("SANITY pairs", dict(c), "->", "STOP (untestable with this VLM)" if stop else "GO", flush=True)
    SANITY.write_text(json.dumps({"rows": rows, "pairs": dict(c), "stop": stop}, indent=1), encoding="utf-8")
    sys.exit(3 if stop else 0)


def cmd_run():
    from benchmark.gold.imp_v_screen import clip_file
    from benchmark.gold import sign_screen as SS
    CACHE.mkdir(parents=True, exist_ok=True)
    A = [a for a in SS.asked(SS.parts()) if not (CACHE / f"{SS.key(a[0], a[1], a[3], a[4])}.json").exists()]
    print(len(A), "specs to ask", flush=True)
    if not A:
        return
    reason, SS, (mdl, proc) = _load()
    for pt, st, w, lab, a0, sts, src in A:
        vp = Path(clip_file(pt, st))
        res = []
        for a, b in sts:
            fr = _frames(vp, a, b)
            if len(fr) < 2:
                res.append({"stretch": [a, b], "sign": False, "replies": [], "pair": None, "n_frames": len(fr), "note": "frames"})
                continue
            v, rep, pair = ask_sign(reason, mdl, proc, lab, fr)
            res.append({"stretch": [a, b], "sign": v, "replies": rep, "pair": list(pair), "n_frames": len(fr)})
        sil = bool(res) and all(r["sign"] is True for r in res)
        rec = {"part": pt, "clip": st, "label": lab, "start": a0, "source": src, "clip_file": str(vp), "stretches": res,
               "silenced": sil}
        (CACHE / f"{SS.key(pt, st, lab, a0)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(pt, st, lab, a0, [(r["pair"], r["replies"]) for r in res], "->", "SILENCED" if sil else "kept", flush=True)


def cmd_score():
    from benchmark.gold import sign_screen as SS
    SS.CACHE, SS.OUT = CACHE, OUT
    SS.cmd_score()
    c = collections.Counter(str(r.get("pair")) for f in CACHE.glob("*.json")
                            for r in json.loads(f.read_text(encoding="utf-8"))["stretches"])
    print("answer pairs (72 stretches):", dict(c))
    res = json.loads(OUT.read_text(encoding="utf-8"))
    res.update({"pairs": dict(c), "q1": Q1, "q2": Q2})
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"sanity": cmd_sanity, "run": cmd_run, "score": cmd_score}[cmd]()
