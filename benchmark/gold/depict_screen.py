"""Round 57 DEPICT-EVENT + LOOK-ALIKE (docs/prereg_round13_detector_push.md "Round 57 DEPICT-EVENT"): on the same 43 saved
SHIP8+MD3 merged-DEV drawn specs as sign_screen.py, drop a picture iff (a) Qwen3.8-27B says its own depicted event (spec
`subject`) is visibly happening at picture start +-1 s (E1 yes AND opposite twin E2 no) AND (b) text-only, the gate's own
named maker B (any B != "nothing") could be mistaken for the label A in both orders. Re-placement and scoring as
sign_screen.cmd_score (CACHE / OUT re-pointed).

    TG_ARMS=SHIP8+MD3 python benchmark/gold/depict_screen.py specs    # CPU: harness check (sign_screen specs)
    TG_ARMS=SHIP8+MD3 python benchmark/gold/depict_screen.py sanity   # GPU: 10 specs; exit 3 if >= 9 share an (E1, E2) pair
    TG_ARMS=SHIP8+MD3 python benchmark/gold/depict_screen.py run      # GPU -> benchmark/gold/depict_pics/
    TG_ARMS=SHIP8+MD3 python benchmark/gold/depict_screen.py score    # CPU -> benchmark/gold/depict_screen.json
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

CACHE = _ROOT / "benchmark" / "gold" / "depict_pics"
OUT = _ROOT / "benchmark" / "gold" / "depict_screen.json"
SANITY = _ROOT / "benchmark" / "gold" / "depict_sanity.json"
E1 = 'Is "{event}" visibly happening in these frames? Answer yes or no.'
E2 = 'Do these frames show no "{event}" happening? Answer yes or no.'
L1 = "Could the sound of {B} be mistaken for the sound of {A}? Answer yes or no."
L2 = "Could the sound of {A} be mistaken for the sound of {B}? Answer yes or no."


def yn(reply: str):
    t = reply.strip().lower().lstrip("*\"'([ ").split()
    w = t[0].strip(".,!*\"')]") if t else ""
    return w if w in ("yes", "no") else None


def specs():
    """[(part, stem, label, start, subject, [B...])] in sign_screen.asked order (the 43 drawn specs)"""
    from benchmark.gold import sign_screen as SS
    out = []
    A = SS.asked(SS.parts())
    assert len(A) == 43 and sum(len(a[5]) for a in A) == 72, (len(A), sum(len(a[5]) for a in A))
    cache = {}
    for pt, st, w, lab, a0, sts, src in A:
        if (pt, st) not in cache:
            vf = w / st / "gate_votes.json"
            cache[(pt, st)] = (json.loads((w / st / "augmentations.json").read_text(encoding="utf-8")),
                               json.loads(vf.read_text(encoding="utf-8")) if vf.exists() else [])
        sp, votes = cache[(pt, st)]
        s = [x for x in sp if x["event_label"] == lab and abs(float(x["start"]) - a0) < 0.011 and x.get("augment")][0]
        subj = (s.get("subject") or s.get("image_prompt") or "").strip()
        bs = []
        for v in votes:
            if v["label"] == lab and abs(float(v["start"]) - a0) < 0.011:
                n = str(v.get("named") or "").strip()
                if n and n.lower() != "nothing" and n not in bs:
                    bs.append(n)
        out.append((pt, st, lab, a0, subj, bs))
    return out


def _frames(vp, t0):
    from src.stage2_video_understanding import _sample_frames_at
    lo, hi = t0 - 1.0, t0 + 1.0
    return _sample_frames_at(vp, [lo + (hi - lo) * t / 5 for t in range(6)])     # 6 frames, unclamped, as 54b


def ask(reason, mdl, proc, pt, st, lab, a0, subj, bs):
    from benchmark.gold.imp_v_screen import clip_file
    vp = Path(clip_file(pt, st))
    rec = {"part": pt, "clip": st, "label": lab, "start": a0, "subject": subj, "makers": bs, "clip_file": str(vp)}
    fr = _frames(vp, a0) if subj else []
    if subj and len(fr) >= 2:
        r1 = reason._ask(mdl, proc, E1.format(event=subj), images=fr, max_new=4)
        r2 = reason._ask(mdl, proc, E2.format(event=subj), images=fr, max_new=4)
        pair = [yn(r1), yn(r2)]
        rec.update({"event_prompts": [E1.format(event=subj), E2.format(event=subj)], "event_replies": [r1, r2],
                    "pair": pair, "event": pair == ["yes", "no"], "n_frames": len(fr)})
    else:
        rec.update({"event_replies": [], "pair": None, "event": False, "n_frames": len(fr), "note": "no subject/frames"})
    A = lab.lower()
    look = []
    for B in bs:
        q1, q2 = L1.format(B=B, A=A), L2.format(B=B, A=A)
        x1 = reason._ask(mdl, proc, q1, images=None, max_new=4)
        x2 = reason._ask(mdl, proc, q2, images=None, max_new=4)
        look.append({"B": B, "prompts": [q1, q2], "replies": [x1, x2], "answers": [yn(x1), yn(x2)],
                     "yes": yn(x1) == "yes" and yn(x2) == "yes"})
    rec["lookalike"] = look
    rec["lookalike_yes"] = any(l["yes"] for l in look)
    rec["silenced"] = bool(rec["event"] and rec["lookalike_yes"])
    return rec


def _load():
    from src.stage5_cross_modal_analysis import reason
    from benchmark.gold import sign_screen as SS
    return reason, reason._load(SS.MODEL, "cuda")


def cmd_specs():
    from benchmark.gold import sign_screen as SS
    SS.cmd_specs()


def cmd_sanity():
    S = specs()
    pick = [S[i] for i in range(0, 37, 4)]
    reason, (mdl, proc) = _load()
    rows = []
    for x in pick:
        r = ask(reason, mdl, proc, *x)
        rows.append(r)
        print("SANITY", r["part"], r["clip"], r["label"], r["start"], repr(r["subject"]), r.get("event_prompts"),
              r["event_replies"], r["pair"], "| lookalike", [(l["B"], l["replies"]) for l in r["lookalike"]], flush=True)
    c = collections.Counter(str(r["pair"]) for r in rows)
    stop = c.most_common(1)[0][1] >= 9
    print("SANITY pairs", dict(c), "->", "STOP (untestable with this VLM)" if stop else "GO", flush=True)
    SANITY.write_text(json.dumps({"rows": rows, "pairs": dict(c), "stop": stop}, indent=1), encoding="utf-8")
    sys.exit(3 if stop else 0)


def cmd_run():
    from benchmark.gold import sign_screen as SS
    CACHE.mkdir(parents=True, exist_ok=True)
    S = [x for x in specs() if not (CACHE / f"{SS.key(x[0], x[1], x[2], x[3])}.json").exists()]
    print(len(S), "specs to ask", flush=True)
    if not S:
        return
    reason, (mdl, proc) = _load()
    for x in S:
        r = ask(reason, mdl, proc, *x)
        (CACHE / f"{SS.key(x[0], x[1], x[2], x[3])}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(r["part"], r["clip"], r["label"], r["start"], repr(r["subject"]), r["pair"], r["event_replies"], "| look",
              [(l["B"], l["answers"]) for l in r["lookalike"]], "->", "DROP" if r["silenced"] else "kept", flush=True)


def cmd_score():
    from benchmark.gold import sign_screen as SS
    SS.CACHE, SS.OUT = CACHE, OUT
    SS.cmd_score()
    R = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(CACHE.glob("*.json"))]
    pairs = collections.Counter(str(r["pair"]) for r in R)
    cov = sum(1 for r in R if r["makers"])
    look = collections.Counter(str(l["answers"]) for r in R for l in r["lookalike"])
    summary = {"event_pairs": dict(pairs), "event_yes": sum(r["event"] for r in R), "lookalike_coverage": cov,
               "lookalike_answer_pairs": dict(look), "lookalike_yes": sum(r["lookalike_yes"] for r in R),
               "dropped": [[r["part"], r["clip"], r["label"], r["start"], r["subject"],
                            [l["B"] for l in r["lookalike"] if l["yes"]]] for r in R if r["silenced"]],
               "event_yes_specs": [[r["clip"], r["label"], r["start"], r["subject"]] for r in R if r["event"]],
               "prompts": {"E1": E1, "E2": E2, "L1": L1, "L2": L2}}
    print(json.dumps(summary, indent=1))
    res = json.loads(OUT.read_text(encoding="utf-8"))
    res.update(summary)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"specs": cmd_specs, "sanity": cmd_sanity, "run": cmd_run, "score": cmd_score}[cmd]()
