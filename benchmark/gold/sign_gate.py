"""Round 54 SIGN step 1 (docs/prereg_round13_detector_push.md "Round 54 SIGN"): effect-centric, source-free gate vote used
as ADD-seen, screened on the gate-gold judge set (49 DEV clips, cached gate frames). For every stretch of every gold sound
the shipped gate VLM (Qwen3.8-27B, greedy, max_new 6) is asked on the gate's own 6 frames (stretch -1 s .. +1 s) the a/b
question STEM / OPT_YES / OPT_NO in both letter orders (as reason._ab); sign = True only if both orders pick OPT_YES.
Primary rule SIGN_add: sound silenced iff shipped majority silences it OR sign on EVERY stretch. Reported only:
SIGN_stretch (per stretch: majority OR sign, all stretches), SIGN_4th (sign as a 4th vote beside name/ab/desc, ties ->
majority). Truth = current gold (box_gate.gold_index).

    python benchmark/gold/sign_gate.py run      # GPU
    python benchmark/gold/sign_gate.py score    # CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import gate_gold as G
from benchmark.gold.box_gate import gold_index, majority

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "sign_Qwen38-27B"
SUMMARY = G.OUT_DIR / "sign_summary.json"
MODEL = "Qwen/Qwen3.8-27B"
BASE = {"seen": 41, "seen_sil": 16, "needed": 38, "needed_kept": 33}
NAMED = ("as_explosion_XJ8lc3I6", "tg_d088")

STEM = "These frames are from a video. Which is true?"
OPT_YES = "the visible SIGN of {label} — the effect or motion that this sound makes — is in these frames, even with no audio"
OPT_NO = "nothing in the frames shows that sound happening"


def ab_logged(reason, mdl, proc, question, opt_yes, opt_no, frames):
    """reason._ab (both orders, letter only, max_new 6) + the two raw replies"""
    replies, votes = [], []
    for flip in (False, True):
        a, b = (opt_no, opt_yes) if flip else (opt_yes, opt_no)
        want = "b" if flip else "a"
        reply = reason._ask(mdl, proc, question + chr(10) + "(a) " + a + chr(10) + "(b) " + b
                            + chr(10) + "Answer with the letter only.", images=frames, max_new=6)
        replies.append(reply)
        votes.append(reply.strip().lower().lstrip("(")[:1] == want)
    return (True if all(votes) else (False if not any(votes) else None)), replies


def gate_times(a, b):
    lo, hi = a - 1.0, b + 1.0
    return [max(0.0, lo + (hi - lo) * t / 5) for t in range(6)]


def run(device="cuda"):
    config.use_v4("5")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    mdl, proc = reason._load(MODEL, device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", f.stem); continue
        for s in d["sounds"]:
            for st in s["stretches"]:
                frames = _sample_frames_at(p, gate_times(st["start"], st["end"]))
                if len(frames) < 2:
                    st["sign"] = {"sign": False, "replies": [], "note": "frames", "n_frames": len(frames)}
                    continue
                v, rep = ab_logged(reason, mdl, proc, STEM, OPT_YES.format(label=s["label"]), OPT_NO, frames)
                st["sign"] = {"sign": v, "replies": rep, "n_frames": len(frames)}
                print(f.stem, s["label"], st["start"], "sign", v, rep, "| maj", majority(st), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def sign(st) -> bool:
    return (st.get("sign") or {}).get("sign") is True


def seen_4th(st) -> bool:
    votes = [st.get("name"), st.get("ab"), st.get("desc"), sign(st)]
    t, f = sum(v is True for v in votes), sum(v is False for v in votes)
    return t > f if t != f else majority(st)


def pred_of(rule, sts):
    base = all(majority(st) for st in sts)
    if rule == "majority":
        return base
    if rule == "SIGN_add":
        return base or all(sign(st) for st in sts)
    if rule == "SIGN_stretch":
        return all(majority(st) or sign(st) for st in sts)
    return all(seen_4th(st) for st in sts)


def score():
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    assert len(files) == 49, f"expected 49 judge clips, got {len(files)}"
    res, flips, per_clip = {}, {}, {}
    cnt = {"stretches": 0, "sign_yes": 0, "sign_no": 0, "sign_split": 0, "maj_seen": 0}
    for rule in ("majority", "SIGN_add", "SIGN_stretch", "SIGN_4th"):
        c = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0}
        fl = []
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            for s in d["sounds"]:
                g = gold.get((f.stem, s["label"], round(s["start"], 2)))
                if g is None:
                    print("NO GOLD MATCH", f.stem, s["label"], s["start"]); continue
                if g["importance"] < 2:
                    continue
                pred, base = pred_of(rule, s["stretches"]), pred_of("majority", s["stretches"])
                if g["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if rule == "SIGN_add":
                    for st in s["stretches"]:
                        v = (st.get("sign") or {}).get("sign")
                        cnt["stretches"] += 1; cnt["sign_yes"] += v is True; cnt["sign_no"] += v is False
                        cnt["sign_split"] += v is None; cnt["maj_seen"] += majority(st)
                    pc = per_clip.setdefault(f.stem, {"needed_lost": 0, "seen_added": 0, "needed": 0, "seen": 0})
                    pc["needed" if not g["seen"] else "seen"] += 1
                    if pred and not base:
                        pc["seen_added" if g["seen"] else "needed_lost"] += 1
                if pred != base:
                    fl.append([f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED", "silenced" if pred else "kept",
                               [(st.get("sign") or {}).get("sign") for st in s["stretches"]],
                               [[st.get("name"), st.get("ab"), st.get("desc")] for st in s["stretches"]],
                               [(st.get("sign") or {}).get("replies", []) for st in s["stretches"]]])
        res[rule] = c; flips[rule] = fl
        if rule == "majority":
            assert c == BASE, f"base mismatch: {c} != {BASE}"
        print(f"[{rule:12s}] seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in fl:
            print("   ", x)
    lost = {k: v for k, v in per_clip.items() if v["needed_lost"]}
    print("per-clip needed lost (SIGN_add):", lost)
    for n in NAMED:
        print("named", n, per_clip.get(n, "not in the gate-gold judge set (named in step 2)"))
    res.update({"flips": flips, "stretch_counts": cnt, "per_clip": per_clip, "needed_lost_clips": lost,
                "named": {n: per_clip.get(n, "not in gate-gold") for n in NAMED}, "stem": STEM, "opt_yes": OPT_YES,
                "opt_no": OPT_NO, "model": MODEL})
    print("stretches:", cnt)
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
