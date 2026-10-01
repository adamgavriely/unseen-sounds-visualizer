"""Round 38 E4 CF (docs/prereg_round13_detector_push.md): the gate VLM answers the GOLD's own two questions.
For every stretch of every gold sound on the DEV judge clips, the shipped gate VLM (Qwen3.8-27B, greedy) is asked, on
the gate's own 6 frames (stretch +/- 1 s), the gold re-check tool's Q1 and Q2 verbatim (sound name filled in), each as
an a/b yes/no in both letter orders (reason._ab). A question counts "yes" only if both orders agree on yes; a split
(None) or no = no. seen_CF = Q1 yes OR Q2 yes. Variant (a): CF replaces the shipped majority. Variant (b): CF is a
fourth vote next to name/ab/desc; ties -> shipped majority. Truth = current gold (seen re-derived per sound).

    python benchmark/gold/cf_gate.py run      # GPU: Qwen3.8-27B
    python benchmark/gold/cf_gate.py score    # CPU
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
OUT = G.OUT_DIR / "cf_Qwen38-27B"
SUMMARY = G.OUT_DIR / "cf_summary.json"
MODEL = "Qwen/Qwen3.8-27B"

CONTEXT = "These frames are from the moment a sound of {label} was heard."
Q1 = CONTEXT + " Is the thing that makes THIS sound on screen while you hear it? A look-alike counts as no."
Q2 = CONTEXT + " With the sound off, would a viewer already know this sound is happening?"
OPT_YES, OPT_NO = "yes", "no"


def ab_logged(reason, mdl, proc, question, frames):
    """reason._ab (both orders) + the two raw replies, for the record."""
    replies = []
    votes = []
    for flip in (False, True):
        a, b = (OPT_NO, OPT_YES) if flip else (OPT_YES, OPT_NO)
        want = "b" if flip else "a"
        reply = reason._ask(mdl, proc, question + chr(10) + "(a) " + a + chr(10) + "(b) " + b
                            + chr(10) + "Answer with the letter only.", images=frames, max_new=6)
        replies.append(reply)
        votes.append(reply.strip().lower().lstrip("(")[:1] == want)
    res = True if all(votes) else (False if not any(votes) else None)
    return res, replies


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
                lo, hi = st["start"] - 1.0, st["end"] + 1.0            # gate_gold.run_vlm frames
                times = [max(0.0, lo + (hi - lo) * t / 5) for t in range(6)]
                frames = _sample_frames_at(p, times)
                if not frames:
                    st["cf"] = {"q1": False, "q2": False, "note": "no frames", "q1_replies": [], "q2_replies": []}
                    continue
                q1, r1 = ab_logged(reason, mdl, proc, Q1.format(label=s["label"]), frames)
                q2, r2 = ab_logged(reason, mdl, proc, Q2.format(label=s["label"]), frames)
                st["cf"] = {"q1": q1, "q2": q2, "q1_replies": r1, "q2_replies": r2, "n_frames": len(frames)}
                print(f.stem, s["label"], st["start"], "q1", q1, r1, "q2", q2, r2, "| maj", majority(st), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def seen_cf(st) -> bool:
    c = st.get("cf") or {}
    return c.get("q1") is True or c.get("q2") is True


def seen_b(st) -> bool:
    votes = [st.get("name"), st.get("ab"), st.get("desc"), seen_cf(st)]
    t, f = sum(v is True for v in votes), sum(v is False for v in votes)
    return t > f if t != f else majority(st)


RULES = {"majority": majority, "CF_a": seen_cf, "CF_b": seen_b}


def score():
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    assert len(files) == 49, f"expected 49 judge clips, got {len(files)}"
    res, flips = {}, {}
    cnt = {"stretches": 0, "q1_yes": 0, "q2_yes": 0, "q1_split": 0, "q2_split": 0, "cf_seen": 0, "maj_seen": 0}
    for rule, fn in RULES.items():
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
                pred = all(fn(st) for st in s["stretches"])
                base = all(majority(st) for st in s["stretches"])
                if g["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if rule == "CF_a":
                    for st in s["stretches"]:
                        cf = st.get("cf") or {}
                        cnt["stretches"] += 1
                        cnt["q1_yes"] += cf.get("q1") is True; cnt["q2_yes"] += cf.get("q2") is True
                        cnt["q1_split"] += cf.get("q1") is None; cnt["q2_split"] += cf.get("q2") is None
                        cnt["cf_seen"] += seen_cf(st); cnt["maj_seen"] += majority(st)
                if pred != base:
                    fl.append([f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED",
                               "silenced" if pred else "kept",
                               [[(st.get("cf") or {}).get("q1"), (st.get("cf") or {}).get("q2")] for st in s["stretches"]],
                               [[st.get("name"), st.get("ab"), st.get("desc")] for st in s["stretches"]],
                               [(st.get("cf") or {}).get("q1_replies", []) + (st.get("cf") or {}).get("q2_replies", [])
                                for st in s["stretches"]]])
        res[rule] = c; flips[rule] = fl
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in fl:
            print("   ", x)
    b = res["majority"]
    go = {}
    for rule in ("CF_a", "CF_b"):
        m = res[rule]
        go[rule] = ((m["seen_sil"] >= b["seen_sil"] + 3 and m["needed_kept"] >= b["needed_kept"] - 1) or
                    (m["needed_kept"] >= b["needed_kept"] + 2 and m["seen_sil"] >= b["seen_sil"] - 1))
    res.update({"flips": flips, "stretch_counts": cnt, "go": go, "go_any": any(go.values())})
    print("stretches:", cnt)
    print("Round 38 E4 CF screen:", {k: ("GO" if v else "STOP") for k, v in go.items()})
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
