"""Round 43 HUMAN (docs/prereg_round13_detector_push.md): the gate asks the annotator's own three steps AT the onset.
For every stretch of every gold sound on the 49 DEV judge clips, the shipped gate VLM (Qwen3.8-27B, greedy) sees one
context frame (onset - 1 s) and five frames densely at the onset (-0.3, -0.1, 0, +0.1, +0.3 s), each with a time caption,
and is asked (a) which object/animal/person in the frames could make the sound (open, "nothing" allowed) and, if one is
named, (b) whether it is visibly DOING the thing that makes the sound at the 0 s frame (a/b in both letter orders).
Stretch seen_HUMAN = (a) names something AND (b) yes in both orders. Variant (a): HUMAN replaces the shipped majority.
Variant (b): HUMAN is a fourth vote next to name/ab/desc; ties -> shipped majority. Truth = current gold.

    python benchmark/gold/human_gate.py run      # GPU: Qwen3.8-27B
    python benchmark/gold/human_gate.py score    # CPU
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
OUT = G.OUT_DIR / "human_Qwen38-27B"
SUMMARY = G.OUT_DIR / "human_summary.json"
MODEL = "Qwen/Qwen3.8-27B"
BASE = {"seen": 41, "seen_sil": 16, "needed": 38, "needed_kept": 33}       # Round 38 CF/BOX base, asserted in score
OFFSETS = [-0.3, -0.1, 0.0, 0.1, 0.3]
NAMED_FLIPS = ("b3_golf_course", "bell_miami", "as_church_bell_pJRAWkLM", "birds_forest", "b3_pet_shop",
               "ambient_nature_rainforest_2179", "ambient_nature_rainforest_7629", "b3_aviary_birds",
               "ambient_weather_storm_16200", "ambient_weather_storm_7200", "london_protest_01")

Q_NAME = ("A sound of {label} {verb} at the 0 s frame. Which object, animal or person in these frames could be making "
          "that sound? Answer with a short noun phrase of at most 5 words, or exactly: nothing.")
Q_ACT_CTX = ("{cand} was named as the likely source of the {label} that {verb} at the 0 s frame. "
             "Look at the 0 s frame and its neighbours.")
OPT_YES = ("at that moment {cand} is visibly DOING or UNDERGOING the thing that makes {label} (striking, swinging, flowing, "
           "flashing, running, calling...) -- the action itself shows in the frames")
OPT_NO = "{cand} is only present, or the action is not visible at that moment"


def frame_plan(t0: float):
    """[(caption, time)] in time order: one context frame, then five at the onset."""
    ctx = t0 - 1.0 if t0 >= 1.0 else t0 + 1.0
    plan = [("context, 1 s before" if t0 >= 1.0 else "context, 1 s after", ctx)]
    for o in OFFSETS:
        cap = "0 s: the moment" if o == 0 else f"{o:+.1f} s"
        plan.append((cap, max(0.0, t0 + o)))
    if t0 < 1.0:   # keep time order: the context frame comes after the dense ones
        plan = plan[1:] + plan[:1]
    return plan


def ask_seq(mdl, proc, parts, max_new: int = 48) -> str:
    """reason._ask with an interleaved content list (captions and images in one user turn); same template, greedy."""
    import torch
    content, images = [], []
    for p in parts:
        if isinstance(p, str):
            content.append({"type": "text", "text": p})
        else:
            content.append({"type": "image"}); images.append(p)
    try:
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images or None, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    reply = proc.batch_decode(new, skip_special_tokens=True)[0].strip()
    if new.shape[1] >= max_new and " " in reply:
        reply = reply.rsplit(" ", 1)[0]
    return reply


def ab_seq(mdl, proc, frames_parts, question, opt_yes, opt_no):
    """reason._ab logic (both letter orders, split -> None) on the captioned frames; returns (vote, raw replies)."""
    replies, votes = [], []
    for flip in (False, True):
        a, b = (opt_no, opt_yes) if flip else (opt_yes, opt_no)
        want = "b" if flip else "a"
        prompt = question + chr(10) + "(a) " + a + chr(10) + "(b) " + b + chr(10) + "Answer with the letter only."
        reply = ask_seq(mdl, proc, frames_parts + [prompt], max_new=6)
        replies.append(reply)
        votes.append(reply.strip().lower().lstrip("(")[:1] == want)
    return (True if all(votes) else (False if not any(votes) else None)), replies


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
            for i, st in enumerate(s["stretches"]):
                verb = "starts" if i == 0 else "is still going on"
                plan = frame_plan(float(st["start"]))
                frames = _sample_frames_at(p, [t for _, t in plan])
                h = {"frames": [[c, round(t, 2)] for c, t in plan], "n_frames": len(frames)}
                if len(frames) < 2:
                    h.update({"cand": None, "act": False, "act_replies": [], "seen": False, "note": "frames"})
                    st["human"] = h; continue
                parts = []
                for (cap, _), img in zip(plan, frames):
                    parts += [cap + ":", img]
                named = reason._clean_phrase(ask_seq(mdl, proc, parts + [Q_NAME.format(label=s["label"], verb=verb)],
                                                     max_new=24), max_words=5)
                low = named.lower()
                if not named or low.startswith(("nothing", "none", "no ", "not ")):
                    h.update({"cand": None, "named_raw": named, "act": False, "act_replies": [], "seen": False})
                else:
                    act, rep = ab_seq(mdl, proc, parts,
                                      Q_ACT_CTX.format(cand=named, label=s["label"], verb=verb),
                                      OPT_YES.format(cand=named, label=s["label"]), OPT_NO.format(cand=named))
                    h.update({"cand": named, "act": act, "act_replies": rep, "seen": act is True})
                st["human"] = h
                print(f.stem, s["label"], st["start"], "cand", h["cand"], "act", h["act"], h["act_replies"],
                      "| maj", majority(st), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def seen_human(st) -> bool:
    return bool((st.get("human") or {}).get("seen"))


def seen_b(st) -> bool:
    votes = [st.get("name"), st.get("ab"), st.get("desc"), seen_human(st)]
    t, f = sum(v is True for v in votes), sum(v is False for v in votes)
    return t > f if t != f else majority(st)


RULES = {"majority": majority, "HUMAN_a": seen_human, "HUMAN_b": seen_b}


def score():
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    assert len(files) == 49, f"expected 49 judge clips, got {len(files)}"
    res, flips, named = {}, {}, []
    cnt = {"stretches": 0, "nothing": 0, "act_yes": 0, "act_no": 0, "act_split": 0, "human_seen": 0, "maj_seen": 0}
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
                row = [f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED",
                       "silenced" if pred else "kept",
                       [[(st.get("human") or {}).get("cand"), (st.get("human") or {}).get("act")] for st in s["stretches"]],
                       [[st.get("name"), st.get("ab"), st.get("desc")] for st in s["stretches"]],
                       [(st.get("human") or {}).get("act_replies", []) for st in s["stretches"]]]
                if rule == "HUMAN_a":
                    for st in s["stretches"]:
                        h = st.get("human") or {}
                        cnt["stretches"] += 1; cnt["nothing"] += h.get("cand") is None
                        cnt["act_yes"] += h.get("act") is True; cnt["act_no"] += h.get("act") is False
                        cnt["act_split"] += h.get("cand") is not None and h.get("act") is None
                        cnt["human_seen"] += seen_human(st); cnt["maj_seen"] += majority(st)
                    if f.stem in NAMED_FLIPS:
                        named.append(row + ["base " + ("silenced" if base else "kept")])
                if pred != base:
                    fl.append(row)
        res[rule] = c; flips[rule] = fl
        if rule == "majority":
            assert c == BASE, f"base mismatch: {c} != {BASE}"
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in fl:
            print("   ", x)
    b = res["majority"]
    go = {}
    for rule in ("HUMAN_a", "HUMAN_b"):
        m = res[rule]
        go[rule] = ((m["seen_sil"] >= 19 and m["needed_kept"] >= 32) or (m["needed_kept"] >= 35 and m["seen_sil"] >= 15))
    res.update({"flips": flips, "named_set_HUMAN_a": named, "stretch_counts": cnt, "go": go, "go_any": any(go.values())})
    print("named set (HUMAN_a):")
    for x in named:
        print("   ", x)
    print("stretches:", cnt)
    print("Round 43 HUMAN screen:", {k: ("GO" if v else "STOP") for k, v in go.items()})
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
