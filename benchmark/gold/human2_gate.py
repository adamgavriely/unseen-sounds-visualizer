"""Round 43b HUMAN-2 (docs/prereg_round13_detector_push.md): Round 43 with a strict-format open question (raw reply stored)
and a majority-of-stretches sound verdict. Everything else as benchmark/gold/human_gate.py (frames, captions, (b) wording).

    python benchmark/gold/human2_gate.py run      # GPU
    python benchmark/gold/human2_gate.py score    # CPU
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
from benchmark.gold import human_gate as H
from benchmark.gold.box_gate import gold_index, majority

OUT = G.OUT_DIR / "human2_Qwen38-27B"
SUMMARY = G.OUT_DIR / "human2_summary.json"
Q_NAME = ("Answer with ONLY a short noun phrase naming the most likely visible source of the {label} sound in these frames "
          "(e.g. 'the golfer', 'the waterfall'), or exactly 'none' if no plausible source is visible.")


def run(device="cuda"):
    config.use_v4("5")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    mdl, proc = reason._load(H.MODEL, device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(H.SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", f.stem); continue
        for s in d["sounds"]:
            for i, st in enumerate(s["stretches"]):
                verb = "starts" if i == 0 else "is still going on"
                plan = H.frame_plan(float(st["start"]))
                frames = _sample_frames_at(p, [t for _, t in plan])
                h = {"frames": [[c, round(t, 2)] for c, t in plan], "n_frames": len(frames)}
                if len(frames) < 2:
                    h.update({"cand": None, "raw": "", "act": False, "act_replies": [], "seen": False, "note": "frames"})
                    st["human"] = h; continue
                parts = []
                for (cap, _), img in zip(plan, frames):
                    parts += [cap + ":", img]
                raw = H.ask_seq(mdl, proc, parts + [Q_NAME.format(label=s["label"])], max_new=24)
                named = reason._clean_phrase(raw, max_words=5)
                low = named.lower()
                h["raw"] = raw
                if not named or low.startswith(("nothing", "none", "no ", "not ")):
                    h.update({"cand": None, "act": False, "act_replies": [], "seen": False})
                else:
                    act, rep = H.ab_seq(mdl, proc, parts, H.Q_ACT_CTX.format(cand=named, label=s["label"], verb=verb),
                                        H.OPT_YES.format(cand=named, label=s["label"]), H.OPT_NO.format(cand=named))
                    h.update({"cand": named, "act": act, "act_replies": rep, "seen": act is True})
                st["human"] = h
                print(f.stem, s["label"], st["start"], "raw", repr(raw[:60]), "act", h["act"], h["act_replies"],
                      "| maj", majority(st), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def sound_majority(stretches, fn) -> bool:
    """Round 43b: the sound is seen iff MORE THAN HALF of its stretches are seen (Round 43 / shipped: every stretch)."""
    return sum(bool(fn(st)) for st in stretches) * 2 > len(stretches)


RULES = {"majority": majority, "HUMAN2_a": H.seen_human, "HUMAN2_b": H.seen_b}


def score():
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    assert len(files) == 49, f"expected 49 judge clips, got {len(files)}"
    res, flips, named = {}, {}, []
    cnt = {"stretches": 0, "none": 0, "raw_over5": 0, "act_yes": 0, "act_no": 0, "act_split": 0, "human_seen": 0, "maj_seen": 0}
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
                base = all(majority(st) for st in s["stretches"])
                pred = base if rule == "majority" else sound_majority(s["stretches"], fn)
                if g["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                row = [f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED", "silenced" if pred else "kept",
                       [[(st.get("human") or {}).get("raw"), (st.get("human") or {}).get("act")] for st in s["stretches"]],
                       [[st.get("name"), st.get("ab"), st.get("desc")] for st in s["stretches"]],
                       [(st.get("human") or {}).get("act_replies", []) for st in s["stretches"]]]
                if rule == "HUMAN2_a":
                    for st in s["stretches"]:
                        h = st.get("human") or {}
                        cnt["stretches"] += 1; cnt["none"] += h.get("cand") is None
                        cnt["raw_over5"] += len((h.get("raw") or "").split()) > 5
                        cnt["act_yes"] += h.get("act") is True; cnt["act_no"] += h.get("act") is False
                        cnt["act_split"] += h.get("cand") is not None and h.get("act") is None
                        cnt["human_seen"] += H.seen_human(st); cnt["maj_seen"] += majority(st)
                    if f.stem in H.NAMED_FLIPS:
                        named.append(row + ["base " + ("silenced" if base else "kept")])
                if pred != base:
                    fl.append(row)
        res[rule] = c; flips[rule] = fl
        if rule == "majority":
            assert c == H.BASE, f"base mismatch: {c} != {H.BASE}"
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in fl:
            print("   ", x)
    go = {}
    for rule in ("HUMAN2_a", "HUMAN2_b"):
        m = res[rule]
        go[rule] = ((m["seen_sil"] >= 19 and m["needed_kept"] >= 32) or (m["needed_kept"] >= 35 and m["seen_sil"] >= 15))
    res.update({"flips": flips, "named_set_HUMAN2_a": named, "stretch_counts": cnt, "go": go, "go_any": any(go.values())})
    print("named set (HUMAN2_a):")
    for x in named:
        print("   ", x)
    print("stretches:", cnt)
    print("Round 43b HUMAN-2 screen:", {k: ("GO" if v else "STOP") for k, v in go.items()})
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
