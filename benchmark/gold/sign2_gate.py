"""Round 54b SIGN-2 step 1 (docs/prereg_round13_detector_push.md "Round 54b SIGN-2"): sign_gate.py with two yes/no
questions of opposite polarity instead of the a/b format. sign = Q1 yes AND Q2 no. Stored in the same "sign" field, so
the rules and scoring are sign_gate.score with OUT / SUMMARY re-pointed.

    python benchmark/gold/sign2_gate.py run      # GPU (from ~/MscProj)
    python benchmark/gold/sign2_gate.py score    # CPU
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import gate_gold as G
from benchmark.gold import sign_gate as S

OUT = G.OUT_DIR / "sign2_Qwen38-27B"
SUMMARY = G.OUT_DIR / "sign2_summary.json"
from benchmark.gold.sign2_screen import Q1, Q2, yn, ask_sign    # one copy of the questions (sign2_screen.py)


def run(device="cuda"):
    config.use_v4("5")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    mdl, proc = reason._load(S.MODEL, device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(S.SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", f.stem); continue
        for s in d["sounds"]:
            for st in s["stretches"]:
                frames = _sample_frames_at(p, S.gate_times(st["start"], st["end"]))
                if len(frames) < 2:
                    st["sign"] = {"sign": False, "replies": [], "pair": None, "note": "frames", "n_frames": len(frames)}
                    continue
                v, rep, pair = ask_sign(reason, mdl, proc, s["label"], frames)
                st["sign"] = {"sign": v, "replies": rep, "pair": list(pair), "n_frames": len(frames)}
                print(f.stem, s["label"], st["start"], "sign", v, rep, flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def score():
    S.OUT, S.SUMMARY = OUT, SUMMARY
    S.score()
    c = collections.Counter()
    for f in OUT.glob("*.json"):
        for s in json.loads(f.read_text(encoding="utf-8"))["sounds"]:
            for st in s["stretches"]:
                c[str((st.get("sign") or {}).get("pair"))] += 1
    print("answer pairs (all cached stretches):", dict(c))
    res = json.loads(SUMMARY.read_text(encoding="utf-8"))
    res.update({"pairs": dict(c), "q1": Q1, "q2": Q2})
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
