"""Round 15 amendment M (docs/prereg_round13_detector_push.md): Set-of-Mark crop vote for the stage-5 gate, DEV only.
For every (gold sound, stretch) of the cached Qwen3.8-27B gate run (benchmark/gold/gate_gold/Qwen38-27B), OWLv2 finds the
family's DETECT_QUERY object on the same 6 frames; the top-2 boxes per frame (score >= 0.1) are cropped and the gate VLM
is asked whether the crop shows that object. The crop vote joins the three cached votes under rule M.

    python benchmark/gold/som_gate.py run      # GPU: OWLv2 + Qwen3.8-27B, DEV judge clips only
    python benchmark/gold/som_gate.py score    # CPU
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
from src.labels import canonical

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "som_Qwen38-27B"
OWL = "google/owlv2-base-patch16-ensemble"
BOX_BAR, TOP, MARGIN, MIN_SIDE = 0.1, 2, 0.2, 224
Q = "This is a close-up cut from a video frame. Is it {phrase}? Answer yes or no."


def phrase_of(label):
    from src.stage2_video_understanding.owl import DETECT_QUERY
    return DETECT_QUERY.get(canonical(label)) or DETECT_QUERY.get(label)


def crops(img, boxes):
    out = []
    W, H = img.size
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        a, b = max(0, int(x0 - MARGIN * w)), max(0, int(y0 - MARGIN * h))
        c, d = min(W, int(x1 + MARGIN * w)), min(H, int(y1 + MARGIN * h))
        if c - a < 4 or d - b < 4:
            continue
        cr = img.crop((a, b, c, d))
        s = MIN_SIDE / min(cr.size)
        if s > 1:
            cr = cr.resize((int(cr.size[0] * s), int(cr.size[1] * s)))
        out.append(cr)
    return out


def run(device="cuda"):
    import torch
    config.use_v4("5")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage2_video_understanding.owl import _load as owl_load
    from src.stage5_cross_modal_analysis import reason
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    omdl, oproc = owl_load(OWL, device)
    vmdl, vproc = reason._load("Qwen/Qwen3.8-27B", device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", f.stem); continue
        for s in d["sounds"]:
            ph = phrase_of(s["label"])
            for st in s["stretches"]:
                st["crop"], st["crop_boxes"], st["crop_answers"] = None, [], []
                if ph is None:
                    continue
                lo, hi = st["start"] - 1.0, st["end"] + 1.0            # gate_gold.run_vlm frames
                times = [max(0.0, lo + (hi - lo) * t / 5) for t in range(6)]
                yes = False
                for img in _sample_frames_at(p, times):
                    inp = oproc(text=[[ph]], images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        o = omdl(**inp)
                    r = oproc.post_process_grounded_object_detection(
                        o, threshold=BOX_BAR, target_sizes=torch.tensor([[img.height, img.width]]).to(device))[0]
                    order = sorted(range(len(r["scores"])), key=lambda i: -float(r["scores"][i]))[:TOP]
                    bx = [[float(v) for v in r["boxes"][i]] for i in order]
                    st["crop_boxes"] += [[round(float(r["scores"][i]), 3)] + [round(v, 1) for v in r["boxes"][i].tolist()]
                                         for i in order]
                    for cr in crops(img, bx):
                        a = reason._ask(vmdl, vproc, Q.format(phrase=ph), images=[cr], max_new=8)
                        st["crop_answers"].append(a)
                        yes = yes or a.strip().lower().startswith("yes")
                st["crop"] = yes
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f.stem, sum(st.get("crop") is True for s in d["sounds"] for st in s["stretches"]), "crop yes", flush=True)


def seen_stretch(st, rule):
    votes = [st.get("name"), st.get("ab"), st.get("desc")]
    yes = sum(v is True for v in votes); no = sum(v is False for v in votes)
    maj = yes > no
    if rule == "majority":
        return maj
    if rule == "M":
        return maj or (st.get("crop") is True and yes >= 1)
    if rule == "M-any":
        return maj or st.get("crop") is True
    raise ValueError(rule)


def score():
    files = sorted(OUT.glob("*.json"))
    res = {}
    for rule in ("majority", "M", "M-any"):
        c = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0}
        flips = []
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            for s in d["sounds"]:
                if s["importance"] < 2:
                    continue
                pred = all(seen_stretch(st, rule) for st in s["stretches"])
                base = all(seen_stretch(st, "majority") for st in s["stretches"])
                if s["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if pred != base:
                    flips.append([f.stem, s["label"], s["start"], "seen" if s["seen"] else "NEEDED", "silenced" if pred else "kept"])
        c["seen_silenced"] = c["seen_sil"] / max(1, c["seen"]); c["needed_kept_r"] = c["needed_kept"] / max(1, c["needed"])
        res[rule] = {**c, "flips": flips}
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} ({c['seen_silenced']:.2f}) | "
              f"needed kept {c['needed_kept']}/{c['needed']} ({c['needed_kept_r']:.2f})")
        for x in flips:
            print("    ", x)
    m, b = res["M"], res["majority"]
    go = (m["seen_sil"] - b["seen_sil"] >= 3) and (b["needed_kept"] - m["needed_kept"] <= 1)
    res["go"] = go
    print("amendment M screen:", "GO (full DEV arm)" if go else "STOP")
    (G.OUT_DIR / "som_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
