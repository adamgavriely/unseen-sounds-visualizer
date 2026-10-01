"""Round 37 BOX (docs/prereg_round13_detector_push.md): the gate VLM's own bounding box checks its "seen".
For every stretch whose shipped majority (cached Qwen3.8-27B run, benchmark/gold/gate_gold/Qwen38-27B) says seen, the
same VLM is asked for a box around the object making the sound on the same 6 frames; the box is cropped and the VLM is
asked, label-free, whether the crop is that object (direct + a/b). A stretch flips to not-seen iff the box is "none" or
both crop answers are no. Truth = current gold (seen re-derived per sound). DEV judge clips only.

    python benchmark/gold/box_gate.py run      # GPU: Qwen3.8-27B
    python benchmark/gold/box_gate.py score    # CPU
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import gate_gold as G
from benchmark.gold.som_gate import crops
from src.labels import canonical

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "box_Qwen38-27B"
CROPS = OUT / "crops"
SUMMARY = G.OUT_DIR / "box_summary.json"
MODEL = "Qwen/Qwen3.8-27B"

OBJECT_OF = {
    "Bird": "a bird", "Water": "water (a river, the sea, waves or a tap)", "Rain": "rain falling", "Drum": "a drum",
    "Walk, footsteps": "a person's feet or legs stepping", "Bell": "a bell (the bell itself)",
    "Laughter": "a person laughing", "Motorcycle": "a motorcycle", "Machine gun": "a gun being fired",
    "Train": "a train", "Rustle": "something rustling (leaves, paper or cloth being moved)",
    "Vehicle": "a car or truck", "Crowd": "a crowd of people", "Chink, clink": "glasses or cutlery touching",
    "Glass": "glass", "Whack, thwack": "something being hit", "Whip": "a whip",
    "Air horn, truck horn": "a truck or vehicle horn", "Cellphone buzz, vibrating alert": "a mobile phone",
    "Siren": "an emergency vehicle", "Horse": "a horse",
}
BOX_Q = ("These frames (numbered 1..{n}) are from the moment a sound of {label} was heard. Find the object that is "
         "making that sound. Answer with JSON only: {{\"frame\": k, \"bbox_2d\": [x1, y1, x2, y2]}} on a 0-1000 grid of "
         "that frame (1000 = full width or height). If no such object is visible, answer exactly: none.")
CROP_PRE = "This is a close-up cut from a video frame."
DIRECT_Q = CROP_PRE + " Is this {obj}? Answer yes or no."
AB_Q = CROP_PRE + " What is it?"


def object_of(label: str) -> str:
    return OBJECT_OF.get(canonical(label)) or OBJECT_OF.get(label) or f"the thing that makes the sound of {label}"


def majority(st) -> bool:
    votes = [st.get("name"), st.get("ab"), st.get("desc")]
    return sum(v is True for v in votes) > sum(v is False for v in votes)


def parse_box(reply: str, n_frames: int):
    """-> ("none", None) | ("box", (k, [x1,y1,x2,y2] on 0-1000)) | ("unparsed", None)"""
    t = reply.strip()
    if re.match(r"^\W*none\b", t, re.I):
        return "none", None
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            k = int(j.get("frame", 0)); bb = [float(v) for v in j.get("bbox_2d", [])]
            if 1 <= k <= n_frames and len(bb) == 4 and bb[2] > bb[0] and bb[3] > bb[1]:
                return "box", (k, bb)
        except Exception:
            pass
    return "unparsed", None


def to_pixels(bb, img):
    W, H = img.size
    x0, y0, x1, y1 = bb
    box = [max(0.0, min(W, x0 / 1000 * W)), max(0.0, min(H, y0 / 1000 * H)),
           max(0.0, min(W, x1 / 1000 * W)), max(0.0, min(H, y1 / 1000 * H))]
    return box


def gold_index():
    idx = {}
    for name, stem, snds in G.gold_sounds():
        for s in snds:
            idx[(stem, s["label"], round(s["start"], 2))] = s
    return idx


def run(device="cuda"):
    config.use_v4("5")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True); CROPS.mkdir(parents=True, exist_ok=True)
    mdl, proc = reason._load(MODEL, device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", f.stem); continue
        for si, s in enumerate(d["sounds"]):
            obj = object_of(s["label"])
            for ti, st in enumerate(s["stretches"]):
                st["box"] = None
                if not majority(st):
                    continue
                lo, hi = st["start"] - 1.0, st["end"] + 1.0            # gate_gold.run_vlm frames
                times = [max(0.0, lo + (hi - lo) * t / 5) for t in range(6)]
                frames = _sample_frames_at(p, times)
                if not frames:
                    st["box"] = {"status": "unparsed", "reply": "", "note": "no frames"}; continue
                reply = reason._ask(mdl, proc, BOX_Q.format(n=len(frames), label=s["label"]), images=frames, max_new=64)
                status, parsed = parse_box(reply, len(frames))
                rec = {"status": status, "reply": reply, "obj": obj, "n_frames": len(frames)}
                if status == "box":
                    k, bb = parsed
                    img = frames[k - 1]
                    px = to_pixels(bb, img)
                    rec.update({"frame": k, "bbox_1000": bb, "bbox_px": [round(v, 1) for v in px], "size": list(img.size)})
                    cr = crops(img, [px])
                    if not cr:
                        rec["status"] = "unparsed"; rec["note"] = "degenerate box"
                    else:
                        cr = cr[0]
                        tag = f"{f.stem}__{si}_{s['label'].replace(' ', '_').replace(',', '')}_{ti}"
                        cr.save(CROPS / f"{tag}_crop.jpg", quality=90)
                        img.save(CROPS / f"{tag}_frame.jpg", quality=85)
                        direct = reason._ask(mdl, proc, DIRECT_Q.format(obj=obj), images=[cr], max_new=8)
                        ab = reason._ab(mdl, proc, AB_Q, obj, "something else", frames=[cr])
                        rec.update({"direct": direct, "direct_no": direct.strip().lower().startswith("n"),
                                    "ab": ab, "crop_file": f"{tag}_crop.jpg"})
                        rec["flip"] = bool(rec["direct_no"] and ab is False)
                if rec["status"] == "none":
                    rec["flip"] = True
                elif rec["status"] == "unparsed":
                    rec["flip"] = False
                st["box"] = rec
                print(f.stem, s["label"], st["start"], rec["status"], "flip" if rec.get("flip") else "-", "|", reply[:80],
                      "|", rec.get("direct"), rec.get("ab"), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def seen_box(st) -> bool:
    if not majority(st):
        return False
    b = st.get("box") or {}
    return not bool(b.get("flip"))


def score():
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    res = {}
    cnt = {"none": 0, "crop_no": 0, "unparsed": 0, "seen_kept": 0, "stretches": 0}
    flips = []
    for rule in ("majority", "BOX"):
        c = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0}
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            for s in d["sounds"]:
                g = gold.get((f.stem, s["label"], round(s["start"], 2)))
                if g is None:
                    print("NO GOLD MATCH", f.stem, s["label"], s["start"]); continue
                if g["importance"] < 2:
                    continue
                pred = all((majority if rule == "majority" else seen_box)(st) for st in s["stretches"])
                base = all(majority(st) for st in s["stretches"])
                if g["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if rule == "BOX":
                    for st in s["stretches"]:
                        b = st.get("box")
                        if not b:
                            continue
                        cnt["stretches"] += 1
                        if b["status"] == "none": cnt["none"] += 1
                        elif b["status"] == "unparsed": cnt["unparsed"] += 1
                        elif b.get("flip"): cnt["crop_no"] += 1
                        else: cnt["seen_kept"] += 1
                    if pred != base:
                        flips.append([f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED",
                                      "silenced" if pred else "kept",
                                      [(st["box"] or {}).get("status") for st in s["stretches"]],
                                      [(st["box"] or {}).get("reply", "")[:60] for st in s["stretches"]]])
        res[rule] = c
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
    for x in flips:
        print("   ", x)
    b, m = res["majority"], res["BOX"]
    go = (m["needed_kept"] - b["needed_kept"] >= 2) and (b["seen_sil"] - m["seen_sil"] <= 1)
    res.update({"flips": flips, "stretch_counts": cnt, "go": go})
    print("stretches:", cnt)
    print("Round 37 BOX screen:", "GO" if go else "STOP")
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
