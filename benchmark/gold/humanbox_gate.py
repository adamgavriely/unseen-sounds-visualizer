"""Round 50 HUMAN-BOX (docs/prereg_round13_detector_push.md): ground HUMAN-2's named source phrase on HUMAN-2's own
frames with BOX-2's strict JSON prompt, crop it (BOX-2 crop rule), and ask the crop "Is this {phrase} making the {label}
sound right now?" in both letter orders. Stretch seen iff phrase named AND box found AND crop says yes.

    python benchmark/gold/humanbox_gate.py run      # GPU
    python benchmark/gold/humanbox_gate.py score    # CPU
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
from benchmark.gold import human2_gate as H2
from benchmark.gold.box_gate import gold_index, majority, parse_reprompt, to_pixels
from benchmark.gold.som_gate import crops

SRC = H2.OUT
OUT = G.OUT_DIR / "humanbox_Qwen38-27B"
CROPS = OUT / "crops"
SUMMARY = G.OUT_DIR / "humanbox_summary.json"
GROUND_Q = ("These frames (numbered 1..{n}) are from the moment a sound of {label} was heard. Reply ONLY with JSON, no "
            "other text, no tools: {{\"frame\": k, \"bbox_2d\": [x1, y1, x2, y2]}} around {phrase} on a 0-1000 grid of "
            "frame k (1000 = full width or height), or {{\"bbox_2d\": null}} if {phrase} is not visible.")
CROP_Q = ("This is a close-up cut from a video frame. Is this {phrase} making the {label} sound right now? "
          "Answer yes or no.")


def run(device="cuda"):
    config.use_v4("5")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    OUT.mkdir(parents=True, exist_ok=True); CROPS.mkdir(parents=True, exist_ok=True)
    mdl, proc = reason._load(H.MODEL, device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(SRC.glob("*.json")):
        if (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        for si, s in enumerate(d["sounds"]):
            for ti, st in enumerate(s["stretches"]):
                h = st.get("human") or {}
                cand = h.get("cand")
                rec = {"cand": cand, "status": "unnamed", "crop_yes": None, "seen": False}
                st["hbox"] = rec
                if not cand or p is None:
                    continue
                times = [t for _, t in h.get("frames", [])]
                frames = _sample_frames_at(p, times)
                if len(frames) < 2:
                    rec["status"] = "frames"; continue
                reply = reason._ask(mdl, proc, GROUND_Q.format(n=len(frames), label=s["label"], phrase=cand),
                                    images=frames, max_new=64)
                status, parsed = parse_reprompt(reply, len(frames))
                rec.update({"status": status, "reply": reply, "n_frames": len(frames)})
                if status == "box":
                    k, bb = parsed
                    img = frames[k - 1]; px = to_pixels(bb, img)
                    rec.update({"frame": k, "bbox_1000": bb, "bbox_px": [round(v, 1) for v in px], "size": list(img.size)})
                    cr = crops(img, [px])
                    if not cr:
                        rec["status"] = "degenerate"
                    else:
                        cr = cr[0]
                        tag = f"{f.stem}__{si}_{s['label'].replace(' ', '_').replace(',', '')}_{ti}"
                        cr.save(CROPS / f"{tag}_crop.jpg", quality=90)
                        yes = reason._ab(mdl, proc, CROP_Q.format(phrase=cand, label=s["label"]), "yes", "no", frames=[cr])
                        rec.update({"crop_yes": yes, "crop_file": f"{tag}_crop.jpg", "seen": yes is True})
                print(f.stem, s["label"], st["start"], repr(cand), rec["status"], rec.get("bbox_1000"), "crop", rec["crop_yes"],
                      "| maj", majority(st), "| h2", h.get("act"), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def seen_hb(st) -> bool:
    return bool((st.get("hbox") or {}).get("seen"))


def seen_hb_b(st) -> bool:
    votes = [st.get("name"), st.get("ab"), st.get("desc"), seen_hb(st)]
    t, f = sum(v is True for v in votes), sum(v is False for v in votes)
    return t > f if t != f else majority(st)


RULES = {"majority": majority, "HBOX_a": seen_hb, "HBOX_b": seen_hb_b}


def score():
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    assert len(files) == 49, f"expected 49 judge clips, got {len(files)}"
    res, flips = {}, {}
    cnt = {"stretches": 0, "named": 0, "box": 0, "null": 0, "unparsed": 0, "degenerate": 0, "frames": 0,
           "crop_yes": 0, "crop_no": 0, "crop_split": 0, "hbox_seen": 0, "maj_seen": 0}
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
                pred = base if rule == "majority" else H2.sound_majority(s["stretches"], fn)
                if g["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if rule == "HBOX_a":
                    for st in s["stretches"]:
                        r = st.get("hbox") or {}
                        cnt["stretches"] += 1; cnt["named"] += bool(r.get("cand"))
                        if r.get("status") in cnt:
                            cnt[r["status"]] += 1
                        if r.get("status") == "box" and "crop_yes" in r and r.get("crop_file"):
                            y = r["crop_yes"]
                            cnt["crop_yes" if y is True else "crop_no" if y is False else "crop_split"] += 1
                        cnt["hbox_seen"] += seen_hb(st); cnt["maj_seen"] += majority(st)
                if pred != base:
                    fl.append([f.stem, s["label"], s["start"], "seen" if g["seen"] else "NEEDED",
                               "silenced" if pred else "kept",
                               [[(st.get("hbox") or {}).get("cand"), (st.get("hbox") or {}).get("status"),
                                 (st.get("hbox") or {}).get("crop_yes")] for st in s["stretches"]],
                               [[st.get("name"), st.get("ab"), st.get("desc")] for st in s["stretches"]]])
        res[rule] = c; flips[rule] = fl
        if rule == "majority":
            assert c == H.BASE, f"base mismatch: {c} != {H.BASE}"
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in fl:
            print("   ", x)
    go = {}
    for rule in ("HBOX_a", "HBOX_b"):
        m = res[rule]
        go[rule] = ((m["seen_sil"] >= 19 and m["needed_kept"] >= 32) or (m["needed_kept"] >= 35 and m["seen_sil"] >= 15))
    res.update({"flips": flips, "stretch_counts": cnt, "go": go, "go_any": any(go.values())})
    print("stretches:", cnt)
    print("Round 50 HUMAN-BOX screen:", {k: ("GO" if v else "STOP") for k, v in go.items()})
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()
