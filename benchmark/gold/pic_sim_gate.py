"""Round 30 PIC-SIM (docs/prereg_round13_detector_push.md): gate picture-vs-frames similarity, DEV judge clips.
For each gold sound (importance >= 2) of the cached Qwen3.8-27B gate screen (benchmark/gold/gate_gold/Qwen38-27B, DEV judge
set JUDGE100), the pipeline's blind_a2i picture of the same family (display span start within [gold start - 0.5, + 1.0])
is compared with the gate's own stretch frames (6 per stretch, lo = a - 1, hi = b + 1, as gate_gold.run_vlm) by SigLIP-2
image-image cosine, max over all frames of all stretches of the sound. Sound-level rule on the majority verdict:
cos >= s_hi flips "not seen" -> seen; cos < s_lo flips seen -> not seen. No picture: majority verdict kept.
Pre-set pair s_hi = 0.80, s_lo = 0.30; GO iff seen silenced >= 20/43 and needed kept >= 31/36.

Picture source: the round-13 arm dirs (SHIP3+DV, then B0r) give the display spans, but hold placeholders only, so each
span's image is taken from the scored DEV render (protocol_blind_a2i_dev_monocap_v31, which B0r reproduces) by the
same event label and spec start.

    python benchmark/gold/pic_sim_gate.py run      # GPU
    python benchmark/gold/pic_sim_gate.py score
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import gate_gold as G
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as DCC

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "picsim_Qwen38-27B.json"
SUMMARY = G.OUT_DIR / "picsim_summary.json"
MODEL = "google/siglip2-so400m-patch14-384"
R13 = DCC.WORK / "r13"
ARMS = ["SHIP3+DV", "B0r"]
RENDER = DCC.scored_dir("blind_a2i")          # real pictures (diffusion) of the scored DEV render
EARLY, LATE = 0.5, 1.0
S_HI, S_LO = 0.80, 0.30
TOL = 0.01


def _spans(root: Path, stem: str):
    """display spans (label, start, spec dict) of one clip, as the compositor places them"""
    f = root / stem / "augmentations.json"
    if not f.exists():
        return None
    specs = json.loads(f.read_text(encoding="utf-8"))
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])]) for s in specs]
    m = root / stem / "media.json"
    dur = float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) if m.exists() else 0.0
    dur = dur or max((o.end for o in objs), default=0.0) + 5.0
    placed, _ = _assign_rows(_display_spans(objs, dur, require_image=True))
    return [(lab, float(a), sp) for _, lab, a, b, sp in placed]


def _render_image(stem: str, label: str, start: float):
    f = RENDER / stem / "augmentations.json"
    if not f.exists():
        return None
    for s in json.loads(f.read_text(encoding="utf-8")):
        ip = s.get("image_path")
        if (s.get("augment") and s["event_label"] == label and abs(float(s["start"]) - start) <= TOL and ip
                and s.get("backend") != "placeholder" and Path(ip).exists()):
            return ip
    return None


def _render_same_label(stem: str, label: str, start: float):
    """fallback: the render's real picture of the same event label in the same clip (nearest start). The image prompt is
    built from the label alone ("A clear, simple illustration of: <label>"), so it stands for the arm's picture"""
    f = RENDER / stem / "augmentations.json"
    if not f.exists():
        return None
    c = [(abs(float(s["start"]) - start), s["image_path"]) for s in json.loads(f.read_text(encoding="utf-8"))
         if s.get("augment") and s["event_label"] == label and s.get("image_path") and s.get("backend") != "placeholder"
         and Path(s["image_path"]).exists()]
    return min(c)[1] if c else None


def find_picture(stem: str, gold_label: str, gold_start: float):
    """(source, label, span start, image path) of the nearest same-family picture in the window, or None.
    source: "<arm>" = exact render image of the arm's span; "<arm>~label" = render image of the same label, other start;
    (label, start, None) = a matched span without any real image"""
    first = None
    for arm in ARMS:
        sp = _spans(R13 / f"{arm}_blind_a2i", stem)
        if sp is None:
            continue
        cands = sorted([(abs(a - gold_start), lab, a, spec) for lab, a, spec in sp
                        if S.same_family(lab, gold_label) and S.in_window(a, gold_start, EARLY, LATE)], key=lambda x: x[0])
        for _, lab, a, spec in cands:
            ip = _render_image(stem, spec.event_label, float(spec.start))
            if ip:
                return arm, lab, a, ip
        if cands and first is None:
            first = (arm, cands[0][1], cands[0][2], cands[0][3])
    if first:
        arm, lab, a, spec = first
        ip = _render_same_label(stem, spec.event_label, float(spec.start))
        return (arm + "~label" if ip else arm), lab, a, ip
    return None


def run():
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from src.stage2_video_understanding import _sample_frames_at
    judge = set(G.JUDGE100.read_text().split())
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

    def emb(imgs):
        inp = proc(images=imgs, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            e = model.get_image_features(pixel_values=inp["pixel_values"].half())
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        return torch.nn.functional.normalize(e.float(), dim=-1)
    rows = []
    for f in sorted(SRC.glob("*.json")):
        if f.stem not in judge:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(d["clip"])
        for s in d["sounds"]:
            if s["importance"] < 2:
                continue
            r = {"stem": f.stem, "label": s["label"], "start": s["start"], "end": s["end"], "seen": s["seen"],
                 "majority": G.decide(s["stretches"], "majority"), "cos": None, "picture": None}
            m = find_picture(f.stem, s["label"], s["start"])
            if m:
                r["picture"] = {"arm": m[0], "label": m[1], "start": m[2], "image": m[3]}
            if m and m[3]:
                frames = []
                for st in s["stretches"]:
                    lo, hi = st["start"] - 1.0, st["end"] + 1.0
                    frames += _sample_frames_at(p, [max(0.0, lo + (hi - lo) * t / 5) for t in range(6)])
                if frames:
                    pic = emb([Image.open(m[3]).convert("RGB")])
                    r["cos"] = float((emb(frames) @ pic.T).max())
                    r["n_frames"] = len(frames)
            rows.append(r)
            print(f.stem, s["label"], s["start"], "seen" if s["seen"] else "NEEDED", r["cos"], flush=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def counts(rows, hi, lo):
    c = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0, "flips": []}
    for r in rows:
        pred = r["majority"]
        if r["cos"] is not None:
            if not pred and r["cos"] >= hi:
                pred = True
            elif pred and r["cos"] < lo:
                pred = False
        if r["seen"]:
            c["seen"] += 1; c["seen_sil"] += pred
        else:
            c["needed"] += 1; c["needed_kept"] += not pred
        if pred != r["majority"]:
            c["flips"].append([r["stem"], r["label"], r["start"], "seen" if r["seen"] else "NEEDED",
                               "silenced" if pred else "kept", round(r["cos"], 3)])
    c["go"] = c["seen_sil"] >= 20 and c["needed_kept"] >= 31
    return c


def score():
    import numpy as np
    rows = json.loads(OUT.read_text(encoding="utf-8"))
    res = {"n": len(rows), "with_picture_span": sum(r["picture"] is not None for r in rows),
           "with_cos": sum(r["cos"] is not None for r in rows)}
    res["source"] = {}
    for r in rows:
        k = "none" if r["picture"] is None else (r["picture"]["arm"] + ("" if r["cos"] is not None else " (no image)"))
        res["source"][k] = res["source"].get(k, 0) + 1
    print(f"sounds {res['n']} | matched a picture span {res['with_picture_span']} | with a real image + cosine {res['with_cos']}"
          f" | by source {res['source']}")
    for grp, sel in (("seen", True), ("needed", False)):
        v = [r["cos"] for r in rows if r["seen"] == sel and r["cos"] is not None]
        q = [round(float(x), 3) for x in np.percentile(v, [0, 25, 50, 75, 100])] if v else []
        res[f"cos_{grp}"] = {"n": len(v), "min_q1_med_q3_max": q}
        print(f"cos {grp}: n {len(v)} min/q1/med/q3/max {q}")
    base = counts(rows, 9.0, -9.0)
    pre = counts(rows, S_HI, S_LO)
    res["majority"], res["preset"] = base, pre
    print(f"[majority] seen silenced {base['seen_sil']}/{base['seen']} | needed kept {base['needed_kept']}/{base['needed']}")
    print(f"[PIC-SIM 0.80/0.30] seen silenced {pre['seen_sil']}/{pre['seen']} | needed kept {pre['needed_kept']}/{pre['needed']}")
    for x in pre["flips"]:
        print("    ", x)
    print("PIC-SIM screen:", "GO" if pre["go"] else "STOP")
    grid = []                                    # exploratory only: GO is the pre-set pair
    for hi in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 9.0):
        for lo in (-9.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            if lo >= hi:
                continue
            c = counts(rows, hi, lo)
            grid.append({"hi": hi, "lo": lo, "seen_sil": c["seen_sil"], "needed_kept": c["needed_kept"], "go": c["go"],
                         "n_flips": len(c["flips"])})
    grid.sort(key=lambda g: (g["go"], g["seen_sil"] / max(1, base["seen"]) + g["needed_kept"] / max(1, base["needed"])), reverse=True)
    res["grid_exploratory"] = grid
    print("EXPLORATORY grid (top 5 by balanced):")
    for g in grid[:5]:
        print("    ", g)
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score()
