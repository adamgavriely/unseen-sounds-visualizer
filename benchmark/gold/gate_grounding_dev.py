"""Grounded visibility gate, DEV only (28 Sept 2026). The stage-5 gate silences a sound when the VLM's
majority vote says its source is "seen" in a <=5-s stretch; it sometimes says so from an assumption
("church bell" with no bell on screen, "macaws", "a small bird"). Rule under test: for every stretch the
gate calls seen, the same VLM (Qwen3.8-27B, config.use_shipped) names the object MAKING the sound (one
noun, from the gate's 6 frames) and boxes it in each frame; a checker (A: OWLv2, B: SAM 3, the noun as
text prompt) must find that noun inside the VLM box (IoU >= 0.3, or the checker box >= 90 % inside it)
with score >= x in >= 2 of the 6 frames, else the stretch flips to "not seen". It can only flip seen ->
not seen. x in {0.1, 0.2, 0.3} per checker. Nothing is run on TEST: the clip list is the DEV subset and
the script refuses any other stem.

    python benchmark/gold/gate_grounding_dev.py collect      # GPU: raw evidence -> data/work/gate_grounding_dev/
    python benchmark/gold/gate_grounding_dev.py score        # CPU: verdicts, metrics, DEV score -> gate_grounding_dev.json
    python benchmark/gold/gate_grounding_dev.py overlays     # CPU: frames with boxes -> docs/gate_grounding/*.png

The collect step stores only evidence (noun, VLM boxes in frame pixels, every checker detection >= 0.05
with its box), so every rule and threshold is re-decided on CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S

TAG = "dev_monocap_v31"
PROPOSED = _ROOT / "data" / "work" / f"protocol_proposed_{TAG}"
BLIND = _ROOT / "data" / "work" / f"protocol_blind_a2i_{TAG}"
import os
OUTDIR = Path(os.environ.get("GG_OUT") or _ROOT / "data" / "work" / "gate_grounding_dev")
RAW = OUTDIR / "raw.jsonl"
RESULT = Path(os.environ.get("GG_RESULT") or _ROOT / "benchmark" / "gold" / "gate_grounding_dev.json")
PNGDIR = _ROOT / "docs" / "gate_grounding"
XS = (0.1, 0.2, 0.3)
IOU_MIN, INSIDE_MIN, MIN_FRAMES = 0.3, 0.9, 2
DET_FLOOR = 0.05

NOUN_PROMPT = (
    "These frames are from the moment a sound of {label} was heard. Which visible object in the frames is "
    "making that sound? Name the object itself (for example: bell, bird, car, dog, waterfall), not the building, "
    "the place or the scene around it. Answer with one or two words only. If no object making that sound is "
    "visible, answer none.")
BOX_PROMPT = (
    "Locate the {noun} that is making the {label} sound in this image. Output its bounding box as JSON in the "
    "form [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"{noun}\"}}]. If there is more than one, list each. "
    "If it is not visible in this image, output [].")


def dev_stems():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    subs = S.subsets_of(gold)
    dev = sorted(subs["dev"])
    assert not (set(dev) & subs["test"]), "DEV list overlaps TEST"
    return gold, dev, subs


def seen_stretches(dev, subs):
    """every (clip, sound, stretch) the scored gate called seen"""
    out = []
    for st in dev:
        assert st not in subs["test"], st
        f = PROPOSED / st / "gate_votes.json"
        if not f.exists():
            continue
        for k, v in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            if v.get("seen"):
                out.append({"clip": st, "k": k, "label": v["label"], "start": v["start"], "end": v["end"],
                            "stretch": v["stretch"], "named": v.get("named", "")})
    return out


# ----------------------------------------------------------------------------- GPU: evidence
def _parse_boxes(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        arr = [[float(x) for x in b] for b in re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)]
        return [b for b in arr if len(b) == 4]
    if isinstance(arr, dict):
        arr = [arr]
    boxes = []
    for a in arr if isinstance(arr, list) else []:
        b = a.get("bbox_2d") if isinstance(a, dict) else a
        if isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
            boxes.append([float(x) for x in b])
    return boxes


def _vlm_boxes(img, prompt, mdl, proc):
    from src.stage5_cross_modal_analysis.reason import _ask
    txt = _ask(mdl, proc, prompt, images=[img], max_new=256)
    return txt, _parse_boxes(txt)


def _to_pixels(boxes, img, mode):
    W, H = img.size
    out = []
    for x1, y1, x2, y2 in boxes:
        if mode == "rel1000":
            x1, x2, y1, y2 = x1 * W / 1000, x2 * W / 1000, y1 * H / 1000, y2 * H / 1000
        out.append([max(0.0, min(W, x1)), max(0.0, min(H, y1)), max(0.0, min(W, x2)), max(0.0, min(H, y2))])
    return out


def calibrate(mdl, proc):
    """Which coordinate frame does the model answer in? A non-square synthetic image with a known box."""
    from PIL import Image, ImageDraw
    W, H, box = 640, 360, (400, 50, 560, 170)
    img = Image.new("RGB", (W, H), (150, 150, 150))
    ImageDraw.Draw(img).rectangle(box, fill=(220, 20, 20))
    txt, bx = _vlm_boxes(img, BOX_PROMPT.format(noun="red rectangle", label="test"), mdl, proc)
    rel = (box[0] * 1000 / W, box[1] * 1000 / H, box[2] * 1000 / W, box[3] * 1000 / H)
    if not bx:
        return {"mode": "rel1000", "raw": txt, "note": "no box parsed; assumed 0-1000"}
    b = bx[0]
    e_pix = sum(abs(p - q) for p, q in zip(b, box))
    e_rel = sum(abs(p - q) for p, q in zip(b, rel))
    return {"mode": "pixel" if e_pix < e_rel else "rel1000", "raw": txt, "err_pixel": e_pix, "err_rel1000": e_rel,
            "truth_pixel": box, "truth_rel1000": rel}


def _owl_dets(img, noun, device):
    import torch
    from src.stage2_video_understanding.owl import _load
    mdl, proc = _load(config.OWL_MODEL, device)
    q = noun if noun.lower().startswith(("a ", "an ", "the ")) else "a " + noun
    inputs = proc(text=[[q]], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**inputs)
    s = max(img.height, img.width)       # OWLv2 pads to a square (bottom/right): boxes are in the padded frame
    res = proc.post_process_grounded_object_detection(out, threshold=DET_FLOOR,
                                                      target_sizes=torch.tensor([[s, s]]).to(device))[0]
    return [{"score": float(sc), "box": [float(v) for v in bx]} for sc, bx in zip(res["scores"], res["boxes"])]


def _sam_dets(img, noun, device):
    import torch
    from src.stage2_video_understanding.sam3 import _load
    mdl, proc = _load(config.SAM3_MODEL, device)
    inputs = proc(images=[img], text=[noun], return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(mdl.dtype)
    with torch.no_grad():
        out = mdl(**inputs)
    r = proc.post_process_instance_segmentation(out, threshold=DET_FLOOR, mask_threshold=0.5,
                                                target_sizes=inputs.get("original_sizes").tolist())[0]
    dets = []
    boxes = r.get("boxes")
    for i, sc in enumerate(r["scores"]):
        if boxes is not None and len(boxes) > i:
            b = [float(v) for v in boxes[i]]
        else:
            m = r["masks"][i]
            ys, xs = m.nonzero(as_tuple=True)
            if not len(xs):
                continue
            b = [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]
        dets.append({"score": float(sc), "box": b})
    return dets


def collect(args):
    import torch
    from PIL import Image
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis.reason import _load, _clean_phrase, _ask
    config.use_shipped()
    print("[cfg] VLM", config.VLM_MODEL, "thinking", getattr(config, "VLM_THINKING", False), flush=True)
    gold, dev, subs = dev_stems()
    items = seen_stretches(dev, subs)
    print(f"[dev] {len(dev)} clips, {len(items)} seen stretches", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "frames").mkdir(exist_ok=True)
    done = {}
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["clip"], r["k"])] = r
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    # phase 1: VLM noun + boxes (skipped for stretches already in raw.jsonl)
    todo = [it for it in items if (it["clip"], it["k"]) not in done]
    if todo:
        mdl, proc = _load(config.VLM_MODEL, dev_)
        cal = calibrate(mdl, proc)
        print("[calibration]", json.dumps(cal), flush=True)
        (OUTDIR / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
        for it in todo:
            media = json.loads((PROPOSED / it["clip"] / "media.json").read_text(encoding="utf-8"))
            a, b = it["stretch"]
            n, lo, hi = 6, a - 1.0, b + 1.0                       # exactly the gate's frames (reason.py)
            times = [lo + (hi - lo) * t / (n - 1) for t in range(n)]
            frames = _sample_frames_at(Path(media["video_path"]), times)
            fpaths = []
            for i, f in enumerate(frames):
                p = OUTDIR / "frames" / f"{it['clip']}__{it['k']}__{i}.png"
                f.save(p)
                fpaths.append(str(p.relative_to(_ROOT)))
            raw_noun = _ask(mdl, proc, NOUN_PROMPT.format(label=it["label"]), images=frames, max_new=16)
            noun = _clean_phrase(raw_noun, max_words=3)
            low = noun.lower()
            if not noun or low.startswith(("none", "nothing", "no ", "not ")):
                noun = ""
            per = []
            for f in frames:
                if not noun:
                    per.append({"vlm_raw": "", "vlm_boxes": []})
                    continue
                txt, bx = _vlm_boxes(f, BOX_PROMPT.format(noun=noun, label=it["label"]), mdl, proc)
                per.append({"vlm_raw": txt[:300], "vlm_boxes": _to_pixels(bx, f, cal["mode"]),
                            "size": list(f.size)})
            rec = {**it, "times": times, "frames": fpaths, "noun_raw": raw_noun, "noun": noun, "per_frame": per,
                   "coord_mode": cal["mode"]}
            done[(it["clip"], it["k"])] = rec
            with RAW.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            nb = sum(1 for p in per if p["vlm_boxes"])
            print(f"[vlm] {it['clip'][:34]:34s} {it['label'][:18]:18s} {a:5.1f}-{b:5.1f} named={it['named'][:18]!r:20s} "
                  f"noun={noun!r} frames-with-box {nb}/{len(per)}", flush=True)
        from src.stage5_cross_modal_analysis import reason as R
        R.unload()
        torch.cuda.empty_cache()

    # phase 2/3: checkers, each on its own so one failing does not lose the other
    for name, fn in (("owlv2", _owl_dets), ("sam3", _sam_dets)):
        try:
            for key, rec in done.items():
                for i, pf in enumerate(rec["per_frame"]):
                    if name in pf:
                        continue
                    if not rec["noun"] or i >= len(rec["frames"]):
                        pf[name] = []
                        continue
                    img = Image.open(_ROOT / rec["frames"][i]).convert("RGB")
                    pf[name] = fn(img, rec["noun"], dev_)
                best = max((d["score"] for pf in rec["per_frame"] for d in pf.get(name, [])), default=0.0)
                print(f"[{name}] {rec['clip'][:34]:34s} {rec['label'][:18]:18s} noun={rec['noun']!r} best {best:.2f}", flush=True)
            RAW.write_text("".join(json.dumps(r) + "\n" for r in done.values()), encoding="utf-8")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("->", RAW)


# ----------------------------------------------------------------------------- CPU: verdicts and scores
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _inside(d, v):
    """share of the checker box d that lies inside the VLM box v"""
    ix = max(0.0, min(d[2], v[2]) - max(d[0], v[0])); iy = max(0.0, min(d[3], v[3]) - max(d[1], v[1]))
    ad = (d[2] - d[0]) * (d[3] - d[1])
    return ix * iy / ad if ad > 0 else 0.0


def frame_ok(pf, checker, x):
    for v in pf.get("vlm_boxes", []):
        for d in pf.get(checker, []):
            if d["score"] >= x and (_iou(v, d["box"]) >= IOU_MIN or _inside(d["box"], v) >= INSIDE_MIN):
                return True
    return False


def stays_seen(rec, checker, x):
    return sum(1 for pf in rec["per_frame"] if frame_ok(pf, checker, x)) >= MIN_FRAMES


def category(rec, gold):
    """gold class of a seen stretch: 'visible' (a same-family gold sound that is visible/obvious overlaps it),
    'needed' (only needed ones), 'mixed', or 'nogold' (no same-family gold sound there: a detector error)"""
    a, b = rec["stretch"]
    m = [g for g in gold[rec["clip"]] if S.same_family(rec["label"], g["label"]) and g["start"] < b + 1 and g["end"] > a - 1]
    vis = [g for g in m if not g["needed"]]
    need = [g for g in m if g["needed"]]
    cat = "mixed" if vis and need else "visible" if vis else "needed" if need else "nogold"
    return cat, [(g["label"], round(g["start"], 1), round(g["end"], 1), g["importance"]) for g in m]


def _match_spec(specs, label, start):
    c = [s for s in specs if s["event_label"] == label]
    return min(c, key=lambda s: abs(float(s["start"]) - start)) if c else None


def variant(flipped):
    """a copy of the scored DEV renders where every gate-silenced spec with a flipped stretch is drawn again,
    and a 'kind of G' spec whose silencer G is drawn again is re-enabled unless another still-silent spec
    silences it (reason.py 1327-1343). Upper bound: re-enabled specs skip the later stage-5 steps."""
    from src.labels import same_source
    from src.stage5_cross_modal_analysis.reason import _overlap
    from types import SimpleNamespace as NS
    tmp = Path(tempfile.mkdtemp(prefix="ground_"))
    changed = []
    for d in PROPOSED.iterdir():
        if not (d / "augmentations.json").exists():
            continue
        (tmp / d.name).mkdir()
        if (d / "media.json").exists():
            shutil.copy(d / "media.json", tmp / d.name / "media.json")
        specs = json.loads((d / "augmentations.json").read_text(encoding="utf-8"))
        anypng = next((str(p) for p in (d / "augmentations").glob("*.png")), str(_ROOT / "README.md"))
        votes = json.loads((d / "gate_votes.json").read_text(encoding="utf-8")) if (d / "gate_votes.json").exists() else []
        by_spec = {}
        for k, v in enumerate(votes):
            by_spec.setdefault((v["label"], round(v["start"], 3)), []).append(k)
        reenabled = set()
        for (label, start), ks in by_spec.items():
            if any((d.name, k) in flipped for k in ks) and all(votes[k]["seen"] for k in ks):
                s = _match_spec(specs, label, start)
                if s is not None and not s.get("augment") and (s.get("reason") or "").startswith("source visible on screen"):
                    s["augment"] = True
                    s["image_path"] = s.get("image_path") or anypng
                    reenabled.add(id(s))
                    changed.append((d.name, label, round(float(s["start"]), 2), "flip"))
        gone = [s for s in specs if not s.get("augment") and (s.get("reason") or "").startswith("source visible on screen")]
        ns = lambda s: NS(spans=[tuple(x) for x in s.get("spans") or []], start=float(s["start"]), end=float(s["end"]))
        for s in specs:
            m = re.match(r"^a kind of (.+), whose source is visible - stay silent$", s.get("reason") or "")
            if not m or s.get("augment"):
                continue
            if any(same_source(s["event_label"], g["event_label"]) and _overlap(ns(s), ns(g)) for g in gone):
                continue
            s["augment"] = True
            s["image_path"] = s.get("image_path") or anypng
            changed.append((d.name, s["event_label"], round(float(s["start"]), 2), "kin of " + m.group(1)))
        for s in specs:
            if s.get("image_path") and not Path(s["image_path"]).exists():
                c = d / "augmentations" / Path(s["image_path"]).name
                s["image_path"] = str(c) if c.exists() else anypng
        (tmp / d.name / "augmentations.json").write_text(json.dumps(specs), encoding="utf-8")
    return tmp, changed


def score_dir(root, gold, dev):
    rows = {st: S.score_clip(gold[st], S.load_pictures(root, st, "proposed") or []) for st in dev}
    a = S.aggregate(list(rows.values()))
    return rows, {"hits": a["hits"], "needed": a["hits"] + a["misses"], "wrong": a["visible"] + a["cross"] + a["phantom"],
                  "visible": a["visible"], "cross": a["cross"], "phantom": a["phantom"], "P": round(a["P"], 3),
                  "R": round(a["R"], 3), "F1": round(a["F1"], 3), "cost": round(a["viewer_cost"], 3)}


def blind_draws(clip, label, start):
    f = BLIND / clip / "augmentations.json"
    if not f.exists():
        return None
    s = _match_spec(json.loads(f.read_text(encoding="utf-8")), label, start)
    return bool(s and s.get("augment") and abs(float(s["start"]) - start) < 1.0)


def score(args):
    gold, dev, subs = dev_stems()
    recs = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r["clip"] in set(dev)]
    cal = json.loads((OUTDIR / "calibration.json").read_text(encoding="utf-8")) if (OUTDIR / "calibration.json").exists() else {}
    for r in recs:
        r["cat"], r["gold"] = category(r, gold)
    n_seen = len(recs)
    checkers = [c for c in ("owlv2", "sam3") if all(c in pf for r in recs for pf in r["per_frame"])]
    _, base = score_dir(PROPOSED, gold, dev)
    out = {"rule": {"iou_min": IOU_MIN, "inside_min": INSIDE_MIN, "min_frames": MIN_FRAMES, "xs": XS,
                    "noun_prompt": NOUN_PROMPT, "box_prompt": BOX_PROMPT, "coord_calibration": cal},
           "dev_clips": len(dev), "seen_stretches": n_seen,
           "categories": {c: sum(1 for r in recs if r["cat"] == c) for c in ("visible", "needed", "mixed", "nogold")},
           "baseline_scored": base, "arms": {}, "stretches": []}
    # silenced specs (spec level): (clip, label, start) -> stretches
    specs_of = {}
    for r in recs:
        specs_of.setdefault((r["clip"], r["label"], round(r["start"], 3)), []).append(r)

    def spec_cat(rs):
        cs = {r["cat"] for r in rs}
        return "needed" if "needed" in cs or "mixed" in cs else "visible" if "visible" in cs else "nogold"

    def evaluate(flip_fn, name):
        flipped = {(r["clip"], r["k"]) for r in recs if flip_fn(r)}
        kept = [r for r in recs if (r["clip"], r["k"]) not in flipped]
        vis = [r for r in recs if r["cat"] == "visible"]
        need = [r for r in recs if r["cat"] in ("needed", "mixed")]
        nog = [r for r in recs if r["cat"] == "nogold"]
        spec_rows = []
        for key, rs in specs_of.items():
            # a spec is silenced only if ALL its stretches were seen; some of its stretches may not be in recs
            votes = json.loads((PROPOSED / key[0] / "gate_votes.json").read_text(encoding="utf-8"))
            allk = [k for k, v in enumerate(votes) if v["label"] == key[1] and round(v["start"], 3) == key[2]]
            silenced = all(votes[k]["seen"] for k in allk)
            if not silenced:
                continue
            drawn = any((key[0], r["k"]) in flipped for r in rs)
            spec_rows.append({"clip": key[0], "label": key[1], "start": key[2], "cat": spec_cat(rs), "drawn": drawn,
                              "blind_arm_draws": blind_draws(key[0], key[1], key[2]) if drawn else None})
        tmp, changed = variant(flipped)
        rows, sc = score_dir(tmp, gold, dev)
        shutil.rmtree(tmp, ignore_errors=True)
        pct = lambda a, b: round(100.0 * a / b, 1) if b else None
        sv = [s for s in spec_rows if s["cat"] == "visible"]; sn = [s for s in spec_rows if s["cat"] == "needed"]
        so = [s for s in spec_rows if s["cat"] == "nogold"]
        res = {
            "a_visible_stretches_kept": f"{sum(1 for r in vis if r in kept)}/{len(vis)}",
            "a_pct": pct(sum(1 for r in vis if r in kept), len(vis)),
            "a_spec_level_visible_specs_still_silent": f"{sum(1 for s in sv if not s['drawn'])}/{len(sv)}",
            "b_needed_stretches_flipped": f"{sum(1 for r in need if r not in kept)}/{len(need)}",
            "b_needed_specs_won_back": f"{sum(1 for s in sn if s['drawn'])}/{len(sn)}",
            "b_pct_specs": pct(sum(1 for s in sn if s["drawn"]), len(sn)),
            "nogold_stretches_kept_silent": f"{sum(1 for r in nog if r in kept)}/{len(nog)}",
            "nogold_specs_redrawn": f"{sum(1 for s in so if s['drawn'])}/{len(so)}",
            "c_seen_truly_visible_before": f"{len(vis)}/{n_seen}", "c_before_pct": pct(len(vis), n_seen),
            "c_seen_truly_visible_after": f"{sum(1 for r in kept if r['cat'] == 'visible')}/{len(kept)}",
            "c_after_pct": pct(sum(1 for r in kept if r["cat"] == "visible"), len(kept)),
            "dev_score": sc,
            "delta_vs_scored": {k: round(sc[k] - base[k], 3) for k in ("hits", "wrong", "F1", "cost")},
            "reenabled_specs": changed, "specs": spec_rows,
            "per_clip_changes": {st: {"hit": rows[st]["hit"], "wrong": rows[st]["visible"] + rows[st]["cross"] + rows[st]["phantom"]}
                                 for st in sorted({c[0] for c in changed})},
        }
        out["arms"][name] = res
        print(f"{name:14s} a {res['a_visible_stretches_kept']:>6s} ({res['a_pct']}%) spec-vis-silent {res['a_spec_level_visible_specs_still_silent']:>5s} | "
              f"b stretches {res['b_needed_stretches_flipped']:>5s} specs {res['b_needed_specs_won_back']:>4s} | "
              f"nogold kept {res['nogold_stretches_kept_silent']:>5s} | c {res['c_before_pct']}% -> {res['c_after_pct']}% | "
              f"hits {sc['hits']}/{sc['needed']} wrong {sc['wrong']} (v{sc['visible']} c{sc['cross']} p{sc['phantom']}) "
              f"F1 {sc['F1']:.3f} cost {sc['cost']:.2f}", flush=True)

    print(f"[dev] {len(dev)} clips, {n_seen} seen stretches {out['categories']}; scored row: {base}")
    evaluate(lambda r: False, "scored_repro")
    for c in checkers:
        for x in XS:
            evaluate(lambda r, c=c, x=x: not stays_seen(r, c, x), f"{c}@{x}")
    # side arms (not the declared rule): the checker alone, i.e. a stretch whose noun step said "none" keeps the
    # gate's verdict and only a named-but-unconfirmed source flips; and the VLM noun/box step alone, no checker
    for c in checkers:
        for x in XS:
            evaluate(lambda r, c=c, x=x: bool(r["noun"]) and not stays_seen(r, c, x), f"{c}@{x}_checker_only")
    evaluate(lambda r: not r["noun"] or not any(pf["vlm_boxes"] for pf in r["per_frame"]), "vlm_box_only")
    evaluate(lambda r: r["cat"] in ("needed", "mixed"), "oracle_flip")
    evaluate(lambda r: True, "flip_all")
    for r in recs:
        row = {"clip": r["clip"], "k": r["k"], "label": r["label"], "stretch": [round(v, 2) for v in r["stretch"]],
               "named": r["named"], "noun": r["noun"], "cat": r["cat"], "gold": r["gold"],
               "frames_with_vlm_box": sum(1 for pf in r["per_frame"] if pf["vlm_boxes"])}
        for c in checkers:
            row[c + "_best"] = round(max((d["score"] for pf in r["per_frame"] for d in pf.get(c, [])), default=0.0), 3)
            row[c + "_frames_ok"] = {str(x): sum(1 for pf in r["per_frame"] if frame_ok(pf, c, x)) for x in XS}
        out["stretches"].append(row)
    RESULT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", RESULT)


# ----------------------------------------------------------------------------- CPU: overlays
def overlays(args):
    from PIL import Image, ImageDraw
    gold, dev, subs = dev_stems()
    recs = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    PNGDIR.mkdir(parents=True, exist_ok=True)
    for r in recs:
        cat, _ = category(r, gold)
        tiles = []
        for i, pf in enumerate(r["per_frame"]):
            if i >= len(r["frames"]):
                continue
            img = Image.open(_ROOT / r["frames"][i]).convert("RGB")
            d = ImageDraw.Draw(img)
            lw = max(2, img.width // 300)
            for c, col in (("owlv2", (0, 200, 255)), ("sam3", (255, 0, 255))):
                for det in pf.get(c, []):
                    if det["score"] >= 0.1:
                        d.rectangle(det["box"], outline=col, width=lw)
                        d.text((det["box"][0] + 4, det["box"][3] - 14), f"{c[:3]} {det['score']:.2f}", fill=col)
            for v in pf["vlm_boxes"]:
                d.rectangle(v, outline=(255, 230, 0), width=lw * 2)
            d.text((6, 6), f"t={r['times'][i]:.1f}s", fill=(255, 255, 255))
            img.thumbnail((400, 400))
            tiles.append(img)
        if not tiles:
            continue
        w, h = tiles[0].size
        head = 40
        sheet = Image.new("RGB", (w * 3, h * 2 + head), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 3) * w, head + (i // 3) * h))
        ImageDraw.Draw(sheet).text((8, 6), f"{r['clip']} | {r['label']} {r['stretch'][0]:.1f}-{r['stretch'][1]:.1f}s | gold: {cat} | "
                                            f"gate named '{r['named']}' | noun '{r['noun']}' | yellow=VLM box, cyan=OWLv2, magenta=SAM3 (>=0.1)",
                                   fill=(255, 255, 255))
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{cat}__{r['clip']}__{r['label']}__{r['k']}")
        sheet.save(PNGDIR / f"{safe}.png")
    print("->", PNGDIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "score", "overlays"))
    a = ap.parse_args()
    {"collect": collect, "score": score, "overlays": overlays}[a.step](a)
