"""Round 38 E1 SYNC (docs/prereg_round13_detector_push.md): audio-visual SYNCHRONY as a vote for the stage-5 visibility
gate. Synchformer (Iashin et al. 2024, v-iashin/Synchformer, MIT, AudioSet checkpoint 24-01-04T16-39-21) predicts the
audio-visual temporal offset of a 5-s window over a 21-class grid (-2 .. +2 s, 0.2 s steps; class 10 = 0 s). A visible
source that is making the sound should give a confident near-zero offset; an off-screen source over a static picture
should give a flat / wrong offset. Earlier gate votes (SSL-SaN, PIC-SIM, GA, BOX) tested semantic match, not synchrony.

For every cached Qwen3.8-27B gate stretch (benchmark/gold/gate_gold/Qwen38-27B, all 139 gold clips) the clip's 5-s window
centred on the stretch (clamped to the clip; shorter clips padded on the right by cloning the last frame / silence) is
re-encoded as Synchformer's demo does (25 fps, 256 short side, 16 kHz mono) and scored with offset 0:
sync = p(|offset| <= 0.2 s) = the summed probability of the three grid classes -0.2 / 0 / +0.2 s (primary; the non-gold
smoke clip ev_kitchen_pan_drop put its in-sync peak at -0.2 s with p0 = 0.25 and p(+-0.2) = 0.996, so p(0) alone is
brittle to a one-class encode latency); p(0) alone is the reported secondary. Writes gate_gold/sync/<stem>.json (the
cached votes + a "sync" record per stretch, full 21-class vector kept).

    python benchmark/gold/sync_gate.py smoke [--device cpu]     # one NON-gold clip: offset 0 and +1.0 s controls
    python benchmark/gold/sync_gate.py run   [--device cuda]    # all cached clips, resumable
    python benchmark/gold/sync_gate.py score                     # CPU, laptop, current gold

Rule (fixed before any number): sound-level sync = MIN over its stretches (the gate silences only when every stretch is
seen); a failed window counts as sync < t. ONE threshold t = best balanced accuracy for "seen" on the NON-judge cached
clips (90 clips, highest t on ties), truth = CURRENT gold (seen = visible or obvious, re-derived per sound). On the DEV
judge clips (49): (a) vote4: sync >= t is a fourth vote next to name / a-b / desc, seen iff yes4 > no4, a 2-2 tie keeps
the shipped majority; (b) veto: seen iff shipped majority AND (sync >= t OR name = ab = desc = True).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import gate_gold as G
from benchmark.gold import detector_dry as DD

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "sync"
SUMMARY = G.OUT_DIR / "sync_summary.json"
REPO = Path(os.environ.get("SYNCHFORMER_DIR", str(Path.home() / "Synchformer")))
EXP = "24-01-04T16-39-21"
VFPS, AFPS, SIDE = 25, 16000, 256
WIN = 5.0            # cfg.data.crop_len_sec
CUT = 5.5            # cut a little more; TemporalCropAndOffset slices exactly 125 frames / 80000 samples from v_start 0
ZERO_CLASS = 10      # make_class_grid(-2, 2, 21)[10] == 0.0 (asserted at load)
KEY = "p_pm02"       # primary sync score; "p0" = secondary (report only)


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")


def duration(path) -> float:
    import av
    with av.open(str(path)) as c:
        return float(c.duration / av.time_base)


def cut_window(src, start: float, length: float, dst):
    """[start, start+length] of src -> dst, 25 fps, short side 256, even dims, 16 kHz mono; right-padded (clone / silence)"""
    vf = (f"fps={VFPS},scale=iw*{SIDE}/'min(iw,ih)':ih*{SIDE}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2,"
          f"tpad=stop_mode=clone:stop_duration={length + 1:.1f}")
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
           "-t", f"{length:.3f}", "-vf", vf, "-af", f"apad=whole_dur={length + 1:.1f}", "-ar", str(AFPS), "-ac", "1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def _train_utils():
    """Synchformer's scripts/train_utils.py by file path: its scripts/ is a namespace package, our repo's scripts/ is a
    regular package and always wins the name"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("synchformer_train_utils", REPO / "scripts" / "train_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["scripts.train_utils"] = mod            # utils/logger.py imports it by that name
    return mod


def load_model(device: str):
    import types
    import torch
    import transformers.pytorch_utils as PU
    # modeling_ast.py (a vendored, older HF file) imports two head-pruning helpers that newer transformers dropped;
    # pruning is never called at inference, so stubs are enough.
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(PU, name):
            setattr(PU, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError(name + " stub")))
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "get_head_mask"):        # removed in newer transformers; None mask -> per-layer None
        PreTrainedModel.get_head_mask = lambda self, hm, n, is_attention_chunked=False: [None] * n if hm is None else hm
    for d in (REPO, REPO / "model" / "modules" / "feat_extractors" / "visual",
              REPO / "model" / "modules" / "feat_extractors" / "audio"):     # motionformer_src / hf_src are imported bare
        sys.path.insert(0, str(d))
    from omegaconf import OmegaConf
    TU = _train_utils()
    get_model, get_transforms = TU.get_model, TU.get_transforms
    from dataset.transforms import make_class_grid
    cfg = OmegaConf.load(REPO / "logs" / "sync_models" / EXP / f"cfg-{EXP}.yaml")
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(".modules.feature_selector.", ".sync_model.")
    _, model = get_model(cfg, torch.device(device))
    ck = torch.load(REPO / "logs" / "sync_models" / EXP / f"{EXP}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    grid = make_class_grid(-cfg.data.max_off_sec, cfg.data.max_off_sec,
                           cfg.model.params.transformer.params.off_head_cfg.params.out_features)
    assert abs(float(grid[ZERO_CLASS])) < 1e-6 and len(grid) == 21, grid
    return model, cfg, get_transforms(cfg, ["test"])["test"], grid


def predict(model, cfg, tf, path, device: str, offset_sec: float = 0.0, v_start_i_sec: float = 0.0):
    """-> 21 softmax probabilities over the offset grid for the window starting at v_start_i_sec"""
    import torch
    from dataset.dataset_utils import get_video_and_audio
    prepare_inputs = _train_utils().prepare_inputs
    rgb, audio, meta = get_video_and_audio(str(path), get_meta=True)
    item = dict(video=rgb, audio=audio, meta=meta, path=str(path), split="test",
                targets={"v_start_i_sec": v_start_i_sec, "offset_sec": offset_sec})
    item = tf(item)
    batch = torch.utils.data.default_collate([item])
    aud, vid, _ = prepare_inputs(batch, torch.device(device))
    aud, vid = aud.float(), vid.float()                 # RGBToHalfToZeroOne gives half; autocast re-casts on cuda
    with torch.no_grad():
        with torch.autocast("cuda", enabled=(device.startswith("cuda") and bool(cfg.training.use_half_precision))):
            _, logits = model(vid, aud)
    return torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()


def window_start(centre: float, dur: float) -> float:
    return max(0.0, min(centre - WIN / 2, dur - WIN))


def sync_record(probs, grid):
    k = max(range(len(probs)), key=lambda i: probs[i])
    return {"p0": round(probs[ZERO_CLASS], 5), "p_pm02": round(sum(probs[ZERO_CLASS - 1:ZERO_CLASS + 2]), 5),
            "argmax_sec": round(float(grid[k]), 2), "pmax": round(probs[k], 5), "probs": [round(p, 5) for p in probs]}


def smoke(device: str, clip: str | None = None):
    """one NON-gold clip, the window at 1 s: offset 0 (expect argmax 0 s) and offset +1.0 (expect argmax +1 s)"""
    gold = {stem for _, stem, _ in G.gold_sounds()}
    cand = Path(clip) if clip else None
    assert cand is None or cand.stem not in gold, "smoke clip must not be a gold clip"
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted"):
        for p in sorted((_ROOT / "data" / "input" / "benchmark" / sub).glob("*.mp4")):
            if cand is None and p.stem not in gold and duration(p) >= 8.0:
                cand = p; break
        if cand:
            break
    assert cand, "no non-gold clip"
    print("smoke clip:", cand, f"{duration(cand):.1f}s")
    model, cfg, tf, grid = load_model(device)
    with tempfile.TemporaryDirectory() as td:
        w = cut_window(cand, 1.0, 7.5, Path(td) / "w.mp4")
        for off in (0.0, 1.0, -1.0):
            r = sync_record(predict(model, cfg, tf, w, device, offset_sec=off, v_start_i_sec=1.0), grid)
            print(f"offset {off:+.1f}: argmax {r['argmax_sec']:+.1f} s (p {r['pmax']:.3f})  p0 {r['p0']:.3f}  p+-0.2 {r['p_pm02']:.3f}")


def run(device: str, limit: int = 0, stems=None):
    OUT.mkdir(parents=True, exist_ok=True)
    model, cfg, tf, grid = load_model(device)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    done = 0
    tmp = Path(tempfile.mkdtemp(prefix="sync_"))
    for f in sorted(SRC.glob("*.json")):
        if stems and f.stem not in stems:
            continue
        if (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = DD.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", d["clip"]); continue
        dur = duration(p)
        for s in d["sounds"]:
            for st in s["stretches"]:
                a, b = float(st["start"]), float(st["end"])
                v0 = window_start((a + b) / 2, dur)
                try:
                    w = cut_window(p, v0, CUT, tmp / f"{f.stem}.mp4")
                    probs = predict(model, cfg, tf, w, device)
                    st["sync"] = {"v_start": round(v0, 3), **sync_record(probs, grid)}
                except Exception as e:                     # a failed window counts as sync < t
                    st["sync"] = {"v_start": round(v0, 3), "error": repr(e)[:200]}
                    print("FAIL", f.stem, s["label"], a, repr(e)[:200], flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f.stem, f"{dur:.1f}s", [(s["label"], [(st["sync"].get("p0"), st["sync"].get("argmax_sec")) for st in s["stretches"]])
                                      for s in d["sounds"]], flush=True)
        done += 1
        if limit and done >= limit:
            break
    shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------- scoring
def _votes(st):
    v = [st.get("name"), st.get("ab"), st.get("desc")]
    return sum(1 for x in v if x is True), sum(1 for x in v if x is False)


def sync_of(st, key: str = None):
    r = st.get("sync") or {}
    return r.get(key or KEY)


def sound_sync(s, key: str = None):
    vals = [sync_of(st, key) for st in s["stretches"]]
    return min(v if v is not None else -1.0 for v in vals) if vals else -1.0


def decide(stretches, rule: str, t: float, key: str = None) -> bool:
    """clip verdict: silent only if every stretch is seen. rules: majority (base), sync (alone), vote4 (a), veto (b)"""
    for st in stretches:
        yes, no = _votes(st)
        v = sync_of(st, key)
        ss = v is not None and v >= t
        maj = yes > no
        if rule == "majority":
            seen = maj
        elif rule == "sync":
            seen = ss
        elif rule == "vote4":
            yes4, no4 = yes + int(ss), no + int(not ss)
            seen = maj if yes4 == no4 else yes4 > no4
        elif rule == "veto":
            seen = maj and (ss or yes == 3)
        else:
            raise ValueError(rule)
        if not seen:
            return False
    return True


def _rates(rows, rule, t, key=None):
    seen_sil = n_seen = kept = n_needed = 0
    for s in rows:
        pred = decide(s["stretches"], rule, t, key)
        if s["_seen"]:
            n_seen += 1; seen_sil += int(pred)
        else:
            n_needed += 1; kept += int(not pred)
    return seen_sil, n_seen, kept, n_needed


def _quartiles(xs):
    import statistics
    xs = sorted(xs)
    if not xs:
        return None
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return {"n": len(xs), "min": round(xs[0], 4), "q1": round(q[0], 4), "median": round(q[1], 4), "q3": round(q[2], 4), "max": round(xs[-1], 4)}


def score():
    from benchmark.gold.box_gate import gold_index
    gold = gold_index()
    judge = set(G.JUDGE100.read_text().split())
    dev, cal = [], []
    n_fail = n_st = 0
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None:
                print("NO GOLD MATCH", f.stem, s["label"], s["start"]); continue
            if g["importance"] < 2:
                continue
            s["_seen"] = bool(g["seen"]); s["_stem"] = f.stem
            for st in s["stretches"]:
                n_st += 1; n_fail += sync_of(st) is None
            (dev if f.stem in judge else cal).append(s)
    print(f"stretches {n_st}, failed windows {n_fail}; DEV judge sounds {len(dev)}, calibration sounds {len(cal)}")
    # calibration on the non-judge clips: t with the best balanced accuracy for "seen" (highest t on ties)
    def calibrate(key):
        scores = sorted({sound_sync(s, key) for s in cal})
        best = None
        for t in scores + [scores[-1] + 1e-6]:
            a, na, b, nb = _rates(cal, "sync", t, key)
            bal = (a / na + b / nb) / 2
            if best is None or bal >= best[0]:
                best = (bal, t, a, na, b, nb)
        bal, t, a, na, b, nb = best
        print(f"calibration [{key}] (non-judge, {len(cal)} sounds): t = {t:.4f}  seen silenced {a}/{na}  needed kept {b}/{nb}  balanced {bal:.3f}")
        return {"t": t, "seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb, "balanced_acc": round(bal, 4), "n": len(cal)}
    out = {"key": KEY, "calibration": calibrate(KEY), "secondary_p0": {"calibration": calibrate("p0")},
           "failed_windows": n_fail, "stretches": n_st, "dev": {}, "flips": {}, "quartiles": {}}
    t = out["calibration"]["t"]; out["t"] = t
    t0 = out["secondary_p0"]["calibration"]["t"]
    for rule in ("majority", "sync", "vote4", "veto"):           # secondary (report only): p0 with its own t
        a, na, b, nb = _rates(dev, rule, t0, "p0")
        out["secondary_p0"][rule] = {"seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb}
    print("secondary p0 DEV:", json.dumps({k: v for k, v in out["secondary_p0"].items() if k != "calibration"}))
    for part, rows in (("dev", dev), ("cal", cal)):
        out["quartiles"][part] = {"seen": _quartiles([sound_sync(s) for s in rows if s["_seen"]]),
                                  "needed": _quartiles([sound_sync(s) for s in rows if not s["_seen"]])}
        # report only: the in-sync fraction (argmax class = 0 s) per group
        for grp, flag in (("seen", True), ("needed", False)):
            am = [st["sync"]["argmax_sec"] for s in rows if s["_seen"] == flag for st in s["stretches"] if sync_of(st) is not None]
            out["quartiles"][part][grp + "_p0"] = _quartiles([sound_sync(s, "p0") for s in rows if s["_seen"] == flag])
            out["quartiles"][part][grp + "_argmax0_frac"] = round(sum(abs(x) < 1e-6 for x in am) / len(am), 3) if am else None
        print(part, json.dumps(out["quartiles"][part]))
    base = {id(s): decide(s["stretches"], "majority", t) for s in dev}
    for rule in ("majority", "sync", "vote4", "veto"):
        a, na, b, nb = _rates(dev, rule, t)
        out["dev"][rule] = {"seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb}
        flips = []
        for s in dev:
            pred = decide(s["stretches"], rule, t)
            if pred != base[id(s)]:
                flips.append({"clip": s["_stem"], "label": s["label"], "start": s["start"], "gold_seen": s["_seen"],
                              "base_seen": base[id(s)], "new_seen": pred,
                              "sync": [sync_of(st) for st in s["stretches"]],
                              "argmax": [(st.get("sync") or {}).get("argmax_sec") for st in s["stretches"]],
                              "votes": [_votes(st) for st in s["stretches"]]})
        out["flips"][rule] = flips
        print(f"DEV judge  {rule:9s} seen silenced {a}/{na}  needed kept {b}/{nb}  flips {len(flips)}")
        for x in flips:
            print(f"   {'good' if x['new_seen'] == x['gold_seen'] else 'BAD '} {x['clip']} {x['label']} {x['start']:.1f}s gold_seen={x['gold_seen']} "
                  f"{x['base_seen']}->{x['new_seen']} sync={x['sync']} argmax={x['argmax']} votes={x['votes']}")
    b0 = out["dev"]["majority"]
    def _go(r):
        return ((r["seen_silenced"] >= b0["seen_silenced"] + 3 and r["needed_kept"] >= b0["needed_kept"] - 1) or
                (r["needed_kept"] >= b0["needed_kept"] + 2 and r["seen_silenced"] >= b0["seen_silenced"] - 1))
    out["GO"] = {r: _go(out["dev"][r]) for r in ("vote4", "veto")}
    print("Round 38 E1 SYNC:", "GO" if any(out["GO"].values()) else "STOP", out["GO"])
    SUMMARY.write_text(json.dumps(out, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------- Round 38 SYNC-2 (ADD-seen)
SUMMARY2 = G.OUT_DIR / "sync2_summary.json"


def decide2(stretches, rule: str, t_hi: float, t: float) -> bool:
    """add: majority OR (every stretch has sync, MIN >= t_hi); add+veto: the veto applied to the majority part only"""
    vals = [sync_of(st) for st in stretches]
    add = bool(vals) and all(v is not None for v in vals) and min(vals) >= t_hi
    maj = decide(stretches, "majority", t)
    if rule == "majority":
        return maj
    if rule == "add":
        return maj or add
    if rule == "add+veto":
        return decide(stretches, "veto", t) or add
    raise ValueError(rule)


def _rates2(rows, rule, t_hi, t):
    seen_sil = n_seen = kept = n_needed = 0
    for s in rows:
        pred = decide2(s["stretches"], rule, t_hi, t)
        if s["_seen"]:
            n_seen += 1; seen_sil += int(pred)
        else:
            n_needed += 1; kept += int(not pred)
    return seen_sil, n_seen, kept, n_needed


def _load_rows():
    from benchmark.gold.box_gate import gold_index
    gold = gold_index()
    judge = set(G.JUDGE100.read_text().split())
    dev, cal = [], []
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None or g["importance"] < 2:
                continue
            s["_seen"] = bool(g["seen"]); s["_stem"] = f.stem
            (dev if f.stem in judge else cal).append(s)
    return dev, cal


def pictures_on_flips(flips):
    """saved SHIP8 pictures (DEV part) whose classify match is a flipped gold sound"""
    os.environ.setdefault("TG_ARMS", "SHIP8")
    from benchmark.gold import btp_screen as B
    from benchmark.gold import cross_group as CG
    from benchmark.gold import score_per_sound as S
    B.ARM = "SHIP8"
    out = []
    by_stem = {}
    for x in flips:
        by_stem.setdefault(x["clip"], []).append(x)
    for part, st, gold, pics in B.parts():
        if part != "dev" or st not in by_stem:
            continue
        cls = CG.classify(gold, [p[:3] for p in pics])
        gs = sorted(gold, key=lambda g: g["start"])
        for lab, a, b, c, i in cls:
            if i is None:
                continue
            g = gs[i]
            for x in by_stem[st]:
                if abs(g["start"] - x["start"]) < 0.05 and S.same_family(x["label"], g["label"]):
                    out.append({"clip": st, "picture": [lab, round(a, 2), round(b, 2)], "class": c, "gold": [g["label"], g["start"]],
                                "flip": x["base_seen"] and "seen->not seen" or "not seen->seen", "gold_seen": x["gold_seen"]})
    return out


def score2():
    dev, cal = _load_rows()
    t = json.loads(SUMMARY.read_text(encoding="utf-8"))["t"]           # Round 38 veto threshold, unchanged
    # t_hi on the non-judge clips: smallest observed sync value with precision >= 0.95 for "seen" among sounds >= it
    vals = sorted({sound_sync(s) for s in cal})
    t_hi = None; prec = None; n_sel = 0
    for v in vals:
        sel = [s for s in cal if sound_sync(s) >= v]
        pr = sum(s["_seen"] for s in sel) / len(sel)
        if pr >= 0.95:
            t_hi, prec, n_sel = v, pr, len(sel); break
    print(f"t_hi = {t_hi} (precision {prec}, n >= t_hi: {n_sel} of {len(cal)}); veto t = {t:.4f}")
    out = {"t_hi": t_hi, "t_hi_precision": prec, "n_at_or_above_t_hi": n_sel, "t": t, "dev": {}, "flips": {}}
    if t_hi is None:
        out["void"] = True; print("rule void: no threshold reaches 0.95 precision")
        SUMMARY2.write_text(json.dumps(out, indent=1), encoding="utf-8"); return
    base = {id(s): decide2(s["stretches"], "majority", t_hi, t) for s in dev}
    for rule in ("majority", "add", "add+veto"):
        a, na, b, nb = _rates2(dev, rule, t_hi, t)
        out["dev"][rule] = {"seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb}
        flips = []
        for s in dev:
            pred = decide2(s["stretches"], rule, t_hi, t)
            if pred != base[id(s)]:
                flips.append({"clip": s["_stem"], "label": s["label"], "start": s["start"], "gold_seen": s["_seen"],
                              "base_seen": base[id(s)], "new_seen": pred, "sync": [sync_of(st) for st in s["stretches"]],
                              "votes": [_votes(st) for st in s["stretches"]]})
        out["flips"][rule] = flips
        print(f"DEV judge  {rule:9s} seen silenced {a}/{na}  needed kept {b}/{nb}  flips {len(flips)}")
        for x in flips:
            tag = "good" if x["new_seen"] == x["gold_seen"] else "BAD "
            print(f"   {tag} {x['clip']} {x['label']} {x['start']:.1f}s gold_seen={x['gold_seen']} "
                  f"{x['base_seen']}->{x['new_seen']} sync={x['sync']} votes={x['votes']}")
    b0 = out["dev"]["majority"]

    def _go(r):
        return ((r["seen_silenced"] >= b0["seen_silenced"] + 3 and r["needed_kept"] >= b0["needed_kept"] - 1) or
                (r["needed_kept"] >= b0["needed_kept"] + 2 and r["seen_silenced"] >= b0["seen_silenced"] - 1))
    out["GO"] = {r: _go(out["dev"][r]) for r in ("add", "add+veto")}
    print("Round 38 SYNC-2:", "GO" if any(out["GO"].values()) else "STOP", out["GO"])
    try:
        out["pictures"] = {r: pictures_on_flips(out["flips"][r]) for r in ("add", "add+veto")}
        for r in ("add", "add+veto"):
            print(f"SHIP8 pictures on {r} flips:", len(out["pictures"][r]))
            for x in out["pictures"][r]:
                print("   ", x)
    except Exception as e:
        out["pictures_error"] = repr(e)[:300]; print("pictures: not available from saved data:", repr(e)[:300])
    SUMMARY2.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "score", "score2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--clip", default=None, help="smoke: a NON-gold clip path")
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(a.device, a.clip)
    elif a.cmd == "run":
        run(a.device, a.limit, set(a.stems) if a.stems else None)
    elif a.cmd == "score2":
        score2()
    else:
        score()


if __name__ == "__main__":
    main()
