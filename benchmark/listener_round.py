"""Amendment 25 (docs/prereg_v4.md): an audio "listener" (Qwen3-Omni-30B-A3B-Instruct) asked yes/no about each candidate span
that cell E (or E-AND) adds over the shipped stack. Stage-4 logic and cost C from benchmark/detector_round2.py.

    python benchmark/listener_round.py pool   [--set calib|heldout]   # candidate spans + hit/false/neutral labels (CPU)
    python benchmark/listener_round.py score  [--set calib|heldout]   # listener scores + null control (GPU, one model load)
    python benchmark/listener_round.py screen                          # AUROC gate on hit vs false (the 280)
    python benchmark/listener_round.py fit                             # E+L, E-AND+L on the 280, the pick
    python benchmark/listener_round.py heldout --cell NAME             # the pick vs shipped on the held-out set
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from src.labels import canonical

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OUT = _ROOT / "benchmark" / "listener_round.json"
POOL = {s: _ROOT / "benchmark" / f"listener_pool_{s}.json" for s in ("calib", "heldout")}
VOCAB = json.loads((_ROOT / "benchmark" / "gold" / "depictable_vocab.json").read_text(encoding="utf-8"))["families"]
QUESTION = "Is the sound of {} present in this recording? Answer yes or no."


def key(cid, e):
    return f"{cid}|{canonical(e.label)}|{e.start:.2f}|{e.end:.2f}"


def added(cid, base_ev, spec):
    """spans of the cell that the shipped stack does not show (no same-family shipped span overlapping)"""
    ev = D.stack(cid, **spec)
    return ev, [e for e in D._ev(ev) if not any(canonical(x.label) == canonical(e.label)
                                                and min(x.end, e.end) - max(x.start, e.start) > 0 for x in base_ev)]


def label_of(c, e):
    fam = [g for g in c["events"] if R.E._same(e.label, g["label"])]
    if any(g["consequential"] and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for g in fam):
        return "hit"
    if not any(min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in fam):
        return "false"
    return "neutral"


def build_pool(set_name):
    R.use_set(set_name)
    cl = D.usable()
    rng = random.Random(0)
    items = {}
    for c in cl:
        base = D.stack(c["id"])
        labelled = {canonical(g["label"]) for g in c["events"]}
        for cell in ("E", "E-AND"):
            _ev, add = added(c["id"], base, D.cells()[cell])
            for e in add:
                k = key(c["id"], e)
                if k in items:
                    items[k]["cells"].append(cell); continue
                absent = [v for v in VOCAB if canonical(v) not in labelled]
                items[k] = {"clip": c["id"], "family": canonical(e.label), "start": e.start, "end": e.end,
                            "label": label_of(c, e), "cells": [cell], "null_family": rng.choice(absent)}
    POOL[set_name].write_text(json.dumps({"set": set_name, "clips": len(cl), "items": items}, indent=1), encoding="utf-8")
    n = {k: sum(v["label"] == k for v in items.values()) for k in ("hit", "false", "neutral")}
    print(f"[pool] {set_name}: {len(cl)} clips, {len(items)} candidate spans {n} -> {POOL[set_name]}")


def score(set_name, limit=0):
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    R.use_set(set_name)
    d = json.loads(POOL[set_name].read_text(encoding="utf-8"))
    items = d["items"]
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    print(f"[score] model loaded in {time.time() - t0:.0f} s; yes ids {yes_ids} no ids {no_ids}", flush=True)
    audio_cache = {}

    def wav(cid):
        if cid not in audio_cache:
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(R.E.VIDEOS / f"{cid}.mp4"), "-vn", "-ac", "1",
                                  "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
            audio_cache.clear(); audio_cache[cid] = np.frombuffer(raw, np.float32).copy()
        return audio_cache[cid]

    def ask(w, fam):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": QUESTION.format(fam.lower())}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16) if hasattr(inp, "to") else inp
        with torch.inference_mode():
            logits = model.thinker(**inp).logits[0, -1].float()
        return float(logits[yes_ids].max() - logits[no_ids].max())

    done = 0
    for k, it in sorted(items.items()):
        if "score" in it:
            continue
        w = wav(it["clip"])
        a, b = int(max(0.0, it["start"] - 1.0) * 16000), int(min(len(w) / 16000, it["end"] + 1.0) * 16000)
        seg = w[a:max(b, a + 16000)]
        it["score"] = ask(seg, it["family"])
        it["null_score"] = ask(seg, it["null_family"])
        done += 1
        if done == 10:
            print(f"[score] load gate: 10 windows in {time.time() - t0:.0f} s wall (limit 7200 s)", flush=True)
        if done % 50 == 0:
            POOL[set_name].write_text(json.dumps(d, indent=1), encoding="utf-8")
            print(f"[score] {done}/{len(items)}", flush=True)
        if limit and done >= limit:
            break
    POOL[set_name].write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"[score] done {done} -> {POOL[set_name]}")


def auroc(s, y):
    pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def screen(log):
    R.use_set("calib")
    d = json.loads(POOL["calib"].read_text(encoding="utf-8"))
    its = [v for v in d["items"].values() if v["label"] in ("hit", "false") and "score" in v]
    y = np.array([v["label"] == "hit" for v in its])
    sl = np.array([v["score"] for v in its])
    sp = []
    for v in its:                                          # PANNs same-family peak within 1 s, on the same spans
        pf = R.load(R.PANNS / f"{v['clip']}.npz")
        class _E: pass
        e = _E(); e.label, e.start, e.end = v["family"], v["start"], v["end"]
        sp.append(D.peak_near(pf, e))
    sp = np.array(sp)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].all() or (~y[i]).all():
            continue
        bs.append(auroc(sl[i], y[i]))
    allv = [v for v in d["items"].values() if "null_score" in v]
    log["screen"] = {"n_hit": int(y.sum()), "n_false": int((~y).sum()), "auroc_listener": auroc(sl, y),
                     "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "auroc_panns": auroc(sp, y),
                     "yes_rate_true_family": float(np.mean([v["score"] > 0 for v in allv])),
                     "yes_rate_absent_family": float(np.mean([v["null_score"] > 0 for v in allv]))}
    s = log["screen"]
    s["pass"] = s["ci"][0] >= 0.70 and s["auroc_listener"] > s["auroc_panns"]
    print(f"[screen] hit {s['n_hit']} / false {s['n_false']}: listener AUROC {s['auroc_listener']:.3f} "
          f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}], PANNs {s['auroc_panns']:.3f}; yes on the asked family "
          f"{s['yes_rate_true_family']:.0%}, on an absent family {s['yes_rate_absent_family']:.0%} -> pass {s['pass']}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def cell_with_listener(base_cell, theta, scores):
    spec = D.cells()[base_cell]

    def run(cid):
        base = D.stack(cid)
        ev, add = added(cid, base, spec)
        drop = {id(e) for e in add if scores.get(key(cid, e), -1e9) < theta}
        return [e for e in ev if id(e) not in drop]
    return run


def run_fn(cl, fn):
    rows = [D.clip_cost(c, fn(c["id"])) for c in cl]
    minutes = sum(c["duration"] for c in cl) / 60.0
    return rows, {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
                  "fp_per_min": sum(r["fp"] for r in rows) / minutes,
                  "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / max(1, sum(r["n_conseq"] for r in rows))}


def fit(log):
    assert log.get("screen", {}).get("pass"), "screen failed: no listener cell is scored"
    R.use_set("calib")
    cl = D.usable()
    d = json.loads(POOL["calib"].read_text(encoding="utf-8"))
    scores = {k: v["score"] for k, v in d["items"].items() if "score" in v}
    grid = [float(np.percentile(list(scores.values()), q)) for q in (20, 40, 60, 80)]
    base = run_fn(cl, lambda cid: D.stack(cid))[1]
    res = {"shipped": base}
    for bc in ("E", "E-AND"):
        best = min(((th, run_fn(cl, cell_with_listener(bc, th, scores))[1]) for th in grid), key=lambda x: x[1]["C_overlap"])
        res[f"{bc}+L"] = {**best[1], "theta": best[0]}
    ok = {k: v for k, v in res.items() if k != "shipped" and v["C_overlap"] < base["C_overlap"] and v["C_onset"] < base["C_onset"]}
    pick = min(ok, key=lambda k: ok[k]["C_overlap"]) if ok else None
    log["grid"] = grid; log["fit"] = res; log["pick"] = pick
    for k, v in res.items():
        print(f"{k:9s} C-overlap {v['C_overlap']:.3f} C-onset {v['C_onset']:.3f} recall {v['recall_overlap']:.1%} false/min {v['fp_per_min']:.2f} {v.get('theta', '')}")
    print(f"PICK: {pick}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(cell, log):
    assert cell == log.get("pick"), "only the committed pick goes to held-out"
    R.use_set("heldout")
    cl = D.usable()
    d = json.loads(POOL["heldout"].read_text(encoding="utf-8"))
    scores = {k: v["score"] for k, v in d["items"].items() if "score" in v}
    b_rows, b = run_fn(cl, lambda cid: D.stack(cid))
    c_rows, c = run_fn(cl, cell_with_listener(cell.replace("+L", ""), log["fit"][cell]["theta"], scores))
    d_ov = [x["C_overlap"] - y["C_overlap"] for x, y in zip(c_rows, b_rows)]
    m, lo, hi = D.boot(d_ov)
    log["heldout"] = {"cell": cell, "clips": len(cl), "shipped": b, "cell_summary": c, "dC_overlap": [m, lo, hi],
                      "dC_onset": D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(c_rows, b_rows)]), "pass": hi < 0}
    print(f"[heldout] {len(cl)} clips; dC-overlap {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if hi < 0 else 'fail'}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "score", "screen", "fit", "heldout"))
    ap.add_argument("--set", default="calib", choices=("calib", "heldout"))
    ap.add_argument("--cell", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if a.step == "pool":
        build_pool(a.set)
    elif a.step == "score":
        score(a.set, a.limit)
    elif a.step == "screen":
        screen(log)
    elif a.step == "fit":
        fit(log)
    else:
        heldout(a.cell, log)


if __name__ == "__main__":
    main()
