"""GP-4 point 4: the pairwise picture check — the last automatic instrument (docs/panel_2026-09-25_scene_prompt.md).

Two pictures meant for the same sound; "which would a viewer who cannot hear more likely recognise as this
sound at a glance — first, second, or neither?", asked in both orders so that a lenient or strict bias
cancels. GLM-4.6V-Flash: not Gemma (the judge), not Qwen3.8 (in the pipeline).

Calibration (bars fixed in the plan): every discordant pair Adam produced — the same sound in two arms, one
he got right / said yes to, the other not — from round 1 (name shown, yes/no) and round 2 (blind naming).
Pass: >= 75 % agreement AND >= 90 % order-swap consistency; "neither" and an inconsistent pair count as
disagreement. Reported with a bootstrap CI. If it passes it may drop only the bottom generator arm, by more
than its own CI; if it fails, automatic screening is closed.

    python benchmark/gold/pairwise.py calibrate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "zai-org/GLM-4.6V-Flash"
PROMPT = ("Both pictures are meant to show this sound to a viewer who cannot hear it: {sound}. Each would be "
          "shown beside a video for about one second. Which picture would that viewer more likely recognise "
          "as this sound at a glance? Answer with exactly one word: first, second, or neither.")


def plain(label: str) -> str:
    return label.split("(")[0].split(",")[0].strip().lower()


def pairs_round1():
    key = json.loads((_ROOT / "data/work/rate_pictures_KEY_do_not_open_before_rating.json").read_text(encoding="utf-8"))
    ans = json.loads((_ROOT / "benchmark/gold/pictures/adam_ratings_2026-09-24.json").read_text(encoding="utf-8"))["answers"]
    by = {}
    for code, v in key.items():
        if ans.get(code) in ("yes", "no"):
            by.setdefault((v["clip"], v["label"], v["i"]), []).append((code, ans[code] == "yes"))
    out = []
    for (clip, label, i), lst in by.items():
        for (c1, g1), (c2, g2) in combinations(lst, 2):
            if g1 != g2:
                good, bad = (c1, c2) if g1 else (c2, c1)
                out.append({"round": 1, "sound": label,
                            "good": str(_ROOT / "data/work/rate_pictures/img" / f"{good}.jpg"),
                            "bad": str(_ROOT / "data/work/rate_pictures/img" / f"{bad}.jpg")})
    return out


def pairs_round2():
    scored = json.loads((_ROOT / "benchmark/gold/pictures/adam_answers_round2_scored.json").read_text(encoding="utf-8"))
    arms = json.loads((_ROOT / "benchmark/gold/pictures/rate_pictures2_ARM_KEY.json").read_text(encoding="utf-8"))
    first = {}
    for code in sorted(scored):
        k = (arms[code]["arm"], arms[code]["i"])
        first.setdefault(k, code)                    # repeats excluded
    by = {}
    for (arm, i), code in first.items():
        by.setdefault(i, []).append((code, scored[code]["class"] in ("correct", "narrower"), scored[code]["source"]))
    out = []
    for i, lst in by.items():
        for (c1, g1, s), (c2, g2, _) in combinations(lst, 2):
            if g1 != g2:
                good, bad = (c1, c2) if g1 else (c2, c1)
                out.append({"round": 2, "sound": s,
                            "good": str(_ROOT / "data/work/rate_pictures2/img" / f"{good}.jpg"),
                            "bad": str(_ROOT / "data/work/rate_pictures2/img" / f"{bad}.jpg")})
    return out


def ask(mdl, proc, img_a, img_b, sound):
    import torch
    from PIL import Image
    ims = [Image.open(p).convert("RGB").resize((384, 384)) for p in (img_a, img_b)]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "text", "text": PROMPT.format(sound=plain(sound))}]}]
    try:
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=160, do_sample=False)
    raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    raw = raw.split("</" + "think>")[-1].replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip().lower()
    m = re.search(r"\b(first|second|neither)\b", raw)
    return (m.group(1) if m else "?"), raw[:80]


def cmd_calibrate(a):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    pairs = pairs_round1() + pairs_round2()
    proc = AutoProcessor.from_pretrained(MODEL)
    mdl = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    rows = []
    for n, p in enumerate(pairs, 1):
        v1, r1 = ask(mdl, proc, p["good"], p["bad"], p["sound"])      # good shown first
        v2, r2 = ask(mdl, proc, p["bad"], p["good"], p["sound"])      # good shown second
        consistent = (v1, v2) in {("first", "second"), ("second", "first"), ("neither", "neither")}
        agree = v1 == "first" and v2 == "second"
        rows.append(dict(p, order1=v1, order2=v2, consistent=consistent, agree=agree, raw1=r1, raw2=r2))
        print(f"  {n}/{len(pairs)} r{p['round']} {plain(p['sound'])[:18]:18s} {v1:7s} {v2:7s} "
              f"{'AGREE' if agree else ''}", flush=True)
    out = _ROOT / "benchmark/gold/pictures/pairwise_calibration.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ag = np.array([r["agree"] for r in rows], float)
    co = np.array([r["consistent"] for r in rows], float)
    rng = np.random.default_rng(0)
    bs = [ag[rng.integers(0, len(ag), len(ag))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ok = ag.mean() >= 0.75 and co.mean() >= 0.90
    for rnd in (1, 2):
        sub = [r for r in rows if r["round"] == rnd]
        if sub:
            print(f"round {rnd}: {len(sub)} pairs, agreement {np.mean([r['agree'] for r in sub]):.0%}, "
                  f"consistency {np.mean([r['consistent'] for r in sub]):.0%}")
    print(f"\n{len(rows)} discordant pairs: agreement {ag.mean():.0%} [{lo:.0%}, {hi:.0%}] (bar 75 %), order-swap "
          f"consistency {co.mean():.0%} (bar 90 %) -> {'PASS: may drop the bottom generator arm' if ok else 'FAIL: automatic screening closed'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate"])
    a = ap.parse_args()
    cmd_calibrate(a)


if __name__ == "__main__":
    main()
