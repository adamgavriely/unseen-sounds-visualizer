"""A picture verifier: the VLM is told which sound the picture is meant to show and says whether a viewer
who cannot hear would recognise it at a glance (docs/prereg_verifier.md, written before this runs).

Adam, 2026-09-25: "clearly a VLM could see most images here are bad; no need for a human to say this."
Two blind checkers (asked "what is this?") failed calibration; this one answers the easier, named
question Adam himself answered in round 1 ("would a viewer understand this sound?", family name shown),
so it is calibrated on those 170 yes/no answers. If it passes, it screens generators; Adam rates only the
final confirmation.

    python benchmark/gold/verifier.py calibrate      # round-1 pictures vs Adam's yes/no
    python benchmark/gold/verifier.py run --items items.json --out verdicts.json   # [{"image", "sound"}]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

MODEL = "google/gemma-4-31B-it"
PROMPT = ("This picture is shown beside a video, for about one second, to a viewer who cannot hear. It is "
          "meant to tell them that this sound is happening: {sound}."
          + chr(10) +
          "Would that viewer recognise, at a glance, that the picture means this sound? Judge the picture as "
          "it is, not as it was meant to be. Answer with one word: yes or no.")


def plain(label: str) -> str:
    return label.split("(")[0].split(",")[0].strip().lower()


def load():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODEL)
    mdl = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    return mdl, proc


def verdict(mdl, proc, image_path, sound):
    import torch
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((384, 384))     # the size the viewer sees
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": PROMPT.format(sound=plain(sound))}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        gen = mdl.generate(**inp, max_new_tokens=4, do_sample=False)
    ans = proc.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    return "yes" if re.match(r"\W*yes", ans) else ("no" if re.match(r"\W*no", ans) else "?"), ans


def cmd_calibrate(a):
    from benchmark.gold.score_answers import _kappa
    key = json.loads((_ROOT / "data/work/rate_pictures_KEY_do_not_open_before_rating.json").read_text(encoding="utf-8"))
    adam = json.loads((_ROOT / "benchmark/gold/pictures/adam_ratings_2026-09-24.json").read_text(encoding="utf-8"))["answers"]
    mdl, proc = load()
    rows = []
    for code, v in sorted(key.items()):
        a_ = adam.get(code)
        if a_ not in ("yes", "no"):
            continue                                     # "unsure" and unanswered are left out, as declared
        img = _ROOT / "data/work/rate_pictures/img" / f"{code}.jpg"
        v_, raw = verdict(mdl, proc, img, v["label"])
        rows.append({"code": code, "arm": v["arm"], "sound": v["label"], "adam": a_, "vlm": v_, "raw": raw})
        print(f"  {code} {v['label'][:22]:22s} adam {a_:3s} vlm {v_}", flush=True)
    out = _ROOT / "benchmark/gold/pictures/verifier_calibration_round1.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ad = np.array([r["adam"] == "yes" for r in rows]); vm = np.array([r["vlm"] == "yes" for r in rows])
    catch = float((~vm[~ad]).mean()); falsrej = float((~vm[ad]).mean()); kap = _kappa(list(ad), list(vm))
    ok = catch >= 0.70 and falsrej <= 0.15 and kap >= 0.5
    print(f"\n{len(rows)} pictures: catches {catch:.0%} of Adam's no (bar 70%), rejects {falsrej:.0%} of his yes "
          f"(bar 15%), kappa {kap:.2f} (bar 0.5), agreement {float((ad == vm).mean()):.0%} -> "
          f"{'PASS: may screen generators' if ok else 'FAIL: Adam screens'}")


def cmd_run(a):
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    mdl, proc = load()
    out = []
    for it in items:
        v_, raw = verdict(mdl, proc, it["image"], it["sound"])
        out.append(dict(it, vlm=v_, raw=raw))
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} verdicts, {sum(o['vlm'] == 'yes' for o in out)} yes -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "run"])
    ap.add_argument("--items")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"calibrate": cmd_calibrate, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
