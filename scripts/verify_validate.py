"""Validate the picture check (PICTURE_VERIFY, src/stage6_visual_augmentation/verify.py) BEFORE it redraws anything.

  1. draw  -- ~33 deliberately wrong pictures with the shipped style (Qwen-Image-2512, subject + RULES_TAIL, the
              screening negative), each paired with a real spec from the 82 inspector pictures, so the options asked
              are exactly the production ones (a flock of crows where "crowd" is intended, a desk bell for a fire-alarm
              bell, a megaphone for a car horn, a kettle for steam, random objects, two pictures with painted words);
  2. check -- the check alone on the 82 existing shipped pictures and on the wrong set, twice (option order salt 0 =
              the order the redraw loop uses, salt 1 = a second shuffle, for stability), plus the VLM's own yes/no
              answer on "letters or words?" for comparison with OCR.

    python scripts/verify_validate.py --phase draw     (generator only)
    python scripts/verify_validate.py --phase check    (VLM + EasyOCR)
Output: data/work/verify_validation/{wrong/*.png, wrong.json, results.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

OUT = _ROOT / "data" / "work" / "verify_validation"
TAGS = {"DEV": "dev_monocap_v31", "TEST": "test_final_v33", "sliceB": "sliceB_v32"}

# (what is drawn, the intended sound: label, source, subject as the pipeline had it)
WRONG = [
    ("a flock of black crows", "Crowd", "Crowd", "crowd"),
    ("a flock of pigeons flying", "Crowd", "Crowd", "concert crowd"),
    ("a brass desk bell on a hotel counter being rung", "Alarm", "Alarm", "Alarm bell ringing loudly"),
    ("a large church bell swinging", "Alarm", "Fire alarm", "Fire alarm ringing loudly"),
    ("a vintage alarm clock ringing", "Alarm", "Alarm", "Alarm bell ringing loudly"),
    ("a red megaphone", "Vehicle", "Vehicle horn, car horn, honking", "A car honking its horn"),
    ("a car with a megaphone mounted on its roof", "Honk", "Honk", "A car honking"),
    ("a city bus with a loudspeaker megaphone on its side", "Vehicle", "Toot", "Bus honking its horn"),
    ("a brass trumpet being played", "Vehicle", "Vehicle horn, car horn, honking", "A car honking its horn"),
    ("a steaming stainless steel kettle", "Steam", "Steam", "Steam rising from a kettle"),
    ("a teapot with steam coming out of its spout", "Steam", "Steam", "Steam rising from a kettle"),
    ("a man shouting with his mouth wide open", "Typing", "Typing", "typing"),
    ("a security camera dome on a ceiling", "Alarm", "Smoke detector, smoke alarm", "ceiling smoke detector sounding"),
    ("a cat meowing", "Dog", "Bow-wow", "A dog barking"),
    ("an acoustic guitar", "Chainsaw", "Chainsaw", "Chainsaw cutting through wood"),
    ("a toaster with bread popping up", "Computer keyboard", "Computer keyboard", "Fingers typing on computer keyboard"),
    ("a red apple", "Bird", "Chirp, tweet", "A bird chirping"),
    ("a bicycle", "Train", "Train", "A train moving on tracks"),
    ("a wooden house", "Siren", "Civil defense siren", "Civil defense siren sounding"),
    ("a teddy bear", "Baby cry, infant cry", "Baby cry, infant cry", "baby crying"),
    ("a snowy mountain landscape", "Thunder", "Thunder", "Thunderstorm clouds rumbling"),
    ("a coffee mug", "Glass", "Shatter", "Glass shattering into pieces"),
    ("a table lamp", "Explosion", "Boom", "Explosion bursting in the air"),
    ("a potted plant", "Footsteps", "Run", "run"),
    ("a goldfish", "Bird", "Crowing, cock-a-doodle-doo", "Rooster crowing"),
    ("a stop sign with the word STOP", "Alarm", "Car alarm", "Car alarm sounding"),
    ("an open book", "Bell", "Church bell", "A church bell swinging"),
    ("a violin", "Gunshot", "Machine gun", "A machine gun firing"),
    ("a wooden chair", "Rain", "Rain", "Rain falling on a window"),
    ("a horse galloping", "Siren", "Ambulance (siren)", "Ambulance driving with siren"),
    ("an owl hooting", "Bird", "Pigeon, dove", "Pigeon cooing"),
    # text cases: the right thing, with words painted on it (the OCR check must catch these)
    ("a red fire alarm pull station on a wall with a big sign reading FIRE EXIT PULL HERE", "Alarm", "Fire alarm",
     "Fire alarm ringing loudly"),
    ("a delivery truck with the words FRESH BREAD painted in large letters on its side", "Vehicle", "Truck",
     "Truck driving down the road"),
]


def subject_drawn(spec) -> str:
    """What _final_picture drew for this spec (template or subject) -- the check is asked about the same thing."""
    from benchmark.gold.gen_screen import TEMPLATES
    source = spec.source or spec.event_label
    key = source if source in TEMPLATES else (spec.event_label if spec.event_label in TEMPLATES else None)
    return TEMPLATES[key] if key else spec.subject


def phase_draw():
    from benchmark.gold.gen_screen import RULES_TAIL, negative_for
    from src.stage6_visual_augmentation import _diffusion_image, unload_generator
    (OUT / "wrong").mkdir(parents=True, exist_ok=True)
    man = []
    for i, (drawn, label, source, subject) in enumerate(WRONG):
        p = OUT / "wrong" / f"w{i:02d}.png"
        seed = zlib.crc32(drawn.encode()) & 0x7FFFFFFF
        t0 = time.time()
        if not p.exists():
            ok = _diffusion_image(p, drawn + RULES_TAIL, (1024, 1024), model=config.GEN_MODEL, device="cuda",
                                  seed=seed, negative=negative_for(drawn) or " ")
            assert ok, drawn
        man.append({"i": i, "drawn": drawn, "label": label, "source": source, "subject": subject, "seed": seed,
                    "image": str(p.relative_to(_ROOT))})
        print(f"  w{i:02d} {time.time() - t0:5.1f}s {drawn}  (for {label} / {subject})", flush=True)
    (OUT / "wrong.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    unload_generator()


def items():
    """The 82 shipped pictures + the wrong set, as (key, group, image path, spec, subject)."""
    out = []
    for split, tag in TAGS.items():
        root = _ROOT / "data" / "work" / f"shipped_{tag}"
        for d in sorted(root.iterdir()):
            if not (d / "repaint.json").exists():
                continue
            specs = {s["index"]: s for s in json.loads((d / "augmentations.json").read_text(encoding="utf-8"))}
            for r in json.loads((d / "repaint.json").read_text(encoding="utf-8")):
                s = specs[r["index"]]
                spec = SimpleNamespace(index=s["index"], event_label=s["event_label"], source=s.get("source", ""),
                                       subject=s.get("subject", ""), start=float(s["start"]))
                out.append((f"{split}/{d.name}/{r['index']}", "shipped", d / "augmentations" / r["image"], spec,
                            subject_drawn(spec), d.name))
    for w in json.loads((OUT / "wrong.json").read_text(encoding="utf-8")):
        spec = SimpleNamespace(index=w["i"], event_label=w["label"], source=w["source"], subject=w["subject"],
                               start=float(w["i"]))
        out.append((f"wrong/w{w['i']:02d}", "wrong", _ROOT / w["image"], spec, subject_drawn(spec), "wrong"))
    return out


def phase_check(out_name: str = "results.json"):
    from PIL import Image
    from src.stage6_visual_augmentation import verify as V
    res = []
    t0 = time.time()
    for key, group, path, spec, subject, tag in items():
        r0 = V.check(path, spec, subject, salt=0, tag=tag, see=True)
        img = Image.open(path).convert("RGB").resize((768, 768))
        mc1 = V.ask_mc(img, spec, subject, salt=1, tag=tag)
        row = {"key": key, "group": group, "label": spec.event_label, "source": spec.source, "subject": subject,
               "intended": r0["mc"]["intended"], "entry": r0["mc"]["entry"],
               "pass1": {"options": r0["mc"]["options"], "picked": r0["mc"]["picked"], "answer": r0["mc"]["answer"],
                         "ok": r0["mc"]["ok"]},
               "pass2": {"options": mc1["options"], "picked": mc1["picked"], "answer": mc1["answer"], "ok": mc1["ok"]},
               "ocr": r0["text"], "vlm_text": V.vlm_text(img), "saw": r0.get("saw", ""),
               "ok1": r0["ok"], "ok2": bool(mc1["ok"] and r0["text"]["ok"])}
        res.append(row)
        print(f"{key:55s} {'OK ' if row['ok1'] else 'REJ'} {'OK ' if row['ok2'] else 'REJ'} "
              f"picked={row['pass1']['picked']!r} / {row['pass2']['picked']!r} text={row['ocr']['words']} "
              f"vlm_text={row['vlm_text']} saw={row['saw']!r}", flush=True)
    (OUT / out_name).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[validate] {len(res)} pictures checked in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["draw", "check", "all"], default="all")
    ap.add_argument("--lookalike-vlm", action="store_true",
                    help="PICTURE_LOOKALIKE_VLM on (the AMBIGUOUS look-alikes written by the VLM), into --out-name")
    ap.add_argument("--out-name", default="results.json")
    a = ap.parse_args()
    print("[cfg]", config.use_shipped(), flush=True)
    if a.lookalike_vlm:
        config.PICTURE_LOOKALIKE_VLM = True
        assert a.out_name != "results.json", "keep the round-2 baseline: write the look-alike run to another file"
    print("[cfg] PICTURE_LOOKALIKE_VLM", config.PICTURE_LOOKALIKE_VLM, "->", a.out_name, flush=True)
    print("[cfg] VLM", config.VLM_MODEL, "GEN", config.GEN_MODEL, flush=True)
    config.DEVICE = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase in ("draw", "all"):
        phase_draw()
    if a.phase in ("check", "all"):
        phase_check(a.out_name)


if __name__ == "__main__":
    main()
