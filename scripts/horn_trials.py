"""Horn wordings on the sliceB Honk clip (and one true car-horn clip), plus a preview of the VLM-written looks and
look-alikes (PICTURE_LOOK_VLM / PICTURE_LOOKALIKE_VLM) for every AMBIGUOUS entry. Adam, 28 Sept 2026: the Honk picture
failed 5 tries (trumpet / megaphone horns stuck on a car) and ended as a HONK word card.

Each wording is drawn exactly as a redraw try from try 3 on is drawn (subject + RULES_TAIL + "no text", the screening
negative + the text negatives, Qwen-Image-2512), with two seeds (the clip's try-1 seed and its try-3 seed), then checked
with verify.check against the clip's own subject. Picked by eye and by the checker.

    python scripts/horn_trials.py
Output: data/work/horn_trials/{<clip>_<wording>_s<k>.png, results.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

OUT = _ROOT / "data" / "work" / "horn_trials"
CLIPS = [("sliceB_v32", "dQbrfZDBq_8_30000", 1), ("sliceB_v32", "97aoiaWwRVk_20000", 0)]
WORDINGS = {
    "wheel": "a car seen from the front, the driver's hand pressing the horn in the middle of the steering wheel",
    "waves": "a car seen from the front with curved sound-wave lines coming out of its front grille",
    "goose": "a goose honking with its beak open",            # the ontology's maker of AudioSet "Honk"
}


def spec_of(tag, clip, idx):
    d = _ROOT / "data" / "work" / f"shipped_v_{tag}" / clip
    r = next(x for x in json.loads((d / "augmentations.json").read_text(encoding="utf-8")) if x["index"] == idx)
    return SimpleNamespace(index=r["index"], event_label=r["event_label"], source=r.get("source", ""),
                           subject=r.get("subject", ""), start=float(r["start"]))


def main():
    print("[cfg]", config.use_shipped(), flush=True)
    config.DEVICE = "cuda"
    config.PICTURE_VERIFY = True
    config.PICTURE_MAKER = True            # a goose subject is not checked as a horn (verify.AMBIGUOUS unless_subject)
    OUT.mkdir(parents=True, exist_ok=True)
    from benchmark.gold.gen_screen import RULES_TAIL, seed_of, negative_for as screen_negative
    from src.stage6_visual_augmentation import _diffusion_image
    from src.stage6_visual_augmentation import verify as V

    # 1. preview: the VLM's look and look-alikes for every entry (text only)
    prev = []
    for e in V.AMBIGUOUS:
        if not e.get("maker"):
            continue
        src = {"horn": "Vehicle horn, car horn, honking", "smoke_alarm": "Smoke detector, smoke alarm",
               "alarm_bell": "Fire alarm", "steam": "Steam", "crowd": "Crowd", "typing": "Typing",
               "rattle": "Rattle (instrument)"}.get(e["name"], e["name"])
        sp = SimpleNamespace(index=0, event_label=src, source=src, subject="", start=0.0)
        maker = e["maker"].format(veh="car")
        prev.append({"entry": e["name"], "maker": maker, "describe": V.describe(maker, e["noise"], sp),
                     "lookalikes": V.lookalikes(maker, e["intended"].format(veh="car")), "handwritten_rewrite": e.get("rewrite"),
                     "handwritten_confusions": e["confusions"]})

    # 2. the horn wordings
    rows = []
    for tag, clip, idx in CLIPS:
        spec = spec_of(tag, clip, idx)
        base = seed_of({"clip": clip, "label": spec.event_label, "start": float(spec.start)}) + 1
        words = dict(WORDINGS)
        if spec.event_label != "Honk":
            words.pop("goose")
        d = V.describe("car", "horn", spec)
        if d:
            words["vlm"] = d
        for name, w in words.items():
            for k, seed in enumerate((base, base + 2000)):
                p = OUT / f"{clip}_{name}_s{k}.png"
                prompt = w + RULES_TAIL + ", no text, no letters, no signs"
                neg = ", ".join(x for x in (screen_negative(w), "text, letters, words, writing, sign, label, logo") if x)
                t0 = time.time()
                if not p.exists():
                    assert _diffusion_image(p, prompt, config.RESOLUTION, model=config.GEN_MODEL, device="cuda",
                                            seed=seed, negative=neg)
                subj = w if name == "goose" else spec.subject
                res = V.check(p, spec, subj, salt=0, tag=clip)
                rows.append({"clip": clip, "label": spec.event_label, "wording": name, "text": w, "seed": seed,
                             "image": p.name, "ok": res["ok"], "picked": res["mc"]["picked"],
                             "intended": res["mc"]["intended"], "options": res["mc"]["options"],
                             "ocr": res["text"]["words"], "saw": res.get("saw", "")})
                print(f"  {clip} {name:6s} s{k}: {'OK ' if res['ok'] else 'REJ'} picked={res['mc']['picked']!r} "
                      f"saw={res.get('saw', '')!r} ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "results.json").write_text(json.dumps({"preview": prev, "trials": rows}, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print("[horn] done", flush=True)


if __name__ == "__main__":
    main()
