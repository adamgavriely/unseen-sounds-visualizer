"""GP-4 screening: draw the picture-DEV sounds with each candidate generator, same text, same seed rule
(docs/panel_2026-09-25_scene_prompt.md, docs/panel_2026-09-25_generator.md). Runs in ~/venvs/gen
(torch 2.7, diffusers from main), not in msproj, so it imports nothing from the pipeline.

    gate:   python benchmark/gold/gen_screen.py --model q21 --gate
            (five fixed sounds; seconds per picture, peak VRAM, negative prompt used or not -> gate json)
    draw:   python benchmark/gold/gen_screen.py --model q21 --bench data/work/picture_bench_fresh \
                --subjects V31 --arm G_q21
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
PLAIN_TAIL = ", plain white background, clearly visible"          # stage6.plain_prompt, unchanged
NEGATIVE_V3 = ["scenery", "landscape", "background scene", "room interior", "street", "buildings",
               "sky background", "text", "letters", "watermark"]
RGBA_WRAP = ("This is an RGBA image with transparency. {subject}. The image has alpha channel and the "
             "background is transparent.")                           # the Qwen-Image-2.1 card's own format

MODELS = {
    # key: (repo, pipeline class, call kwargs, uses negative prompt)
    "q2512": ("Qwen/Qwen-Image-2512", "QwenImagePipeline",
              {"num_inference_steps": 50, "true_cfg_scale": 4.0}, True),
    "q21": ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline", {"num_inference_steps": 40}, False),
    "q21rgba": ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline", {"num_inference_steps": 40}, False),
}


def seed_of(item) -> int:                                   # benchmark/gold/picture_bench.py, unchanged
    return zlib.crc32(f"{item['clip']}|{item['label']}|{item['start']:.2f}".encode()) & 0x7FFFFFFF


def negative_for(subject: str) -> str:                      # stage6.negative_for, unchanged
    words = {w.strip(",.").lower() for w in subject.split()}
    weather = words & {"cloud", "clouds", "lightning", "rain", "raining", "storm", "thunder",
                       "thunderstorm", "wind", "snow", "hail"}
    keep = [n for n in NEGATIVE_V3 if not (set(n.split()) & words)
            and not (weather and n in ("sky background", "landscape"))]
    return ", ".join(keep)


def ink(img) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return float((a.min(axis=2) < 238).mean())


def load(model_key):
    import torch
    import diffusers
    repo, cls, _, _ = MODELS[model_key]
    pipe = getattr(diffusers, cls).from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def draw(pipe, model_key, subject, seed):
    import torch
    from PIL import Image
    _, _, kw, uses_neg = MODELS[model_key]
    kw = dict(kw, width=1024, height=1024, generator=torch.Generator("cuda").manual_seed(seed))
    if model_key == "q21rgba":
        prompt = RGBA_WRAP.format(subject=subject)
    else:
        prompt = subject + PLAIN_TAIL
    if uses_neg:
        kw["negative_prompt"] = negative_for(subject) or " "
    img = pipe(prompt=prompt, **kw).images[0]
    if img.mode == "RGBA":                     # the viewer sees it on white
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("RGB"), prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--bench", default="data/work/picture_bench_fresh")
    ap.add_argument("--subjects", default="V31")
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed-offset", type=int, default=1)      # GP-4: seed_of(item) + 1 for every arm
    a = ap.parse_args()
    import torch
    bench = _ROOT / a.bench
    specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
    subj = json.loads((bench / f"subjects_{a.subjects}.json").read_text(encoding="utf-8"))
    if a.gate:                                 # five fixed sounds: the first five by seed
        specs = sorted(specs, key=seed_of)[:5]
    out = bench / (a.arm or f"gate_{a.model}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = load(a.model)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    manifest, times = [], []
    for it in specs:
        subject = subj.get(str(it["i"]), {}).get("subject") or it["subject"]
        seed = seed_of(it) + a.seed_offset
        t = time.time()
        img, prompt = draw(pipe, a.model, subject, seed)
        if ink(img) < 0.05:                    # the blank guard, as in the pipeline: redraw once
            img, prompt = draw(pipe, a.model, subject, seed + 1)
        times.append(time.time() - t)
        img.save(out / f"{it['i']:02d}.png")
        manifest.append({"i": it["i"], "subject": subject, "prompt": prompt, "seed": seed,
                         "ink": round(ink(img), 4), "seconds": round(times[-1], 1)})
        print(f"  {a.model} {it['i']:2d} {times[-1]:5.1f}s {subject}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    gate = {"model": a.model, "repo": MODELS[a.model][0], "load_seconds": round(load_s, 1),
            "seconds_per_picture_median": round(float(np.median(times)), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            "negative_prompt_used": MODELS[a.model][3], "gpu": torch.cuda.get_device_name(0),
            "pass_20s": float(np.median(times)) <= 20.0}
    print(json.dumps(gate))
    if a.gate:
        (_ROOT / "benchmark/gold/pictures" / f"gate_{a.model}.json").write_text(json.dumps(gate, indent=1),
                                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
